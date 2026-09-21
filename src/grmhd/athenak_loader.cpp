#include "grmhd/athenak_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

namespace kpolaris {
namespace {

using RealT = DefaultReal;

constexpr int P_RHO = 0;
constexpr int P_U1 = 1;
constexpr int P_U2 = 2;
constexpr int P_U3 = 3;
constexpr int P_UU = 4;
constexpr int P_B1 = 5;
constexpr int P_B2 = 6;
constexpr int P_B3 = 7;

struct AthenaKHeader {
    double time = 0.0;
    int loc_size = 8;
    int var_size = 4;
    int nvar = 0;
    int mb_nx1 = 0;
    int mb_nx2 = 0;
    int mb_nx3 = 0;
    double spin = 0.9375;
    double gamma = 13.0 / 9.0;
    double mesh_x1min = -1.0;
    double mesh_x1max = 1.0;
    double mesh_x2min = -1.0;
    double mesh_x2max = 1.0;
    double mesh_x3min = -1.0;
    double mesh_x3max = 1.0;
    std::streamoff data_start = 0;
    std::vector<std::string> variables;
    std::unordered_map<std::string, std::string> params;
};

struct MeshBlock {
    int level = 0;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
};

struct MeshBlockIndex {
    int nx = 0, ny = 0, nz = 0;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    std::vector<int> offsets;
    std::vector<int> candidates;
};

struct AthenaKDump {
    AthenaKHeader header;
    std::vector<MeshBlock> blocks;
    std::vector<float> prims; // contiguous block/variable/k/j/i storage
    MeshBlockIndex index;
};

size_t checked_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("AthenaK payload size overflow");
    return a * b;
}

void report_timing(int enabled, const char* name, double seconds) {
    if (enabled) {
        std::cout << "timing " << name << ' ' << seconds << " s\n";
    }
}

std::string trim_copy(std::string s) {
    size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) ++first;
    size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) --last;
    return s.substr(first, last - first);
}

std::vector<std::string> split_words(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    std::string word;
    while (in >> word) out.push_back(word);
    return out;
}

int read_i32(std::ifstream& in) {
    std::int32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("failed reading AthenaK int32");
    return static_cast<int>(v);
}

double read_float_or_double(std::ifstream& in, int bytes) {
    if (bytes == 8) {
        double v = 0.0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!in) throw std::runtime_error("failed reading AthenaK float64");
        return v;
    }
    if (bytes == 4) {
        float v = 0.0f;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!in) throw std::runtime_error("failed reading AthenaK float32");
        return static_cast<double>(v);
    }
    throw std::runtime_error("unsupported AthenaK floating-point byte size");
}

std::string value_without_comment(const std::string& raw) {
    std::string s = raw;
    const size_t hash = s.find('#');
    if (hash != std::string::npos) s.erase(hash);
    return trim_copy(s);
}

void parse_header_block(const std::string& text, AthenaKHeader& header) {
    std::string group;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line.front() == '<' && line.back() == '>') {
            group = line.substr(1, line.size() - 2);
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim_copy(line.substr(0, eq));
        const std::string value = value_without_comment(line.substr(eq + 1));
        if (key.empty() || value.empty()) continue;
        const std::string full_key = group.empty() ? key : group + "/" + key;
        header.params[full_key] = value;
    }
}

std::string param_value(const AthenaKHeader& header, const std::string& key,
                        const std::string& def) {
    auto it = header.params.find(key);
    if (it == header.params.end()) return def;
    return it->second;
}

int param_int(const AthenaKHeader& header, const std::string& key, int def) {
    return std::stoi(param_value(header, key, std::to_string(def)));
}

double param_double(const AthenaKHeader& header, const std::string& key, double def) {
    return std::stod(param_value(header, key, std::to_string(def)));
}

AthenaKHeader read_athenak_header(std::ifstream& in, const std::string& path) {
    AthenaKHeader header;
    std::string line;
    std::getline(in, line);
    if (line != "Athena binary output version=1.1") {
        throw std::runtime_error("unsupported AthenaK binary header in " + path + ": " + line);
    }
    std::getline(in, line);
    const size_t eq_pre = line.find('=');
    if (eq_pre == std::string::npos) throw std::runtime_error("invalid AthenaK preheader size line");
    const int preheader_size = std::stoi(trim_copy(line.substr(eq_pre + 1)));
    if (preheader_size < 1 || preheader_size > 4096)
        throw std::runtime_error("invalid AthenaK preheader size");
    for (int i = 0; i < preheader_size - 1; ++i) {
        std::getline(in, line);
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim_copy(line.substr(0, eq));
        const std::string value = value_without_comment(line.substr(eq + 1));
        header.params[key] = value;
        if (key == "time") header.time = std::stod(value);
        if (key == "size of location") header.loc_size = std::stoi(value);
        if (key == "size of variable") header.var_size = std::stoi(value);
    }
    std::getline(in, line);
    const size_t eq_nvar = line.find('=');
    if (eq_nvar == std::string::npos) throw std::runtime_error("invalid AthenaK variable count line");
    header.nvar = std::stoi(trim_copy(line.substr(eq_nvar + 1)));
    std::getline(in, line);
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
        header.variables = split_words(line.substr(colon + 1));
    }
    std::getline(in, line);
    const size_t eq_header = line.find('=');
    if (eq_header == std::string::npos) throw std::runtime_error("invalid AthenaK header offset line");
    const int header_offset = std::stoi(trim_copy(line.substr(eq_header + 1)));
    const std::streamoff header_begin = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streamoff header_file_size = in.tellg();
    if (header_offset < 0 || header_begin < 0 || header_file_size < header_begin ||
        static_cast<std::streamoff>(header_offset) > header_file_size - header_begin)
        throw std::runtime_error("invalid AthenaK header offset");
    in.seekg(header_begin, std::ios::beg);
    std::string header_text(static_cast<size_t>(header_offset), '\0');
    in.read(header_text.data(), header_offset);
    if (!in) throw std::runtime_error("failed reading AthenaK PAR_DUMP header");
    parse_header_block(header_text, header);
    header.data_start = in.tellg();

    header.mb_nx1 = param_int(header, "meshblock/nx1", 0);
    header.mb_nx2 = param_int(header, "meshblock/nx2", 0);
    header.mb_nx3 = param_int(header, "meshblock/nx3", 0);
    header.spin = param_double(header, "coord/a", header.spin);
    header.gamma = param_double(header, "mhd/gamma", header.gamma);
    header.mesh_x1min = param_double(header, "mesh/x1min", header.mesh_x1min);
    header.mesh_x1max = param_double(header, "mesh/x1max", header.mesh_x1max);
    header.mesh_x2min = param_double(header, "mesh/x2min", header.mesh_x2min);
    header.mesh_x2max = param_double(header, "mesh/x2max", header.mesh_x2max);
    header.mesh_x3min = param_double(header, "mesh/x3min", header.mesh_x3min);
    header.mesh_x3max = param_double(header, "mesh/x3max", header.mesh_x3max);
    if (!(header.nvar >= 8 && header.mb_nx1 > 1 && header.mb_nx2 > 1 && header.mb_nx3 > 1)) {
        throw std::runtime_error("invalid AthenaK nvar/meshblock dimensions");
    }
    if (!std::isfinite(header.time) || !std::isfinite(header.spin) ||
        std::abs(header.spin) > 1.0 || !std::isfinite(header.gamma) || header.gamma <= 1.0)
        throw std::runtime_error("invalid AthenaK time/spin/gamma");
    return header;
}

AthenaKDump read_athenak_dump(const std::string& path, int timing) {
    Kokkos::Timer timer;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open AthenaK dump: " + path);
    AthenaKDump dump;
    dump.header = read_athenak_header(in, path);
    report_timing(timing, "athenak_header", timer.seconds());
    timer.reset();

    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(dump.header.data_start, std::ios::beg);
    if (dump.header.loc_size != 4 && dump.header.loc_size != 8)
        throw std::runtime_error("unsupported AthenaK floating-point byte size");
    if (dump.header.var_size != 4 && dump.header.var_size != 8)
        throw std::runtime_error("unsupported AthenaK variable size");
    const size_t cells_per_block = checked_product(checked_product(
        static_cast<size_t>(dump.header.mb_nx1), static_cast<size_t>(dump.header.mb_nx2)),
        static_cast<size_t>(dump.header.mb_nx3));
    const size_t values_per_block = checked_product(static_cast<size_t>(dump.header.nvar), cells_per_block);
    const size_t payload_bytes = checked_product(values_per_block, static_cast<size_t>(dump.header.var_size));
    const size_t metadata_bytes = 10u * sizeof(std::int32_t) + 6u * static_cast<size_t>(dump.header.loc_size);
    if (payload_bytes > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) - metadata_bytes)
        throw std::runtime_error("AthenaK payload size overflow");
    const size_t block_bytes = metadata_bytes + payload_bytes;
    if (file_size < dump.header.data_start ||
        (file_size - dump.header.data_start) % static_cast<std::streamoff>(block_bytes) != 0)
        throw std::runtime_error("AthenaK truncated meshblock payload or trailing data");
    const size_t nblocks = static_cast<size_t>((file_size - dump.header.data_start) /
                                              static_cast<std::streamoff>(block_bytes));
    if (nblocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("AthenaK meshblock count exceeds device index capacity");
    dump.blocks.reserve(nblocks);
    // Read directly into final host storage. Staging transfers ownership instead
    // of retaining one allocation per block and making a second full dump copy.
    dump.prims.resize(checked_product(nblocks, values_per_block));
    std::vector<double> wide_values;
    if (dump.header.var_size == 8) wide_values.resize(values_per_block);
    for (size_t mb = 0; mb < nblocks; ++mb) {
        const int si = read_i32(in);
        const int ei = read_i32(in);
        const int sj = read_i32(in);
        const int ej = read_i32(in);
        const int sk = read_i32(in);
        const int ek = read_i32(in);
        const auto nx1 = static_cast<std::int64_t>(ei) - si + 1;
        const auto nx2 = static_cast<std::int64_t>(ej) - sj + 1;
        const auto nx3 = static_cast<std::int64_t>(ek) - sk + 1;
        if (nx1 != dump.header.mb_nx1 || nx2 != dump.header.mb_nx2 || nx3 != dump.header.mb_nx3) {
            throw std::runtime_error("AthenaK meshblock size mismatch");
        }
        (void)read_i32(in);
        (void)read_i32(in);
        (void)read_i32(in);
        MeshBlock block;
        block.level = read_i32(in);
        block.xmin = read_float_or_double(in, dump.header.loc_size);
        block.xmax = read_float_or_double(in, dump.header.loc_size);
        block.ymin = read_float_or_double(in, dump.header.loc_size);
        block.ymax = read_float_or_double(in, dump.header.loc_size);
        block.zmin = read_float_or_double(in, dump.header.loc_size);
        block.zmax = read_float_or_double(in, dump.header.loc_size);
        if (!std::isfinite(block.xmin) || !std::isfinite(block.xmax) ||
            !std::isfinite(block.ymin) || !std::isfinite(block.ymax) ||
            !std::isfinite(block.zmin) || !std::isfinite(block.zmax) ||
            !(block.xmax > block.xmin && block.ymax > block.ymin && block.zmax > block.zmin))
            throw std::runtime_error("invalid AthenaK meshblock extent");
        float* const destination = dump.prims.data() + mb * values_per_block;
        if (dump.header.var_size == 4) {
            in.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(payload_bytes));
        } else {
            in.read(reinterpret_cast<char*>(wide_values.data()), static_cast<std::streamsize>(payload_bytes));
            if (!in) throw std::runtime_error("failed reading AthenaK meshblock data");
            for (size_t n = 0; n < values_per_block; ++n)
                destination[n] = static_cast<float>(wide_values[n]);
        }
        if (!in) throw std::runtime_error("failed reading AthenaK meshblock data");
        for (size_t n = 0; n < values_per_block; ++n)
            if (!std::isfinite(destination[n]))
                throw std::runtime_error("non-finite AthenaK primitive field");
        dump.blocks.push_back(std::move(block));
    }
    report_timing(timing, "athenak_binary_read", timer.seconds());
    return dump;
}

int clamp_bin_index(double x, double xmin, double dx, int n) {
    int i = static_cast<int>(std::floor((x - xmin) / dx));
    return std::clamp(i, 0, n - 1);
}

size_t flat_bin_index(const MeshBlockIndex& idx, int ix, int iy, int iz) {
    return (static_cast<size_t>(iz) * static_cast<size_t>(idx.ny) + static_cast<size_t>(iy)) *
           static_cast<size_t>(idx.nx) + static_cast<size_t>(ix);
}

void build_meshblock_index(AthenaKDump& dump, int timing) {
    Kokkos::Timer timer;
    if (dump.blocks.empty()) {
        throw std::runtime_error("AthenaK dump contains no meshblocks");
    }
    MeshBlockIndex idx;
    idx.xmin = dump.blocks.front().xmin; idx.xmax = dump.blocks.front().xmax;
    idx.ymin = dump.blocks.front().ymin; idx.ymax = dump.blocks.front().ymax;
    idx.zmin = dump.blocks.front().zmin; idx.zmax = dump.blocks.front().zmax;
    for (const MeshBlock& b : dump.blocks) {
        idx.xmin = std::min(idx.xmin, b.xmin); idx.xmax = std::max(idx.xmax, b.xmax);
        idx.ymin = std::min(idx.ymin, b.ymin); idx.ymax = std::max(idx.ymax, b.ymax);
        idx.zmin = std::min(idx.zmin, b.zmin); idx.zmax = std::max(idx.zmax, b.zmax);
    }
    const double root = std::cbrt(static_cast<double>(dump.blocks.size()));
    const int nb = std::clamp(static_cast<int>(std::ceil(root * 4.0)), 16, 96);
    idx.nx = nb;
    idx.ny = nb;
    idx.nz = nb;
    idx.dx = (idx.xmax - idx.xmin) / static_cast<double>(idx.nx);
    idx.dy = (idx.ymax - idx.ymin) / static_cast<double>(idx.ny);
    idx.dz = (idx.zmax - idx.zmin) / static_cast<double>(idx.nz);
    if (!(idx.dx > 0.0 && idx.dy > 0.0 && idx.dz > 0.0)) {
        throw std::runtime_error("invalid AthenaK meshblock index extent");
    }
    const size_t nbin = static_cast<size_t>(idx.nx) * static_cast<size_t>(idx.ny) * static_cast<size_t>(idx.nz);
    std::vector<std::vector<int>> bins(nbin);
    for (size_t mb = 0; mb < dump.blocks.size(); ++mb) {
        const MeshBlock& b = dump.blocks[mb];
        const int ix0 = clamp_bin_index(b.xmin, idx.xmin, idx.dx, idx.nx);
        const int iy0 = clamp_bin_index(b.ymin, idx.ymin, idx.dy, idx.ny);
        const int iz0 = clamp_bin_index(b.zmin, idx.zmin, idx.dz, idx.nz);
        const int ix1 = clamp_bin_index(std::nextafter(b.xmax, b.xmin), idx.xmin, idx.dx, idx.nx);
        const int iy1 = clamp_bin_index(std::nextafter(b.ymax, b.ymin), idx.ymin, idx.dy, idx.ny);
        const int iz1 = clamp_bin_index(std::nextafter(b.zmax, b.zmin), idx.zmin, idx.dz, idx.nz);
        for (int iz = iz0; iz <= iz1; ++iz) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                for (int ix = ix0; ix <= ix1; ++ix) {
                    bins[flat_bin_index(idx, ix, iy, iz)].push_back(static_cast<int>(mb));
                }
            }
        }
    }
    // Keep all candidates. Device lookup chooses the highest AMR level so refined
    // AthenaK blocks override coarse blocks in overlapping regions.
    idx.offsets.resize(nbin + 1, 0);
    size_t total = 0;
    for (size_t b = 0; b < nbin; ++b) {
        idx.offsets[b] = static_cast<int>(total);
        total += bins[b].size();
    }
    idx.offsets[nbin] = static_cast<int>(total);
    idx.candidates.reserve(total);
    for (const auto& bin : bins) {
        idx.candidates.insert(idx.candidates.end(), bin.begin(), bin.end());
    }
    dump.index = std::move(idx);
    if (timing) {
        const double avg_candidates = total > 0 ? static_cast<double>(total) / static_cast<double>(nbin) : 0.0;
        std::cout << "timing athenak_meshblock_index " << timer.seconds() << " s"
                  << " bins " << dump.index.nx << 'x' << dump.index.ny << 'x' << dump.index.nz
                  << " blocks " << dump.blocks.size()
                  << " avg_candidates " << avg_candidates << "\n";
    }
}




} // namespace

DefaultReal read_athenak_dump_time(const std::string& dump_path) {
    if (dump_path.empty()) {
        throw std::runtime_error("AthenaK dump path is empty");
    }
    std::ifstream in(dump_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open AthenaK dump: " + dump_path);
    }
    const AthenaKHeader header = read_athenak_header(in, dump_path);
    return DefaultReal(header.time);
}


AthenaKStagedDump read_athenak_staged_dump(const AthenaKLoadOptions& opt) {
    if (opt.dump_path.empty()) {
        throw std::runtime_error("--athenak_dump is required for --model=athenak");
    }
    AthenaKDump dump = read_athenak_dump(opt.dump_path, opt.timing);
    build_meshblock_index(dump, opt.timing);
    Kokkos::Timer flatten_timer;

    AthenaKStagedDump staged;
    staged.nblocks = static_cast<int>(dump.blocks.size());
    staged.nvar = dump.header.nvar;
    staged.nx1 = dump.header.mb_nx1;
    staged.nx2 = dump.header.mb_nx2;
    staged.nx3 = dump.header.mb_nx3;
    staged.spin = static_cast<DefaultReal>(dump.header.spin);
    staged.gamma = static_cast<DefaultReal>(dump.header.gamma);

    const size_t nblocks = dump.blocks.size();
    staged.extents.resize(nblocks * 6u);
    staged.cell_geom.resize(nblocks * 6u);
    staged.prims = std::move(dump.prims);
    staged.levels.resize(nblocks);
    staged.index_offsets = std::move(dump.index.offsets);
    staged.index_candidates = std::move(dump.index.candidates);

    for (size_t mb = 0; mb < nblocks; ++mb) {
        const MeshBlock& b = dump.blocks[mb];
        staged.extents[mb * 6u + 0u] = static_cast<DefaultReal>(b.xmin);
        staged.extents[mb * 6u + 1u] = static_cast<DefaultReal>(b.xmax);
        staged.extents[mb * 6u + 2u] = static_cast<DefaultReal>(b.ymin);
        staged.extents[mb * 6u + 3u] = static_cast<DefaultReal>(b.ymax);
        staged.extents[mb * 6u + 4u] = static_cast<DefaultReal>(b.zmin);
        staged.extents[mb * 6u + 5u] = static_cast<DefaultReal>(b.zmax);
        staged.cell_geom[mb * 6u + 0u] = static_cast<DefaultReal>(b.xmin);
        staged.cell_geom[mb * 6u + 1u] = static_cast<DefaultReal>(b.ymin);
        staged.cell_geom[mb * 6u + 2u] = static_cast<DefaultReal>(b.zmin);
        staged.cell_geom[mb * 6u + 3u] = static_cast<DefaultReal>(dump.header.mb_nx1) / static_cast<DefaultReal>(b.xmax - b.xmin);
        staged.cell_geom[mb * 6u + 4u] = static_cast<DefaultReal>(dump.header.mb_nx2) / static_cast<DefaultReal>(b.ymax - b.ymin);
        staged.cell_geom[mb * 6u + 5u] = static_cast<DefaultReal>(dump.header.mb_nx3) / static_cast<DefaultReal>(b.zmax - b.zmin);
        staged.levels[mb] = b.level;
    }

    staged.index_nx = dump.index.nx;
    staged.index_ny = dump.index.ny;
    staged.index_nz = dump.index.nz;
    staged.index_xmin = static_cast<DefaultReal>(dump.index.xmin);
    staged.index_xmax = static_cast<DefaultReal>(dump.index.xmax);
    staged.index_ymin = static_cast<DefaultReal>(dump.index.ymin);
    staged.index_ymax = static_cast<DefaultReal>(dump.index.ymax);
    staged.index_zmin = static_cast<DefaultReal>(dump.index.zmin);
    staged.index_zmax = static_cast<DefaultReal>(dump.index.zmax);
    staged.index_dx = static_cast<DefaultReal>(dump.index.dx);
    staged.index_dy = static_cast<DefaultReal>(dump.index.dy);
    staged.index_dz = static_cast<DefaultReal>(dump.index.dz);
    report_timing(opt.timing, "athenak_host_flatten", flatten_timer.seconds());
    return staged;
}

AthenaKDirectRadiationModel<DefaultReal> materialize_athenak_direct_model_from_staged(const AthenaKStagedDump& staged,
                                                                                      const AthenaKLoadOptions& opt) {
    Kokkos::Timer total_timer;
    Kokkos::Timer pack_timer;

    AthenaKDirectRadiationModel<RealT> model;
    model.nblocks = staged.nblocks;
    model.nvar = staged.nvar;
    model.nx1 = staged.nx1;
    model.nx2 = staged.nx2;
    model.nx3 = staged.nx3;
    model.spin = static_cast<RealT>(staged.spin);
    model.gam = opt.gamma > DefaultReal(0) ? static_cast<RealT>(opt.gamma) : static_cast<RealT>(staged.gamma);
    const RealT rh = RealT(1) + Kokkos::sqrt(max_val(RealT(0), RealT(1) - model.spin * model.spin));
    model.r_in = opt.resample_r_in > RealT(0) ? opt.resample_r_in : RealT(0.9) * rh;
    model.r_out = opt.resample_r_out > RealT(0) ? opt.resample_r_out : RealT(1000);
    model.freq_cgs = opt.freq;
    model.M_unit = opt.M_unit;
    model.mbh_solar = opt.mbh_solar;
    model.trat_small = opt.trat_small;
    model.trat_large = opt.trat_large;
    model.beta_crit = opt.beta_crit;
    model.sigma_cut = opt.sigma_cut;
    model.sigma_cut_high = opt.sigma_cut_high;
    model.emission_type = opt.emission_type;
    model.profile_mode = opt.profile_mode;
    model.nonthermal_kappa = opt.nonthermal_kappa;
    model.variable_kappa = opt.variable_kappa;
    model.variable_kappa_min = opt.variable_kappa_min;
    model.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    model.variable_kappa_max = opt.variable_kappa_max;
    model.powerlaw_p = opt.powerlaw_p;
    model.powerlaw_eta = opt.powerlaw_eta;
    model.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    model.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    model.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;

    const size_t nblocks = static_cast<size_t>(model.nblocks);
    const size_t values_per_block = static_cast<size_t>(model.nvar) *
                                    static_cast<size_t>(model.nx1) *
                                    static_cast<size_t>(model.nx2) *
                                    static_cast<size_t>(model.nx3);
    model.extents = typename AthenaKDirectRadiationModel<RealT>::RealView("athenak_cks_extents", nblocks * 6u);
    model.cell_geom = typename AthenaKDirectRadiationModel<RealT>::RealView("athenak_cks_cell_geom", nblocks * 6u);
    model.prims = typename AthenaKDirectRadiationModel<RealT>::FloatView("athenak_cks_prims", nblocks * values_per_block);
    const size_t derived_per_block = static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::NumDerived) *
                                     static_cast<size_t>(model.nx1) *
                                     static_cast<size_t>(model.nx2) *
                                     static_cast<size_t>(model.nx3);
    model.derived = typename AthenaKDirectRadiationModel<RealT>::FloatView("athenak_cks_derived", nblocks * derived_per_block);
    model.levels = typename AthenaKDirectRadiationModel<RealT>::IntView("athenak_cks_levels", nblocks);
    model.index_offsets = typename AthenaKDirectRadiationModel<RealT>::IntView("athenak_cks_index_offsets", staged.index_offsets.size());
    model.index_candidates = typename AthenaKDirectRadiationModel<RealT>::IntView("athenak_cks_index_candidates", staged.index_candidates.size());

    using HostRealView = Kokkos::View<const RealT*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using HostFloatView = Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using HostIntView = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    if (!staged.extents.empty()) Kokkos::deep_copy(model.extents, HostRealView(staged.extents.data(), staged.extents.size()));
    if (!staged.cell_geom.empty()) Kokkos::deep_copy(model.cell_geom, HostRealView(staged.cell_geom.data(), staged.cell_geom.size()));
    if (!staged.prims.empty()) Kokkos::deep_copy(model.prims, HostFloatView(staged.prims.data(), staged.prims.size()));
    if (!staged.levels.empty()) Kokkos::deep_copy(model.levels, HostIntView(staged.levels.data(), staged.levels.size()));
    if (!staged.index_offsets.empty()) Kokkos::deep_copy(model.index_offsets, HostIntView(staged.index_offsets.data(), staged.index_offsets.size()));
    if (!staged.index_candidates.empty()) Kokkos::deep_copy(model.index_candidates, HostIntView(staged.index_candidates.data(), staged.index_candidates.size()));
    model.index_nx = staged.index_nx;
    model.index_ny = staged.index_ny;
    model.index_nz = staged.index_nz;
    model.index_xmin = static_cast<RealT>(staged.index_xmin);
    model.index_xmax = static_cast<RealT>(staged.index_xmax);
    model.index_ymin = static_cast<RealT>(staged.index_ymin);
    model.index_ymax = static_cast<RealT>(staged.index_ymax);
    model.index_zmin = static_cast<RealT>(staged.index_zmin);
    model.index_zmax = static_cast<RealT>(staged.index_zmax);
    model.index_dx = static_cast<RealT>(staged.index_dx);
    model.index_dy = static_cast<RealT>(staged.index_dy);
    model.index_dz = static_cast<RealT>(staged.index_dz);
    model.index_inv_dx = RealT(1) / model.index_dx;
    model.index_inv_dy = RealT(1) / model.index_dy;
    model.index_inv_dz = RealT(1) / model.index_dz;

    AthenaKDirectRadiationModel<RealT> device_model = model;
    const size_t cells_per_block = static_cast<size_t>(model.nx1) *
                                   static_cast<size_t>(model.nx2) *
                                   static_cast<size_t>(model.nx3);
    const size_t nx1s = static_cast<size_t>(model.nx1);
    const size_t nx2s = static_cast<size_t>(model.nx2);
    const size_t nvars = static_cast<size_t>(model.nvar);
    const size_t num_derived = static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::NumDerived);
    const RealT spin = model.spin;
    const RealT gam = model.gam;
    const RealT rho_unit = model.rho_unit_cgs();
    const RealT b_unit = model.b_unit_cgs();
    const RealT trat_small = model.trat_small;
    const RealT trat_large = model.trat_large;
    const RealT beta_crit2 = max_val(model.beta_crit * model.beta_crit, RealT(1e-40));
    Kokkos::parallel_for(
        "KPOLARISAthenaKDerivedPack",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<size_t>>(0, nblocks * cells_per_block),
        KOKKOS_LAMBDA(const size_t n) {
            const size_t mb = n / cells_per_block;
            const size_t local = n - mb * cells_per_block;
            const int i = static_cast<int>(local % nx1s);
            const int j = static_cast<int>((local / nx1s) % nx2s);
            const int k = static_cast<int>(local / (nx1s * nx2s));
            const size_t ebase = mb * 6u;
            const RealT xmin = device_model.cell_geom(ebase + 0u);
            const RealT ymin = device_model.cell_geom(ebase + 1u);
            const RealT zmin = device_model.cell_geom(ebase + 2u);
            const RealT dx = RealT(1) / device_model.cell_geom(ebase + 3u);
            const RealT dy = RealT(1) / device_model.cell_geom(ebase + 4u);
            const RealT dz = RealT(1) / device_model.cell_geom(ebase + 5u);
            const RealT x = xmin + (RealT(i) + RealT(0.5)) * dx;
            const RealT y = ymin + (RealT(j) + RealT(0.5)) * dy;
            const RealT z = zmin + (RealT(k) + RealT(0.5)) * dz;
            const size_t var_stride = cells_per_block;
            const size_t cell = (static_cast<size_t>(k) * nx2s + static_cast<size_t>(j)) * nx1s +
                                static_cast<size_t>(i);
            const size_t prim_block = mb * nvars * var_stride;
            auto prim = [&](int v) -> RealT {
                return static_cast<RealT>(device_model.prims(prim_block + static_cast<size_t>(v) * var_stride + cell));
            };
            const RealT rho = prim(P_RHO);
            const RealT uu = prim(P_UU);
            RealT ne = RealT(0);
            RealT thetae = RealT(0);
            RealT b_cgs = RealT(0);
            RealT sigma = RealT(1e300);
            RealT beta = RealT(1e-300);
            if (rho > RealT(0) && uu > RealT(0)) {
                const RealT U1 = prim(P_U1);
                const RealT U2 = prim(P_U2);
                const RealT U3 = prim(P_U3);
                const RealT B1 = prim(P_B1);
                const RealT B2 = prim(P_B2);
                const RealT B3 = prim(P_B3);
                const RealT radius2 = x * x + y * y + z * z;
                const RealT a2 = spin * spin;
                const RealT s = radius2 - a2;
                const RealT discr = Kokkos::sqrt(max_val(s * s + RealT(4) * a2 * z * z, RealT(1e-300)));
                const RealT r = Kokkos::sqrt(max_val(RealT(0.5) * (s + discr), RealT(1e-300)));
                const RealT r2 = r * r;
                const RealT denom = r2 * r2 + a2 * z * z;
                const RealT amp = RealT(2) * r2 * r / denom;
                RealT l[ndim] = {
                    RealT(1),
                    (r * x + spin * y) / (r2 + a2),
                    (r * y - spin * x) / (r2 + a2),
                    z / r};
                RealT gcov[ndim][ndim];
                for (int mu = 0; mu < ndim; ++mu) {
                    for (int nu = 0; nu < ndim; ++nu) {
                        const RealT eta = (mu == nu ? (mu == 0 ? RealT(-1) : RealT(1)) : RealT(0));
                        gcov[mu][nu] = eta + amp * l[mu] * l[nu];
                    }
                }
                const RealT gcon00 = RealT(-1) - amp;
                const RealT alpha = RealT(1) / Kokkos::sqrt(-gcon00);
                const RealT udotu =
                    gcov[1][1] * U1 * U1 + RealT(2) * gcov[1][2] * U1 * U2 + RealT(2) * gcov[1][3] * U1 * U3 +
                    gcov[2][2] * U2 * U2 + RealT(2) * gcov[2][3] * U2 * U3 + gcov[3][3] * U3 * U3;
                const RealT gamma = Kokkos::sqrt(RealT(1) + abs_val(udotu));
                RealT ucon[ndim];
                ucon[0] = gamma / alpha;
                ucon[1] = U1 - gamma * alpha * gcov[0][1];
                ucon[2] = U2 - gamma * alpha * gcov[0][2];
                ucon[3] = U3 - gamma * alpha * gcov[0][3];
                RealT ucov[ndim];
                for (int mu = 0; mu < ndim; ++mu) {
                    ucov[mu] = gcov[mu][0] * ucon[0] + gcov[mu][1] * ucon[1] +
                               gcov[mu][2] * ucon[2] + gcov[mu][3] * ucon[3];
                }
                RealT bcon[ndim];
                bcon[0] = B1 * ucov[1] + B2 * ucov[2] + B3 * ucov[3];
                const RealT inv_u0 = RealT(1) / max_val(ucon[0], RealT(1e-300));
                bcon[1] = (B1 + ucon[1] * bcon[0]) * inv_u0;
                bcon[2] = (B2 + ucon[2] * bcon[0]) * inv_u0;
                bcon[3] = (B3 + ucon[3] * bcon[0]) * inv_u0;
                RealT bsq = RealT(0);
                for (int mu = 0; mu < ndim; ++mu) {
                    const RealT bcov_mu = gcov[mu][0] * bcon[0] + gcov[mu][1] * bcon[1] +
                                          gcov[mu][2] * bcon[2] + gcov[mu][3] * bcon[3];
                    bsq += bcon[mu] * bcov_mu;
                }
                bsq = max_val(abs_val(bsq), RealT(1e-40));
                sigma = max_val(bsq / max_val(rho, RealT(1e-300)), RealT(1e-300));
                beta = max_val(uu * (gam - RealT(1)) / (RealT(0.5) * bsq), RealT(1e-300));
                ne = rho * rho_unit / RealT(1.67353264826e-24);
                b_cgs = Kokkos::sqrt(bsq) * b_unit;
                const RealT betasq = beta * beta / beta_crit2;
                const RealT trat = (trat_large * betasq + trat_small) / (RealT(1) + betasq);
                const RealT game = RealT(4) / RealT(3);
                const RealT gamp = RealT(5) / RealT(3);
                const RealT mp_me = RealT(1836.1526737600675);
                const RealT thetae_unit = mp_me * (game - RealT(1)) * (gamp - RealT(1)) /
                    ((gamp - RealT(1)) + (game - RealT(1)) * trat);
                thetae = max_val(thetae_unit * uu / max_val(rho, RealT(1e-300)), RealT(1e-3));
            }
            const size_t derived_block = mb * num_derived * var_stride;
            device_model.derived(derived_block + static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::DerivedNe) * var_stride + cell) = static_cast<float>(ne);
            device_model.derived(derived_block + static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::DerivedThetae) * var_stride + cell) = static_cast<float>(thetae);
            device_model.derived(derived_block + static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::DerivedB) * var_stride + cell) = static_cast<float>(b_cgs);
            device_model.derived(derived_block + static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::DerivedSigma) * var_stride + cell) = static_cast<float>(sigma);
            device_model.derived(derived_block + static_cast<size_t>(AthenaKDirectRadiationModel<RealT>::DerivedBeta) * var_stride + cell) = static_cast<float>(beta);
        });
    Kokkos::fence();

    report_timing(opt.timing, "athenak_direct_device_pack", pack_timer.seconds());
    report_timing(opt.timing, "athenak_materialize_total", total_timer.seconds());
    return model;
}

AthenaKDirectRadiationModel<DefaultReal> load_athenak_direct_model_from_binary(const AthenaKLoadOptions& opt) {
    Kokkos::Timer total_timer;
    AthenaKStagedDump staged = read_athenak_staged_dump(opt);
    AthenaKDirectRadiationModel<DefaultReal> model = materialize_athenak_direct_model_from_staged(staged, opt);
    report_timing(opt.timing, "athenak_load_total", total_timer.seconds());
    return model;
}

} // namespace kpolaris
