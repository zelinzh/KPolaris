#include "grmhd/hamr_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
constexpr int HAMR_RAW_NVAR_MIN = 9;
constexpr std::uint64_t HAMR_RECORD_OFFSET = 0x108;
constexpr std::uint64_t HAMR_HEADER_BYTES = 0x108;

struct HAMRHeader {
    double time = 0.0;
    int chunk_blocks = 0;
    int total_blocks = 0;
    int nx1 = 0;
    int nx2 = 0;
    int nx3 = 0;
    int root_n1 = 0;
    int root_n2 = 0;
    int root_n3 = 0;
    int max_block_slots = 0;
    int nvar = 0;
    double startx1 = 0.0;
    double startx2_raw = 0.0;
    double startx3 = 0.0;
    double dx1 = 0.0;
    double dx2_raw = 0.0;
    double dx3 = 0.0;
    double gamma = 13.0 / 9.0;
    double spin = 0.0;
    double r_in = 0.0;
    double r_out = 0.0;
    double stopx2_raw = 0.0;
    std::vector<int> block_ids;
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

struct HAMRBlockGeometry {
    int level = 0;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
};

void report_timing(int enabled, const char* name, double seconds) {
    if (enabled) {
        std::cout << "timing " << name << ' ' << seconds << " s\n";
    }
}

template<class T>
T read_at(const std::vector<char>& bytes, std::uint64_t offset, const char* name) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error(std::string("H-AMR parameters missing field ") + name);
    }
    T value{};
    std::copy(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
              bytes.begin() + static_cast<std::ptrdiff_t>(offset + sizeof(T)),
              reinterpret_cast<char*>(&value));
    return value;
}

std::uint64_t file_size_bytes(const std::filesystem::path& path) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    return static_cast<std::uint64_t>(n);
}

std::vector<char> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("failed to open H-AMR file: " + path.string());
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error("failed to size H-AMR file: " + path.string());
    std::vector<char> bytes(static_cast<size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        in.read(bytes.data(), size);
        if (!in) throw std::runtime_error("failed to read H-AMR file: " + path.string());
    }
    return bytes;
}

std::filesystem::path parameter_path_for_dump(const std::string& dump_path) {
    if (dump_path.empty()) {
        throw std::runtime_error("--hamr_dump is required for --model=hamr");
    }
    std::filesystem::path path(dump_path);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return path / "parameters";
    if (path.filename() == "parameters") return path;
    return path.parent_path() / "parameters";
}

std::filesystem::path data_directory_for_dump(const std::string& dump_path) {
    std::filesystem::path path(dump_path);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return path;
    return path.parent_path();
}

int numeric_suffix_after_new_dump(const std::string& name) {
    constexpr const char* prefix = "new_dump";
    constexpr size_t prefix_len = 8;
    if (name.size() <= prefix_len || name.compare(0, prefix_len, prefix) != 0) return -1;
    int value = 0;
    for (size_t i = prefix_len; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
        value = value * 10 + (name[i] - '0');
    }
    return value;
}

std::vector<std::filesystem::path> hamr_data_files(const std::string& dump_path) {
    std::filesystem::path dir = data_directory_for_dump(dump_path);
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        throw std::runtime_error("H-AMR dump path must be a directory or a file next to parameters");
    }
    std::vector<std::pair<int, std::filesystem::path>> numbered;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const int suffix = numeric_suffix_after_new_dump(entry.path().filename().string());
        if (suffix >= 0) numbered.push_back({suffix, entry.path()});
    }
    if (numbered.empty()) {
        throw std::runtime_error("H-AMR dump directory contains no new_dump* files: " + dir.string());
    }
    std::sort(numbered.begin(), numbered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::filesystem::path> out;
    out.reserve(numbered.size());
    for (const auto& item : numbered) out.push_back(item.second);
    return out;
}

HAMRHeader read_hamr_header(const std::filesystem::path& parameter_path) {
    const std::vector<char> bytes = read_file_bytes(parameter_path);
    if (bytes.size() < HAMR_HEADER_BYTES) {
        throw std::runtime_error("H-AMR parameters file is too small: " + parameter_path.string());
    }
    HAMRHeader h;
    h.time = read_at<double>(bytes, 0x000, "time");
    h.chunk_blocks = read_at<std::int32_t>(bytes, 0x008, "chunk_blocks");
    h.total_blocks = read_at<std::int32_t>(bytes, 0x00c, "total_blocks");
    h.nx1 = read_at<std::int32_t>(bytes, 0x040, "nx1");
    h.nx2 = read_at<std::int32_t>(bytes, 0x044, "nx2");
    h.nx3 = read_at<std::int32_t>(bytes, 0x048, "nx3");
    h.max_block_slots = read_at<std::int32_t>(bytes, 0x04c, "max_block_slots");
    h.root_n1 = read_at<std::int32_t>(bytes, 0x050, "root_n1");
    h.root_n2 = read_at<std::int32_t>(bytes, 0x054, "root_n2");
    h.root_n3 = read_at<std::int32_t>(bytes, 0x058, "root_n3");
    h.startx1 = read_at<double>(bytes, 0x05c, "startx1");
    h.startx2_raw = read_at<double>(bytes, 0x064, "startx2");
    h.startx3 = read_at<double>(bytes, 0x06c, "startx3");
    h.dx1 = read_at<double>(bytes, 0x074, "dx1");
    h.dx2_raw = read_at<double>(bytes, 0x07c, "dx2");
    h.dx3 = read_at<double>(bytes, 0x084, "dx3");
    // H-AMR keeps the HARM-style scalar ordering here: tf, spin, gamma, courant.
    h.spin = read_at<double>(bytes, 0x094, "spin");
    h.gamma = read_at<double>(bytes, 0x09c, "gamma");
    h.r_in = read_at<double>(bytes, 0x0ac, "r_in");
    h.r_out = read_at<double>(bytes, 0x0b4, "r_out");
    h.stopx2_raw = read_at<double>(bytes, 0x0c4, "stopx2");
    if (!(h.chunk_blocks > 0 && h.total_blocks > 0 && h.nx1 > 0 && h.nx2 > 0 && h.nx3 > 0 &&
          h.root_n1 > 0 && h.root_n2 > 0 && h.root_n3 > 0 && h.max_block_slots > 0 &&
          h.dx1 > 0.0 && h.dx2_raw > 0.0 && h.dx3 > 0.0)) {
        throw std::runtime_error("unsupported or corrupt H-AMR parameters header");
    }
    const std::uint64_t record_bytes = static_cast<std::uint64_t>(h.total_blocks) * 3u * sizeof(std::int32_t);
    if (HAMR_RECORD_OFFSET + record_bytes > bytes.size()) {
        throw std::runtime_error("H-AMR parameters do not contain the expected block-id table");
    }
    h.block_ids.resize(static_cast<size_t>(h.total_blocks));
    for (int b = 0; b < h.total_blocks; ++b) {
        const std::uint64_t off = HAMR_RECORD_OFFSET + static_cast<std::uint64_t>(b) * 12u;
        const int block_id = read_at<std::int32_t>(bytes, off + 0u, "block_id");
        const int active = read_at<std::int32_t>(bytes, off + 4u, "active");
        if (active != 1) {
            throw std::runtime_error("H-AMR block-id table contains an inactive block");
        }
        if (block_id < 0 || block_id >= h.max_block_slots) {
            throw std::runtime_error("H-AMR block id is outside max_block_slots");
        }
        h.block_ids[static_cast<size_t>(b)] = block_id;
    }
    return h;
}

int clamp_bin_index(double x, double xmin, double dx, int n) {
    int i = static_cast<int>(std::floor((x - xmin) / dx));
    return std::clamp(i, 0, n - 1);
}

size_t flat_bin_index(const MeshBlockIndex& idx, int ix, int iy, int iz) {
    return (static_cast<size_t>(iz) * static_cast<size_t>(idx.ny) + static_cast<size_t>(iy)) *
           static_cast<size_t>(idx.nx) + static_cast<size_t>(ix);
}

MeshBlockIndex build_meshblock_index(const std::vector<HAMRBlockGeometry>& blocks, int timing) {
    Kokkos::Timer timer;
    if (blocks.empty()) throw std::runtime_error("H-AMR dump contains no meshblocks");
    MeshBlockIndex idx;
    idx.xmin = blocks.front().xmin; idx.xmax = blocks.front().xmax;
    idx.ymin = blocks.front().ymin; idx.ymax = blocks.front().ymax;
    idx.zmin = blocks.front().zmin; idx.zmax = blocks.front().zmax;
    for (const HAMRBlockGeometry& b : blocks) {
        idx.xmin = std::min(idx.xmin, b.xmin); idx.xmax = std::max(idx.xmax, b.xmax);
        idx.ymin = std::min(idx.ymin, b.ymin); idx.ymax = std::max(idx.ymax, b.ymax);
        idx.zmin = std::min(idx.zmin, b.zmin); idx.zmax = std::max(idx.zmax, b.zmax);
    }
    const double root = std::cbrt(static_cast<double>(blocks.size()));
    const int nb = std::clamp(static_cast<int>(std::ceil(root * 4.0)), 16, 96);
    idx.nx = nb; idx.ny = nb; idx.nz = nb;
    idx.dx = (idx.xmax - idx.xmin) / double(idx.nx);
    idx.dy = (idx.ymax - idx.ymin) / double(idx.ny);
    idx.dz = (idx.zmax - idx.zmin) / double(idx.nz);
    if (!(idx.dx > 0.0 && idx.dy > 0.0 && idx.dz > 0.0)) {
        throw std::runtime_error("invalid H-AMR AMR index extent");
    }
    const size_t nbin = static_cast<size_t>(idx.nx) * static_cast<size_t>(idx.ny) * static_cast<size_t>(idx.nz);
    std::vector<std::vector<int>> bins(nbin);
    for (size_t mb = 0; mb < blocks.size(); ++mb) {
        const HAMRBlockGeometry& b = blocks[mb];
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
    idx.offsets.resize(nbin + 1u, 0);
    size_t total = 0;
    for (size_t b = 0; b < nbin; ++b) {
        idx.offsets[b] = static_cast<int>(total);
        total += bins[b].size();
    }
    idx.offsets[nbin] = static_cast<int>(total);
    idx.candidates.reserve(total);
    for (const auto& bin : bins) idx.candidates.insert(idx.candidates.end(), bin.begin(), bin.end());
    if (timing) {
        const double avg = total > 0 ? double(total) / double(nbin) : 0.0;
        std::cout << "timing hamr_meshblock_index " << timer.seconds() << " s bins "
                  << idx.nx << 'x' << idx.ny << 'x' << idx.nz
                  << " blocks " << blocks.size() << " avg_candidates " << avg << "\n";
    }
    return idx;
}

std::array<int, 3> root_coords(int root_id, const HAMRHeader& h, const std::string& root_order) {
    const int root_count = h.root_n1 * h.root_n2 * h.root_n3;
    if (root_id < 0 || root_id >= root_count) {
        throw std::runtime_error("H-AMR root block id is out of range");
    }
    if (root_order == "morton" || root_order == "morton_lsb" || root_order == "sfc_morton") {
        const int max_root = std::max({h.root_n1, h.root_n2, h.root_n3});
        int bits_per_axis = 0;
        while ((1 << bits_per_axis) < max_root) ++bits_per_axis;
        if (bits_per_axis <= 0 || bits_per_axis >= 10) {
            throw std::runtime_error("unsupported H-AMR Morton root grid dimensions");
        }
        int lane[3] = {0, 0, 0};
        for (int bit = 0; bit < bits_per_axis; ++bit) {
            for (int axis = 0; axis < 3; ++axis) {
                lane[axis] |= ((root_id >> (3 * bit + axis)) & 1) << bit;
            }
        }
        const std::array<int, 3> coords{lane[2], lane[1], lane[0]};
        if (coords[0] < h.root_n1 && coords[1] < h.root_n2 && coords[2] < h.root_n3) {
            return coords;
        }
        throw std::runtime_error("H-AMR Morton root id decodes outside root grid");
    }
    if (root_order == "x1x2x3") {
        return {root_id % h.root_n1,
                (root_id / h.root_n1) % h.root_n2,
                root_id / (h.root_n1 * h.root_n2)};
    }
    if (root_order == "x1x3x2") {
        return {root_id % h.root_n1,
                root_id / (h.root_n1 * h.root_n3),
                (root_id / h.root_n1) % h.root_n3};
    }
    if (root_order == "x3x2x1") {
        return {root_id / (h.root_n2 * h.root_n3),
                (root_id / h.root_n3) % h.root_n2,
                root_id % h.root_n3};
    }
    throw std::runtime_error("unsupported H-AMR root_order: " + root_order);
}

HAMRBlockGeometry geometry_for_block_id(const HAMRHeader& h, int block_id, const HAMRLoadOptions& opt,
                                        double dx2_native, double startx2_native) {
    const int root_count = h.root_n1 * h.root_n2 * h.root_n3;
    const double root_dx1 = double(h.nx1) * h.dx1;
    const double root_dx2 = double(h.nx2) * dx2_native;
    const double root_dx3 = double(h.nx3) * h.dx3;
    int root_id = block_id;
    int child = -1;
    HAMRBlockGeometry out;
    if (opt.id_order == "root_slot" || opt.id_order == "slot9") {
        root_id = block_id / 9;
        const int slot = block_id % 9;
        if (slot > 0) {
            child = slot - 1;
            out.level = 1;
        }
    } else if (block_id >= root_count) {
        const int n = block_id - root_count;
        if (opt.id_order == "root_major") {
            root_id = n / 8;
            child = n % 8;
        } else if (opt.id_order == "child_major") {
            root_id = n % root_count;
            child = n / root_count;
        } else {
            throw std::runtime_error("unsupported H-AMR id_order: " + opt.id_order);
        }
        if (child < 0 || child >= 8) throw std::runtime_error("H-AMR child id is out of range");
        out.level = 1;
    }
    const auto rc = root_coords(root_id, h, opt.root_order);
    double xmin = h.startx1 + double(rc[0]) * root_dx1;
    double ymin = startx2_native + double(rc[1]) * root_dx2;
    double zmin = h.startx3 + double(rc[2]) * root_dx3;
    double dx1 = root_dx1;
    double dx2 = root_dx2;
    double dx3 = root_dx3;
    if (child >= 0) {
        dx1 *= 0.5;
        dx2 *= 0.5;
        dx3 *= 0.5;
        xmin += double((child >> 2) & 1) * dx1;
        ymin += double((child >> 1) & 1) * dx2;
        zmin += double(child & 1) * dx3;
    }
    out.xmin = xmin;
    out.xmax = xmin + dx1;
    out.ymin = ymin;
    out.ymax = ymin + dx2;
    out.zmin = zmin;
    out.zmax = zmin + dx3;
    return out;
}

void read_raw_chunk(const std::filesystem::path& path, int blocks_in_file, const HAMRHeader& h,
                    std::vector<float>& raw_values, int& next_block) {
    const size_t cells = static_cast<size_t>(h.nx1) * static_cast<size_t>(h.nx2) * static_cast<size_t>(h.nx3);
    const size_t values_per_block = cells * static_cast<size_t>(h.nvar);
    const size_t values_in_file = static_cast<size_t>(blocks_in_file) * values_per_block;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open H-AMR data file: " + path.string());
    if (next_block + blocks_in_file > h.total_blocks) {
        throw std::runtime_error("H-AMR data files contain more blocks than parameters");
    }
    in.read(reinterpret_cast<char*>(raw_values.data() + static_cast<size_t>(next_block) * values_per_block),
            static_cast<std::streamsize>(values_in_file * sizeof(float)));
    if (!in) throw std::runtime_error("failed to read H-AMR data file: " + path.string());
    next_block += blocks_in_file;
}

BHACLoadOptions make_bhac_options_for_hamr(const HAMRLoadOptions& opt, const HAMRHeader& h) {
    BHACLoadOptions out;
    out.freq = opt.freq;
    out.M_unit = opt.M_unit;
    out.mbh_solar = opt.mbh_solar;
    out.trat_small = opt.trat_small;
    out.trat_large = opt.trat_large;
    out.beta_crit = opt.beta_crit;
    out.gamma = opt.gamma > DefaultReal(0) ? opt.gamma : DefaultReal(h.gamma);
    out.sigma_cut = opt.sigma_cut;
    out.sigma_cut_high = opt.sigma_cut_high;
    out.emission_type = opt.emission_type;
    out.profile_mode = opt.profile_mode;
    out.nonthermal_kappa = opt.nonthermal_kappa;
    out.variable_kappa = opt.variable_kappa;
    out.variable_kappa_min = opt.variable_kappa_min;
    out.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    out.variable_kappa_max = opt.variable_kappa_max;
    out.powerlaw_p = opt.powerlaw_p;
    out.powerlaw_eta = opt.powerlaw_eta;
    out.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    out.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    out.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    out.r_in = opt.r_in > DefaultReal(0) ? opt.r_in : DefaultReal(h.r_in);
    out.r_out = opt.r_out > DefaultReal(0) ? opt.r_out : DefaultReal(h.r_out);
    out.hslope = DefaultReal(1) - opt.hslope;
    out.timing = opt.timing;
    return out;
}

BHACLoadOptions make_bhac_options_for_hamr_staged(const HAMRLoadOptions& opt, const BHACStagedDump& staged) {
    BHACLoadOptions out;
    out.freq = opt.freq;
    out.M_unit = opt.M_unit;
    out.mbh_solar = opt.mbh_solar;
    out.trat_small = opt.trat_small;
    out.trat_large = opt.trat_large;
    out.beta_crit = opt.beta_crit;
    out.gamma = opt.gamma > DefaultReal(0) ? opt.gamma : staged.gamma;
    out.sigma_cut = opt.sigma_cut;
    out.sigma_cut_high = opt.sigma_cut_high;
    out.emission_type = opt.emission_type;
    out.profile_mode = opt.profile_mode;
    out.nonthermal_kappa = opt.nonthermal_kappa;
    out.variable_kappa = opt.variable_kappa;
    out.variable_kappa_min = opt.variable_kappa_min;
    out.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    out.variable_kappa_max = opt.variable_kappa_max;
    out.powerlaw_p = opt.powerlaw_p;
    out.powerlaw_eta = opt.powerlaw_eta;
    out.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    out.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    out.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    out.r_in = opt.r_in > DefaultReal(0) ? opt.r_in : staged.r_in;
    out.r_out = opt.r_out > DefaultReal(0) ? opt.r_out : staged.r_out;
    out.hslope = staged.hslope;
    out.timing = opt.timing;
    return out;
}

BHACStagedDump read_hamr_staged_dump_impl(const HAMRLoadOptions& opt) {
    Kokkos::Timer timer;
    const std::filesystem::path param_path = parameter_path_for_dump(opt.dump_path);
    HAMRHeader h = read_hamr_header(param_path);
    const std::vector<std::filesystem::path> data_files = hamr_data_files(opt.dump_path);
    if (opt.timing) {
        std::cout << "timing hamr_header blocks " << h.total_blocks
                  << " chunk_blocks " << h.chunk_blocks
                  << " block_nx " << h.nx1 << 'x' << h.nx2 << 'x' << h.nx3
                  << " root_blocks " << h.root_n1 << 'x' << h.root_n2 << 'x' << h.root_n3
                  << " nfiles " << data_files.size() << "\n";
    }

    const size_t cells = static_cast<size_t>(h.nx1) * static_cast<size_t>(h.nx2) * static_cast<size_t>(h.nx3);
    const size_t raw_values_per_block_guess = cells * HAMR_RAW_NVAR_MIN;
    const std::uint64_t first_size = file_size_bytes(data_files.front());
    if (first_size == 0 || first_size % (static_cast<std::uint64_t>(h.chunk_blocks) * cells * sizeof(float)) != 0) {
        throw std::runtime_error("H-AMR data file size does not match parameters block geometry");
    }
    h.nvar = static_cast<int>(first_size / (static_cast<std::uint64_t>(h.chunk_blocks) * cells * sizeof(float)));
    if (h.nvar < HAMR_RAW_NVAR_MIN) {
        throw std::runtime_error("H-AMR data file has fewer variables than expected");
    }
    (void)raw_values_per_block_guess;

    const size_t raw_values_per_block = cells * static_cast<size_t>(h.nvar);
    std::vector<float> raw_values(static_cast<size_t>(h.total_blocks) * raw_values_per_block);
    int next_block = 0;
    for (const auto& path : data_files) {
        const std::uint64_t size = file_size_bytes(path);
        if (size % (raw_values_per_block * sizeof(float)) != 0) {
            throw std::runtime_error("H-AMR data file size is not an integer number of blocks: " + path.string());
        }
        const int blocks_in_file = static_cast<int>(size / (raw_values_per_block * sizeof(float)));
        read_raw_chunk(path, blocks_in_file, h, raw_values, next_block);
    }
    if (next_block != h.total_blocks) {
        throw std::runtime_error("H-AMR data files do not match parameters total_blocks");
    }
    report_timing(opt.timing, "hamr_binary_read", timer.seconds());
    timer.reset();

    const double pi = 3.141592653589793238462643383279502884;
    const double startx2_native = pi * (h.startx2_raw + 1.0) * 0.5;
    const double dx2_native = pi * h.dx2_raw * 0.5;
    const double x2_scale = pi * 0.5;
    const std::vector<int>& block_ids = h.block_ids;

    std::vector<HAMRBlockGeometry> geometries(static_cast<size_t>(h.total_blocks));
    for (int mb = 0; mb < h.total_blocks; ++mb) {
        geometries[static_cast<size_t>(mb)] =
            geometry_for_block_id(h, block_ids[static_cast<size_t>(mb)], opt, dx2_native, startx2_native);
    }
    MeshBlockIndex index = build_meshblock_index(geometries, opt.timing);

    BHACStagedDump staged;
    staged.nblocks = h.total_blocks;
    staged.nvar = 8;
    staged.nx1 = h.nx1;
    staged.nx2 = h.nx2;
    staged.nx3 = h.nx3;
    staged.spin = DefaultReal(h.spin);
    staged.gamma = opt.gamma > DefaultReal(0) ? opt.gamma : DefaultReal(h.gamma);
    staged.time = DefaultReal(h.time);
    staged.startx1 = DefaultReal(h.startx1);
    staged.startx2 = DefaultReal(startx2_native);
    staged.startx3 = DefaultReal(h.startx3);
    staged.stopx1 = DefaultReal(h.startx1 + double(h.root_n1) * double(h.nx1) * h.dx1);
    staged.stopx2 = DefaultReal(startx2_native + double(h.root_n2) * double(h.nx2) * dx2_native);
    staged.stopx3 = DefaultReal(h.startx3 + double(h.root_n3) * double(h.nx3) * h.dx3);
    staged.r_in = DefaultReal(h.r_in);
    staged.r_out = DefaultReal(h.r_out);
    staged.hslope = DefaultReal(1) - opt.hslope;
    staged.extents.resize(static_cast<size_t>(staged.nblocks) * 6u);
    staged.cell_geom.resize(static_cast<size_t>(staged.nblocks) * 6u);
    staged.levels.resize(static_cast<size_t>(staged.nblocks));
    const size_t prim_values_per_block = 8u * cells;
    staged.prims.resize(static_cast<size_t>(staged.nblocks) * prim_values_per_block, 0.0f);

    for (int mb = 0; mb < h.total_blocks; ++mb) {
        const HAMRBlockGeometry& g = geometries[static_cast<size_t>(mb)];
        const size_t base = static_cast<size_t>(mb) * 6u;
        staged.extents[base + 0u] = DefaultReal(g.xmin);
        staged.extents[base + 1u] = DefaultReal(g.xmax);
        staged.extents[base + 2u] = DefaultReal(g.ymin);
        staged.extents[base + 3u] = DefaultReal(g.ymax);
        staged.extents[base + 4u] = DefaultReal(g.zmin);
        staged.extents[base + 5u] = DefaultReal(g.zmax);
        staged.cell_geom[base + 0u] = DefaultReal(g.xmin);
        staged.cell_geom[base + 1u] = DefaultReal(g.ymin);
        staged.cell_geom[base + 2u] = DefaultReal(g.zmin);
        staged.cell_geom[base + 3u] = DefaultReal(h.nx1) / DefaultReal(g.xmax - g.xmin);
        staged.cell_geom[base + 4u] = DefaultReal(h.nx2) / DefaultReal(g.ymax - g.ymin);
        staged.cell_geom[base + 5u] = DefaultReal(h.nx3) / DefaultReal(g.zmax - g.zmin);
        staged.levels[static_cast<size_t>(mb)] = g.level;

        const size_t raw_block = static_cast<size_t>(mb) * raw_values_per_block;
        const size_t prim_block = static_cast<size_t>(mb) * prim_values_per_block;
        for (int i = 0; i < h.nx1; ++i) {
            for (int j = 0; j < h.nx2; ++j) {
                for (int k = 0; k < h.nx3; ++k) {
                    const size_t raw_cell = ((static_cast<size_t>(i) * static_cast<size_t>(h.nx2) +
                                             static_cast<size_t>(j)) * static_cast<size_t>(h.nx3) +
                                             static_cast<size_t>(k)) * static_cast<size_t>(h.nvar);
                    const size_t cell = (static_cast<size_t>(k) * static_cast<size_t>(h.nx2) +
                                         static_cast<size_t>(j)) * static_cast<size_t>(h.nx1) +
                                        static_cast<size_t>(i);
                    auto raw = [&](int v) -> float {
                        return raw_values[raw_block + raw_cell + static_cast<size_t>(v)];
                    };
                    const float bpol = opt.reverse_field ? -1.0f : 1.0f;
                    staged.prims[prim_block + static_cast<size_t>(P_RHO) * cells + cell] = raw(0);
                    staged.prims[prim_block + static_cast<size_t>(P_U1) * cells + cell] = raw(3);
                    staged.prims[prim_block + static_cast<size_t>(P_U2) * cells + cell] =
                        static_cast<float>(x2_scale * double(raw(4)));
                    staged.prims[prim_block + static_cast<size_t>(P_U3) * cells + cell] = raw(5);
                    staged.prims[prim_block + static_cast<size_t>(P_UU) * cells + cell] = raw(1);
                    staged.prims[prim_block + static_cast<size_t>(P_B1) * cells + cell] = bpol * raw(6);
                    staged.prims[prim_block + static_cast<size_t>(P_B2) * cells + cell] =
                        bpol * static_cast<float>(x2_scale * double(raw(7)));
                    staged.prims[prim_block + static_cast<size_t>(P_B3) * cells + cell] = bpol * raw(8);
                }
            }
        }
    }
    staged.index_offsets = std::move(index.offsets);
    staged.index_candidates = std::move(index.candidates);
    staged.index_nx = index.nx;
    staged.index_ny = index.ny;
    staged.index_nz = index.nz;
    staged.index_xmin = DefaultReal(index.xmin);
    staged.index_xmax = DefaultReal(index.xmax);
    staged.index_ymin = DefaultReal(index.ymin);
    staged.index_ymax = DefaultReal(index.ymax);
    staged.index_zmin = DefaultReal(index.zmin);
    staged.index_zmax = DefaultReal(index.zmax);
    staged.index_dx = DefaultReal(index.dx);
    staged.index_dy = DefaultReal(index.dy);
    staged.index_dz = DefaultReal(index.dz);
    report_timing(opt.timing, "hamr_host_stage", timer.seconds());
    return staged;
}

} // namespace

DefaultReal read_hamr_dump_time(const std::string& dump_path) {
    return DefaultReal(read_hamr_header(parameter_path_for_dump(dump_path)).time);
}

BHACStagedDump read_hamr_staged_dump(const HAMRLoadOptions& opt) {
    return read_hamr_staged_dump_impl(opt);
}

HAMRRadiationModel<DefaultReal> materialize_hamr_model_from_staged(const BHACStagedDump& staged,
                                                                   const HAMRLoadOptions& opt) {
    BHACLoadOptions bhac_opt = make_bhac_options_for_hamr_staged(opt, staged);
    return materialize_bhac_model_from_staged(staged, bhac_opt);
}

HAMRRadiationModel<DefaultReal> load_hamr_model_from_dump(const HAMRLoadOptions& opt) {
    Kokkos::Timer total_timer;
    const std::filesystem::path param_path = parameter_path_for_dump(opt.dump_path);
    HAMRHeader header = read_hamr_header(param_path);
    BHACStagedDump staged = read_hamr_staged_dump(opt);
    BHACLoadOptions bhac_opt = make_bhac_options_for_hamr(opt, header);
    HAMRRadiationModel<DefaultReal> model = materialize_bhac_model_from_staged(staged, bhac_opt);
    report_timing(opt.timing, "hamr_load_total", total_timer.seconds());
    return model;
}

} // namespace kpolaris
