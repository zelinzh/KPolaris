#include "grmhd/bhac_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

constexpr int D = 0;
constexpr int S1 = 1;
constexpr int S2 = 2;
constexpr int S3 = 3;
constexpr int DS = 8;
constexpr int RAPTOR_SPIN_INDEX = 3;
constexpr std::array<char, 16> BHAC_CACHE_MAGIC{
    'K', 'P', 'B', 'H', 'A', 'C', 'S', 'T', 'A', 'G', 'E', '1', '\0', '\0', '\0', '\0'
};
constexpr std::uint32_t BHAC_CACHE_VERSION = 1;
constexpr std::uint32_t BHAC_CACHE_ENDIAN = 0x01020304u;

struct BHACMetadata {
    int nleafs = 0;
    int levmax = 0;
    int ndim = 0;
    int ndir = 0;
    int nw = 0;
    int nws = 0;
    int neqpar = 0;
    int it = 0;
    double time = 0.0;
    std::array<int, 3> nx{0, 0, 0};
    std::vector<double> eqpar;
};

struct BHACBlock {
    int ind[3]{0, 0, 0};
    int level = 1;
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
    std::vector<double> conserved;
    std::vector<float> prims;
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

struct BHACDump {
    BHACMetadata meta;
    std::vector<BHACBlock> blocks;
    MeshBlockIndex index;
};

void report_timing(int enabled, const char* name, double seconds) {
    if (enabled) {
        std::cout << "timing " << name << ' ' << seconds << " s\n";
    }
}

std::uint64_t file_size_bytes(const std::string& path) {
    if (path.empty()) return 0;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    const std::streamoff size = in.tellg();
    return size > 0 ? static_cast<std::uint64_t>(size) : 0;
}

template<class T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing BHAC staged cache");
}

template<class T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("failed reading BHAC staged cache");
    return value;
}

void write_string(std::ofstream& out, const std::string& value) {
    const std::uint64_t n = static_cast<std::uint64_t>(value.size());
    write_pod(out, n);
    if (n > 0) {
        out.write(value.data(), static_cast<std::streamsize>(n));
        if (!out) throw std::runtime_error("failed writing BHAC staged cache string");
    }
}

std::string read_string(std::ifstream& in) {
    const std::uint64_t n = read_pod<std::uint64_t>(in);
    if (n > (1ull << 30)) {
        throw std::runtime_error("BHAC staged cache string is unreasonably large");
    }
    std::string value(static_cast<size_t>(n), '\0');
    if (n > 0) {
        in.read(value.data(), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("failed reading BHAC staged cache string");
    }
    return value;
}

template<class T>
void write_vector(std::ofstream& out, const std::vector<T>& values) {
    const std::uint64_t n = static_cast<std::uint64_t>(values.size());
    write_pod(out, n);
    if (n > 0) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(n * sizeof(T)));
        if (!out) throw std::runtime_error("failed writing BHAC staged cache vector");
    }
}

template<class T>
std::vector<T> read_vector(std::ifstream& in) {
    const std::uint64_t n = read_pod<std::uint64_t>(in);
    if (n > (1ull << 34) / sizeof(T)) {
        throw std::runtime_error("BHAC staged cache vector is unreasonably large");
    }
    std::vector<T> values(static_cast<size_t>(n));
    if (n > 0) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(n * sizeof(T)));
        if (!in) throw std::runtime_error("failed reading BHAC staged cache vector");
    }
    return values;
}

bool same_real(DefaultReal a, DefaultReal b) {
    return a == b;
}

int read_i32(std::ifstream& in) {
    std::int32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("failed reading BHAC int32");
    return static_cast<int>(v);
}

double read_f64(std::ifstream& in) {
    double v = 0.0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("failed reading BHAC float64");
    return v;
}

BHACMetadata read_bhac_metadata(std::ifstream& in, const std::string& path) {
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    if (file_size < 40) {
        throw std::runtime_error("BHAC dump is too small: " + path);
    }
    BHACMetadata meta;
    in.seekg(file_size - std::streamoff(40), std::ios::beg);
    meta.nleafs = read_i32(in);
    meta.levmax = read_i32(in);
    meta.ndim = read_i32(in);
    meta.ndir = read_i32(in);
    meta.nw = read_i32(in);
    meta.nws = read_i32(in);
    meta.neqpar = read_i32(in);
    meta.it = read_i32(in);
    meta.time = read_f64(in);
    if (!(meta.nleafs > 0 && meta.ndim == 3 && meta.nw >= 13 && meta.nws >= 0 && meta.neqpar > 0)) {
        throw std::runtime_error("unsupported BHAC metadata in " + path);
    }

    const std::streamoff meta_begin = file_size - std::streamoff(40 + meta.ndim * 4 + meta.neqpar * 8);
    if (meta_begin < 0) {
        throw std::runtime_error("invalid BHAC metadata offset in " + path);
    }
    in.seekg(meta_begin, std::ios::beg);
    for (int d = 0; d < meta.ndim; ++d) {
        meta.nx[d] = read_i32(in);
        if (meta.nx[d] <= 1) throw std::runtime_error("invalid BHAC block dimensions");
    }
    meta.eqpar.resize(static_cast<size_t>(meta.neqpar));
    for (int n = 0; n < meta.neqpar; ++n) {
        meta.eqpar[static_cast<size_t>(n)] = read_f64(in);
    }
    return meta;
}

bool is_physical_spin_candidate(double value, bool allow_extremal) {
    const double av = std::abs(value);
    return allow_extremal ? av <= 1.0 : av < 1.0;
}

int choose_spin_index(const BHACMetadata& meta, const BHACLoadOptions& opt) {
    if (opt.spin_index >= 0) {
        if (opt.spin_index >= meta.neqpar) {
            throw std::runtime_error("BHAC spin_index is outside neqpar");
        }
        if (!is_physical_spin_candidate(meta.eqpar[static_cast<size_t>(opt.spin_index)], true)) {
            throw std::runtime_error("BHAC spin_index points to an unphysical spin");
        }
        return opt.spin_index;
    }
    if (meta.neqpar > RAPTOR_SPIN_INDEX &&
        is_physical_spin_candidate(meta.eqpar[static_cast<size_t>(RAPTOR_SPIN_INDEX)], true)) {
        return RAPTOR_SPIN_INDEX;
    }
    if (meta.neqpar > 5 && is_physical_spin_candidate(meta.eqpar[5], false)) {
        return 5;
    }
    int best = -1;
    double best_abs = -1.0;
    for (int i = 0; i < meta.neqpar; ++i) {
        const double av = std::abs(meta.eqpar[static_cast<size_t>(i)]);
        if (av < 1.0 && av > best_abs) {
            best = i;
            best_abs = av;
        }
    }
    if (best >= 0) return best;
    for (int i = 0; i < meta.neqpar; ++i) {
        if (is_physical_spin_candidate(meta.eqpar[static_cast<size_t>(i)], true)) return i;
    }
    throw std::runtime_error("failed to infer BHAC spin index from eqpar");
}

double metadata_spin(const BHACMetadata& meta, const BHACLoadOptions& opt) {
    return meta.eqpar[static_cast<size_t>(choose_spin_index(meta, opt))];
}

uint64_t morton_encode(unsigned int i, unsigned int j, unsigned int k) {
    uint64_t out = 0;
    for (uint64_t bit = 0; bit < 21; ++bit) {
        out |= (uint64_t(i) & (uint64_t(1) << bit)) << (2 * bit);
        out |= (uint64_t(j) & (uint64_t(1) << bit)) << (2 * bit + 1);
        out |= (uint64_t(k) & (uint64_t(1) << bit)) << (2 * bit + 2);
    }
    return out;
}

std::vector<std::array<int, 3>> root_block_order(const std::array<int, 3>& ng, int sfc) {
    std::vector<std::array<int, 3>> order;
    order.reserve(static_cast<size_t>(ng[0]) * static_cast<size_t>(ng[1]) * static_cast<size_t>(ng[2]));
    for (int k = 0; k < ng[2]; ++k) {
        for (int j = 0; j < ng[1]; ++j) {
            for (int i = 0; i < ng[0]; ++i) {
                order.push_back({i, j, k});
            }
        }
    }
    if (sfc) {
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            return morton_encode(static_cast<unsigned>(a[0]), static_cast<unsigned>(a[1]),
                                 static_cast<unsigned>(a[2])) <
                   morton_encode(static_cast<unsigned>(b[0]), static_cast<unsigned>(b[1]),
                                 static_cast<unsigned>(b[2]));
        });
    }
    return order;
}

void child_index(int ndim, int child, int ip, int jp, int kp, int& i, int& j, int& k) {
    i = 2 * ip + child % 2;
    j = ndim >= 2 ? 2 * jp + (child / 2) % 2 : 0;
    k = ndim >= 3 ? 2 * kp + child / 4 : 0;
}

int count_node_leaves(const std::vector<int>& forest, size_t& pos, int ndim) {
    if (pos >= forest.size()) {
        throw std::runtime_error("BHAC forest ended while inferring root grid");
    }
    const int leaf = forest[pos++];
    if (leaf) return 1;
    int leaves = 0;
    const int nchild = 1 << ndim;
    for (int c = 0; c < nchild; ++c) {
        leaves += count_node_leaves(forest, pos, ndim);
    }
    return leaves;
}

int infer_root_count(const std::vector<int>& forest, int ndim, int nleafs) {
    size_t pos = 0;
    int roots = 0;
    int leaves = 0;
    while (pos < forest.size() && leaves < nleafs) {
        leaves += count_node_leaves(forest, pos, ndim);
        ++roots;
    }
    if (leaves != nleafs || pos != forest.size()) {
        throw std::runtime_error("BHAC forest structure is inconsistent with metadata");
    }
    return roots;
}

std::array<int, 3> choose_root_dims(const BHACMetadata& meta,
                                    const BHACLoadOptions& opt,
                                    int root_count) {
    const std::array<int, 3> requested_nxlone{opt.nxlone1, opt.nxlone2, opt.nxlone3};
    std::array<int, 3> requested_ng{0, 0, 0};
    bool has_explicit = false;
    for (int d = 0; d < 3; ++d) {
        if (requested_nxlone[d] > 0) {
            has_explicit = true;
            if (requested_nxlone[d] % meta.nx[d] != 0) {
                throw std::runtime_error("BHAC nxlone must be positive multiples of block nx");
            }
            requested_ng[d] = requested_nxlone[d] / meta.nx[d];
        }
    }
    if (has_explicit && requested_ng[0] > 0 && requested_ng[1] > 0 && requested_ng[2] > 0) {
        const int product = requested_ng[0] * requested_ng[1] * requested_ng[2];
        if (product != root_count) {
            throw std::runtime_error("BHAC nxlone root count does not match dump forest; use 0 to auto-infer");
        }
        return requested_ng;
    }

    std::array<int, 3> best{0, 0, 0};
    double best_score = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= root_count; ++i) {
        if (root_count % i != 0) continue;
        if (requested_ng[0] > 0 && i != requested_ng[0]) continue;
        const int rem = root_count / i;
        for (int j = 1; j <= rem; ++j) {
            if (rem % j != 0) continue;
            const int k = rem / j;
            if (requested_ng[1] > 0 && j != requested_ng[1]) continue;
            if (requested_ng[2] > 0 && k != requested_ng[2]) continue;
            const double li = std::log(double(i));
            const double lj = std::log(double(j));
            const double lk = std::log(double(k));
            const double score = (li - lj) * (li - lj) + (li - lk) * (li - lk) + (lj - lk) * (lj - lk);
            if (score < best_score) {
                best_score = score;
                best = {i, j, k};
            }
        }
    }
    if (best[0] <= 0 || best[1] <= 0 || best[2] <= 0) {
        throw std::runtime_error("failed to infer BHAC root grid dimensions from forest");
    }
    return best;
}

void read_node(const std::vector<int>& forest, size_t& pos, std::vector<BHACBlock>& blocks,
               int ndim, int level, int i, int j, int k) {
    if (pos >= forest.size()) {
        throw std::runtime_error("BHAC forest ended while reading AMR tree");
    }
    const int leaf = forest[pos++];
    if (leaf) {
        BHACBlock block;
        block.ind[0] = i;
        block.ind[1] = j;
        block.ind[2] = k;
        block.level = level;
        blocks.push_back(std::move(block));
        return;
    }
    const int nchild = 1 << ndim;
    for (int c = 0; c < nchild; ++c) {
        int ci = 0, cj = 0, ck = 0;
        child_index(ndim, c, i, j, k, ci, cj, ck);
        read_node(forest, pos, blocks, ndim, level + 1, ci, cj, ck);
    }
}

double bhac_theta(double x2, double hslope) {
    return x2 + 0.5 * hslope * std::sin(2.0 * x2);
}

double bhac_dtheta_dx2(double x2, double hslope) {
    return 1.0 + hslope * std::cos(2.0 * x2);
}

void bhac_native_metric(double spin, double hslope, const double X[3],
                        double gcov[4][4], double gcon[4][4]) {
    const double r = std::exp(X[0]);
    const double th = bhac_theta(X[1], hslope);
    const double hfac = bhac_dtheta_dx2(X[1], hslope);
    const double cth = std::cos(th);
    const double sth = std::sin(th);
    const double s2 = sth * sth;
    const double a2 = spin * spin;
    const double rho2 = r * r + a2 * cth * cth;
    const double f = 2.0 * r / rho2;
    const double delta = r * r - 2.0 * r + a2;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            gcov[mu][nu] = 0.0;
            gcon[mu][nu] = 0.0;
        }
    }
    gcov[0][0] = -1.0 + f;
    gcov[0][1] = f * r;
    gcov[1][0] = gcov[0][1];
    gcov[0][3] = -2.0 * spin * r * s2 / rho2;
    gcov[3][0] = gcov[0][3];
    gcov[1][1] = (1.0 + f) * r * r;
    gcov[1][3] = -spin * s2 * (1.0 + f) * r;
    gcov[3][1] = gcov[1][3];
    gcov[2][2] = rho2 * hfac * hfac;
    gcov[3][3] = s2 * (rho2 + a2 * s2 * (1.0 + f));

    gcon[0][0] = -(1.0 + f);
    gcon[0][1] = f / std::max(r, 1e-300);
    gcon[1][0] = gcon[0][1];
    gcon[1][1] = delta / (rho2 * std::max(r * r, 1e-300));
    gcon[1][3] = spin / (rho2 * std::max(r, 1e-300));
    gcon[3][1] = gcon[1][3];
    gcon[2][2] = 1.0 / (rho2 * std::max(hfac * hfac, 1e-300));
    gcon[3][3] = 1.0 / (rho2 * std::max(s2, 1e-300));
}

double detgamma_native(double spin, double hslope, const double X[3]) {
    const double r = std::exp(X[0]);
    const double th = bhac_theta(X[1], hslope);
    const double hfac = bhac_dtheta_dx2(X[1], hslope);
    const double cth = std::cos(th);
    const double sth = std::sin(th);
    const double rho2 = r * r + spin * spin * cth * cth;
    return std::abs(r * hfac * sth) * std::sqrt(std::max(rho2 * (rho2 + 2.0 * r), 0.0));
}

void cell_center(int c, const std::array<int, 3>& nx, const double lb[3],
                 const double dx[3], double X[3]) {
    const int i = c % nx[0];
    const int j = (c / nx[0]) % nx[1];
    const int k = c / (nx[0] * nx[1]);
    X[0] = lb[0] + (double(i) + 0.5) * dx[0];
    X[1] = lb[1] + (double(j) + 0.5) * dx[1];
    X[2] = lb[2] + (double(k) + 0.5) * dx[2];
}

void volume_center(double spin, double hslope, const double X[3], const double dx[3], double Xbar[3]) {
    constexpr double w1[3] = {1.0, 4.0, 1.0};
    constexpr double off[3] = {-1.0, 0.0, 1.0};
    double norm = 0.0;
    Xbar[0] = Xbar[1] = Xbar[2] = 0.0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                double Xq[3] = {
                    X[0] + 0.5 * dx[0] * off[i],
                    X[1] + 0.5 * dx[1] * off[j],
                    X[2] + 0.5 * dx[2] * off[k]};
                const double weight = w1[i] * w1[j] * w1[k] * detgamma_native(spin, hslope, Xq);
                norm += weight;
                Xbar[0] += weight * Xq[0];
                Xbar[1] += weight * Xq[1];
                Xbar[2] += weight * Xq[2];
            }
        }
    }
    if (norm > 0.0) {
        Xbar[0] /= norm;
        Xbar[1] /= norm;
        Xbar[2] /= norm;
    } else {
        Xbar[0] = X[0];
        Xbar[1] = X[1];
        Xbar[2] = X[2];
    }
}

void conserved_to_prims(const std::vector<double>& conserved, int c, int cells,
                        int nw, double spin, double gam, double hslope,
                        const double Xbar[3], int reverse_field,
                        float out[8]) {
    for (int v = 0; v < 8; ++v) out[v] = 0.0f;
    const int LFAC = nw - 2;
    const int XI = nw - 1;
    auto val = [&](int v) -> double {
        return conserved[static_cast<size_t>(v) * static_cast<size_t>(cells) + static_cast<size_t>(c)];
    };
    if (std::exp(Xbar[0]) < 1.0 || !(val(LFAC) > 0.0 && val(XI) > 0.0)) {
        return;
    }
    double gcov[4][4], gcon[4][4];
    bhac_native_metric(spin, hslope, Xbar, gcov, gcon);
    const double B[3] = {val(P_B1), val(P_B2), val(P_B3)};
    const double S[3] = {val(S1), val(S2), val(S3)};
    const double BS = S[0] * B[0] + S[1] * B[1] + S[2] * B[2];
    double gamma_spatial[4][4]{};
    double Bcov[4]{};
    double Scon[4]{};
    for (int i = 1; i < 4; ++i) {
        for (int j = 1; j < 4; ++j) {
            gamma_spatial[i][j] = gcon[i][j] + gcon[0][i] * gcon[0][j] / (-gcon[0][0]);
            Scon[j] += gamma_spatial[i][j] * S[i - 1];
            Bcov[j] += gcov[i][j] * B[i - 1];
        }
    }
    double bsq_lab = 0.0;
    for (int i = 1; i < 4; ++i) {
        bsq_lab += Bcov[i] * B[i - 1];
    }
    const double rho = val(D) / val(LFAC);
    double uu = (gam - 1.0) / gam * (val(XI) / (val(LFAC) * val(LFAC)) - rho) / (gam - 1.0);
    if (!(uu > 0.0) && DS < nw && val(D) != 0.0) {
        uu = (val(DS) / val(D)) * std::pow(std::max(rho, 1e-300), gam - 1.0) / (gam - 1.0);
    }
    if (!(rho > 0.0 && uu > 0.0)) {
        return;
    }
    double U[3];
    for (int i = 0; i < 3; ++i) {
        const double Bi = B[i];
        U[i] = Scon[i + 1] / (val(XI) + bsq_lab) +
               Bi * BS / (val(XI) * (val(XI) + bsq_lab));
        U[i] *= val(LFAC);
    }
    const double bpol = reverse_field ? -1.0 : 1.0;
    out[P_RHO] = static_cast<float>(rho);
    out[P_U1] = static_cast<float>(U[0]);
    out[P_U2] = static_cast<float>(U[1]);
    out[P_U3] = static_cast<float>(U[2]);
    out[P_UU] = static_cast<float>(uu);
    out[P_B1] = static_cast<float>(bpol * B[0]);
    out[P_B2] = static_cast<float>(bpol * B[1]);
    out[P_B3] = static_cast<float>(bpol * B[2]);
}

int clamp_bin_index(double x, double xmin, double dx, int n) {
    int i = static_cast<int>(std::floor((x - xmin) / dx));
    return std::clamp(i, 0, n - 1);
}

size_t flat_bin_index(const MeshBlockIndex& idx, int ix, int iy, int iz) {
    return (static_cast<size_t>(iz) * static_cast<size_t>(idx.ny) + static_cast<size_t>(iy)) *
           static_cast<size_t>(idx.nx) + static_cast<size_t>(ix);
}

void build_meshblock_index(BHACDump& dump, int timing) {
    Kokkos::Timer timer;
    if (dump.blocks.empty()) throw std::runtime_error("BHAC dump contains no leaf blocks");
    MeshBlockIndex idx;
    idx.xmin = dump.blocks.front().xmin; idx.xmax = dump.blocks.front().xmax;
    idx.ymin = dump.blocks.front().ymin; idx.ymax = dump.blocks.front().ymax;
    idx.zmin = dump.blocks.front().zmin; idx.zmax = dump.blocks.front().zmax;
    for (const BHACBlock& b : dump.blocks) {
        idx.xmin = std::min(idx.xmin, b.xmin); idx.xmax = std::max(idx.xmax, b.xmax);
        idx.ymin = std::min(idx.ymin, b.ymin); idx.ymax = std::max(idx.ymax, b.ymax);
        idx.zmin = std::min(idx.zmin, b.zmin); idx.zmax = std::max(idx.zmax, b.zmax);
    }
    const double root = std::cbrt(static_cast<double>(dump.blocks.size()));
    const int nb = std::clamp(static_cast<int>(std::ceil(root * 4.0)), 16, 96);
    idx.nx = nb; idx.ny = nb; idx.nz = nb;
    idx.dx = (idx.xmax - idx.xmin) / double(idx.nx);
    idx.dy = (idx.ymax - idx.ymin) / double(idx.ny);
    idx.dz = (idx.zmax - idx.zmin) / double(idx.nz);
    if (!(idx.dx > 0.0 && idx.dy > 0.0 && idx.dz > 0.0)) {
        throw std::runtime_error("invalid BHAC AMR index extent");
    }
    const size_t nbin = static_cast<size_t>(idx.nx) * static_cast<size_t>(idx.ny) * static_cast<size_t>(idx.nz);
    std::vector<std::vector<int>> bins(nbin);
    for (size_t mb = 0; mb < dump.blocks.size(); ++mb) {
        const BHACBlock& b = dump.blocks[mb];
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
    idx.offsets.resize(nbin + 1, 0);
    size_t total = 0;
    for (size_t b = 0; b < nbin; ++b) {
        idx.offsets[b] = static_cast<int>(total);
        total += bins[b].size();
    }
    idx.offsets[nbin] = static_cast<int>(total);
    idx.candidates.reserve(total);
    for (const auto& bin : bins) idx.candidates.insert(idx.candidates.end(), bin.begin(), bin.end());
    dump.index = std::move(idx);
    if (timing) {
        const double avg = total > 0 ? double(total) / double(nbin) : 0.0;
        std::cout << "timing bhac_meshblock_index " << timer.seconds() << " s bins "
                  << dump.index.nx << 'x' << dump.index.ny << 'x' << dump.index.nz
                  << " blocks " << dump.blocks.size() << " avg_candidates " << avg << "\n";
    }
}

BHACDump read_bhac_dump(const BHACLoadOptions& opt, bool convert_to_prims) {
    if (opt.dump_path.empty()) throw std::runtime_error("--bhac_dump is required for --model=bhac");
    Kokkos::Timer timer;
    std::ifstream in(opt.dump_path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open BHAC dump: " + opt.dump_path);
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    BHACDump dump;
    dump.meta = read_bhac_metadata(in, opt.dump_path);
    report_timing(opt.timing, "bhac_metadata", timer.seconds());
    timer.reset();

    const std::array<double, 3> xmin{double(opt.x1_min), double(opt.x2_min), double(opt.x3_min)};
    const std::array<double, 3> xmax{double(opt.x1_max), double(opt.x2_max), double(opt.x3_max)};

    const int cells = dump.meta.nx[0] * dump.meta.nx[1] * dump.meta.nx[2];
    const std::streamoff prim_bytes = std::streamoff(dump.meta.nleafs) * dump.meta.nw *
                                      std::streamoff(cells) * std::streamoff(8);
    const std::streamoff stag_bytes = std::streamoff(dump.meta.nleafs) *
                                      std::streamoff(dump.meta.nx[0] + 1) *
                                      std::streamoff(dump.meta.nx[1] + 1) *
                                      std::streamoff(dump.meta.nx[2] + 1) *
                                      dump.meta.nws * std::streamoff(8);
    const std::streamoff metadata_bytes = std::streamoff(40 + dump.meta.ndim * 4 + dump.meta.neqpar * 8);
    const std::streamoff metadata_begin = file_size - metadata_bytes;
    const std::streamoff forest_begin = prim_bytes + stag_bytes;
    if (metadata_begin < forest_begin || ((metadata_begin - forest_begin) % 4) != 0) {
        throw std::runtime_error("invalid BHAC forest offset");
    }
    const size_t forest_ints = static_cast<size_t>((metadata_begin - forest_begin) / 4);
    std::vector<int> forest(forest_ints);
    in.seekg(forest_begin, std::ios::beg);
    for (size_t n = 0; n < forest_ints; ++n) {
        forest[n] = read_i32(in);
    }
    const int root_count = infer_root_count(forest, dump.meta.ndim, dump.meta.nleafs);
    const std::array<int, 3> ng = choose_root_dims(dump.meta, opt, root_count);
    const std::array<int, 3> nxlone{
        ng[0] * dump.meta.nx[0],
        ng[1] * dump.meta.nx[1],
        ng[2] * dump.meta.nx[2]};
    const auto roots = root_block_order(ng, opt.sfc);
    if (static_cast<int>(roots.size()) != root_count) {
        throw std::runtime_error("inferred BHAC root grid does not match forest root count");
    }
    dump.blocks.reserve(static_cast<size_t>(dump.meta.nleafs));
    size_t forest_pos = 0;
    for (const auto& root : roots) {
        read_node(forest, forest_pos, dump.blocks, dump.meta.ndim, 1, root[0], root[1], root[2]);
    }
    if (static_cast<int>(dump.blocks.size()) != dump.meta.nleafs || forest_pos != forest.size()) {
        throw std::runtime_error("BHAC forest leaf count does not match metadata; try toggling --bhac_sfc");
    }
    if (opt.timing) {
        std::cout << "timing bhac_root_grid nxlone " << nxlone[0] << 'x' << nxlone[1] << 'x' << nxlone[2]
                  << " roots " << ng[0] << 'x' << ng[1] << 'x' << ng[2]
                  << " forest_ints " << forest.size() << "\n";
    }
    report_timing(opt.timing, "bhac_forest", timer.seconds());
    timer.reset();

    const std::array<double, 3> dx_root{
        (xmax[0] - xmin[0]) / double(ng[0]),
        (xmax[1] - xmin[1]) / double(ng[1]),
        (xmax[2] - xmin[2]) / double(ng[2])};
    const std::array<double, 3> dx_level1{
        (xmax[0] - xmin[0]) / double(nxlone[0]),
        (xmax[1] - xmin[1]) / double(nxlone[1]),
        (xmax[2] - xmin[2]) / double(nxlone[2])};
    for (BHACBlock& block : dump.blocks) {
        const double scale = std::ldexp(1.0, block.level - 1);
        const double lb[3] = {
            xmin[0] + double(block.ind[0]) * dx_root[0] / scale,
            xmin[1] + double(block.ind[1]) * dx_root[1] / scale,
            xmin[2] + double(block.ind[2]) * dx_root[2] / scale};
        const double dx[3] = {dx_level1[0] / scale, dx_level1[1] / scale, dx_level1[2] / scale};
        block.xmin = lb[0]; block.xmax = lb[0] + double(dump.meta.nx[0]) * dx[0];
        block.ymin = lb[1]; block.ymax = lb[1] + double(dump.meta.nx[1]) * dx[1];
        block.zmin = lb[2]; block.zmax = lb[2] + double(dump.meta.nx[2]) * dx[2];
    }

    const int spin_index = choose_spin_index(dump.meta, opt);
    const double spin = dump.meta.eqpar[static_cast<size_t>(spin_index)];
    const double gam = dump.meta.eqpar[0];
    if (opt.timing) {
        std::cout << "timing bhac_eqpar gamma " << gam << " spin_index " << spin_index
                  << " spin " << spin << "\n";
    }
    const size_t values_per_block = static_cast<size_t>(dump.meta.nw) * static_cast<size_t>(cells);
    const std::streamoff stagger_per_block = std::streamoff(dump.meta.nx[0] + 1) *
                                             std::streamoff(dump.meta.nx[1] + 1) *
                                             std::streamoff(dump.meta.nx[2] + 1) *
                                             dump.meta.nws * std::streamoff(8);
    in.seekg(0, std::ios::beg);
    for (BHACBlock& block : dump.blocks) {
        block.conserved.resize(values_per_block);
        in.read(reinterpret_cast<char*>(block.conserved.data()),
                static_cast<std::streamsize>(values_per_block * sizeof(double)));
        if (!in) {
            throw std::runtime_error("failed reading BHAC conserved block data");
        }
        if (convert_to_prims) {
            const double lb[3] = {block.xmin, block.ymin, block.zmin};
            const double dx[3] = {
                (block.xmax - block.xmin) / double(dump.meta.nx[0]),
                (block.ymax - block.ymin) / double(dump.meta.nx[1]),
                (block.zmax - block.zmin) / double(dump.meta.nx[2])};
            block.prims.resize(static_cast<size_t>(8) * static_cast<size_t>(cells), 0.0f);
            for (int c = 0; c < cells; ++c) {
                double X[3], Xbar[3];
                cell_center(c, dump.meta.nx, lb, dx, X);
                volume_center(spin, opt.hslope, X, dx, Xbar);
                float prim[8];
                conserved_to_prims(block.conserved, c, cells, dump.meta.nw, spin, gam, opt.hslope,
                                   Xbar, opt.reverse_field, prim);
                for (int v = 0; v < 8; ++v) {
                    block.prims[static_cast<size_t>(v) * static_cast<size_t>(cells) + static_cast<size_t>(c)] = prim[v];
                }
            }
        }
        in.seekg(stagger_per_block, std::ios::cur);
    }
    report_timing(opt.timing, convert_to_prims ? "bhac_binary_read_convert" : "bhac_binary_read_raw",
                  timer.seconds());
    build_meshblock_index(dump, opt.timing);
    return dump;
}

} // namespace

DefaultReal read_bhac_dump_time(const std::string& dump_path) {
    std::ifstream in(dump_path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open BHAC dump: " + dump_path);
    return DefaultReal(read_bhac_metadata(in, dump_path).time);
}

BHACStagedDump read_bhac_staged_dump(const BHACLoadOptions& opt) {
    Kokkos::Timer timer;
    BHACDump dump = read_bhac_dump(opt, true);
    BHACStagedDump staged;
    staged.nblocks = static_cast<int>(dump.blocks.size());
    staged.nvar = 8;
    staged.nx1 = dump.meta.nx[0];
    staged.nx2 = dump.meta.nx[1];
    staged.nx3 = dump.meta.nx[2];
    staged.spin = DefaultReal(metadata_spin(dump.meta, opt));
    staged.gamma = DefaultReal(dump.meta.eqpar[0]);
    staged.time = DefaultReal(dump.meta.time);
    staged.startx1 = opt.x1_min;
    staged.startx2 = opt.x2_min;
    staged.startx3 = opt.x3_min;
    staged.stopx1 = opt.x1_max;
    staged.stopx2 = opt.x2_max;
    staged.stopx3 = opt.x3_max;
    staged.hslope = opt.hslope;
    const size_t cells_per_block = static_cast<size_t>(staged.nx1) *
                                   static_cast<size_t>(staged.nx2) *
                                   static_cast<size_t>(staged.nx3);
    const size_t values_per_block = static_cast<size_t>(staged.nvar) * cells_per_block;
    staged.extents.resize(static_cast<size_t>(staged.nblocks) * 6u);
    staged.cell_geom.resize(static_cast<size_t>(staged.nblocks) * 6u);
    staged.prims.resize(static_cast<size_t>(staged.nblocks) * values_per_block);
    staged.levels.resize(static_cast<size_t>(staged.nblocks));
    for (size_t mb = 0; mb < dump.blocks.size(); ++mb) {
        const BHACBlock& b = dump.blocks[mb];
        staged.extents[mb * 6u + 0u] = DefaultReal(b.xmin);
        staged.extents[mb * 6u + 1u] = DefaultReal(b.xmax);
        staged.extents[mb * 6u + 2u] = DefaultReal(b.ymin);
        staged.extents[mb * 6u + 3u] = DefaultReal(b.ymax);
        staged.extents[mb * 6u + 4u] = DefaultReal(b.zmin);
        staged.extents[mb * 6u + 5u] = DefaultReal(b.zmax);
        staged.cell_geom[mb * 6u + 0u] = DefaultReal(b.xmin);
        staged.cell_geom[mb * 6u + 1u] = DefaultReal(b.ymin);
        staged.cell_geom[mb * 6u + 2u] = DefaultReal(b.zmin);
        staged.cell_geom[mb * 6u + 3u] = DefaultReal(staged.nx1) / DefaultReal(b.xmax - b.xmin);
        staged.cell_geom[mb * 6u + 4u] = DefaultReal(staged.nx2) / DefaultReal(b.ymax - b.ymin);
        staged.cell_geom[mb * 6u + 5u] = DefaultReal(staged.nx3) / DefaultReal(b.zmax - b.zmin);
        staged.levels[mb] = b.level;
        std::copy(b.prims.begin(), b.prims.end(),
                  staged.prims.begin() + static_cast<std::ptrdiff_t>(mb * values_per_block));
    }
    staged.index_offsets = dump.index.offsets;
    staged.index_candidates = dump.index.candidates;
    staged.index_nx = dump.index.nx;
    staged.index_ny = dump.index.ny;
    staged.index_nz = dump.index.nz;
    staged.index_xmin = DefaultReal(dump.index.xmin);
    staged.index_xmax = DefaultReal(dump.index.xmax);
    staged.index_ymin = DefaultReal(dump.index.ymin);
    staged.index_ymax = DefaultReal(dump.index.ymax);
    staged.index_zmin = DefaultReal(dump.index.zmin);
    staged.index_zmax = DefaultReal(dump.index.zmax);
    staged.index_dx = DefaultReal(dump.index.dx);
    staged.index_dy = DefaultReal(dump.index.dy);
    staged.index_dz = DefaultReal(dump.index.dz);
    report_timing(opt.timing, "bhac_host_flatten", timer.seconds());
    return staged;
}

BHACRawStagedDump read_bhac_raw_staged_dump(const BHACLoadOptions& opt) {
    Kokkos::Timer timer;
    BHACDump dump = read_bhac_dump(opt, false);
    BHACRawStagedDump raw;
    raw.nblocks = static_cast<int>(dump.blocks.size());
    raw.nw = dump.meta.nw;
    raw.nx1 = dump.meta.nx[0];
    raw.nx2 = dump.meta.nx[1];
    raw.nx3 = dump.meta.nx[2];
    raw.spin = DefaultReal(metadata_spin(dump.meta, opt));
    raw.gamma = DefaultReal(dump.meta.eqpar[0]);
    raw.time = DefaultReal(dump.meta.time);
    raw.startx1 = opt.x1_min;
    raw.startx2 = opt.x2_min;
    raw.startx3 = opt.x3_min;
    raw.stopx1 = opt.x1_max;
    raw.stopx2 = opt.x2_max;
    raw.stopx3 = opt.x3_max;
    raw.hslope = opt.hslope;
    const size_t cells_per_block = static_cast<size_t>(raw.nx1) *
                                   static_cast<size_t>(raw.nx2) *
                                   static_cast<size_t>(raw.nx3);
    const size_t raw_values_per_block = static_cast<size_t>(raw.nw) * cells_per_block;
    raw.extents.resize(static_cast<size_t>(raw.nblocks) * 6u);
    raw.cell_geom.resize(static_cast<size_t>(raw.nblocks) * 6u);
    raw.conserved.resize(static_cast<size_t>(raw.nblocks) * raw_values_per_block);
    raw.levels.resize(static_cast<size_t>(raw.nblocks));
    for (size_t mb = 0; mb < dump.blocks.size(); ++mb) {
        const BHACBlock& b = dump.blocks[mb];
        raw.extents[mb * 6u + 0u] = DefaultReal(b.xmin);
        raw.extents[mb * 6u + 1u] = DefaultReal(b.xmax);
        raw.extents[mb * 6u + 2u] = DefaultReal(b.ymin);
        raw.extents[mb * 6u + 3u] = DefaultReal(b.ymax);
        raw.extents[mb * 6u + 4u] = DefaultReal(b.zmin);
        raw.extents[mb * 6u + 5u] = DefaultReal(b.zmax);
        raw.cell_geom[mb * 6u + 0u] = DefaultReal(b.xmin);
        raw.cell_geom[mb * 6u + 1u] = DefaultReal(b.ymin);
        raw.cell_geom[mb * 6u + 2u] = DefaultReal(b.zmin);
        raw.cell_geom[mb * 6u + 3u] = DefaultReal(raw.nx1) / DefaultReal(b.xmax - b.xmin);
        raw.cell_geom[mb * 6u + 4u] = DefaultReal(raw.nx2) / DefaultReal(b.ymax - b.ymin);
        raw.cell_geom[mb * 6u + 5u] = DefaultReal(raw.nx3) / DefaultReal(b.zmax - b.zmin);
        raw.levels[mb] = b.level;
        if (b.conserved.size() != raw_values_per_block) {
            throw std::runtime_error("BHAC raw block conserved vector size mismatch");
        }
        std::copy(b.conserved.begin(), b.conserved.end(),
                  raw.conserved.begin() + static_cast<std::ptrdiff_t>(mb * raw_values_per_block));
    }
    raw.index_offsets = dump.index.offsets;
    raw.index_candidates = dump.index.candidates;
    raw.index_nx = dump.index.nx;
    raw.index_ny = dump.index.ny;
    raw.index_nz = dump.index.nz;
    raw.index_xmin = DefaultReal(dump.index.xmin);
    raw.index_xmax = DefaultReal(dump.index.xmax);
    raw.index_ymin = DefaultReal(dump.index.ymin);
    raw.index_ymax = DefaultReal(dump.index.ymax);
    raw.index_zmin = DefaultReal(dump.index.zmin);
    raw.index_zmax = DefaultReal(dump.index.zmax);
    raw.index_dx = DefaultReal(dump.index.dx);
    raw.index_dy = DefaultReal(dump.index.dy);
    raw.index_dz = DefaultReal(dump.index.dz);
    report_timing(opt.timing, "bhac_host_flatten_raw", timer.seconds());
    return raw;
}

void write_bhac_staged_cache(const std::string& path,
                             const BHACStagedDump& staged,
                             const BHACLoadOptions& opt) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open BHAC staged cache for write: " + path);
    out.write(BHAC_CACHE_MAGIC.data(), static_cast<std::streamsize>(BHAC_CACHE_MAGIC.size()));
    write_pod(out, BHAC_CACHE_VERSION);
    write_pod(out, BHAC_CACHE_ENDIAN);
    write_pod(out, static_cast<std::uint32_t>(sizeof(DefaultReal)));
    write_pod(out, static_cast<std::uint32_t>(sizeof(float)));
    write_pod(out, static_cast<std::uint32_t>(sizeof(int)));
    write_pod(out, file_size_bytes(opt.dump_path));
    write_string(out, opt.dump_path);

    write_pod(out, opt.nxlone1);
    write_pod(out, opt.nxlone2);
    write_pod(out, opt.nxlone3);
    write_pod(out, opt.spin_index);
    write_pod(out, opt.sfc);
    write_pod(out, opt.reverse_field);
    write_pod(out, opt.x1_min);
    write_pod(out, opt.x1_max);
    write_pod(out, opt.x2_min);
    write_pod(out, opt.x2_max);
    write_pod(out, opt.x3_min);
    write_pod(out, opt.x3_max);
    write_pod(out, opt.hslope);

    write_pod(out, staged.nblocks);
    write_pod(out, staged.nvar);
    write_pod(out, staged.nx1);
    write_pod(out, staged.nx2);
    write_pod(out, staged.nx3);
    write_pod(out, staged.spin);
    write_pod(out, staged.gamma);
    write_pod(out, staged.time);
    write_pod(out, staged.startx1);
    write_pod(out, staged.startx2);
    write_pod(out, staged.startx3);
    write_pod(out, staged.stopx1);
    write_pod(out, staged.stopx2);
    write_pod(out, staged.stopx3);
    write_pod(out, staged.hslope);
    write_pod(out, staged.index_nx);
    write_pod(out, staged.index_ny);
    write_pod(out, staged.index_nz);
    write_pod(out, staged.index_xmin);
    write_pod(out, staged.index_xmax);
    write_pod(out, staged.index_ymin);
    write_pod(out, staged.index_ymax);
    write_pod(out, staged.index_zmin);
    write_pod(out, staged.index_zmax);
    write_pod(out, staged.index_dx);
    write_pod(out, staged.index_dy);
    write_pod(out, staged.index_dz);

    write_vector(out, staged.extents);
    write_vector(out, staged.cell_geom);
    write_vector(out, staged.prims);
    write_vector(out, staged.levels);
    write_vector(out, staged.index_offsets);
    write_vector(out, staged.index_candidates);
}

BHACStagedDump read_bhac_staged_cache(const std::string& path,
                                      const BHACLoadOptions& opt) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open BHAC staged cache for read: " + path);
    std::array<char, 16> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != BHAC_CACHE_MAGIC) {
        throw std::runtime_error("invalid BHAC staged cache magic");
    }
    const std::uint32_t version = read_pod<std::uint32_t>(in);
    const std::uint32_t endian = read_pod<std::uint32_t>(in);
    const std::uint32_t real_size = read_pod<std::uint32_t>(in);
    const std::uint32_t float_size = read_pod<std::uint32_t>(in);
    const std::uint32_t int_size = read_pod<std::uint32_t>(in);
    if (version != BHAC_CACHE_VERSION || endian != BHAC_CACHE_ENDIAN ||
        real_size != sizeof(DefaultReal) || float_size != sizeof(float) || int_size != sizeof(int)) {
        throw std::runtime_error("BHAC staged cache ABI/version mismatch");
    }
    const std::uint64_t cached_source_size = read_pod<std::uint64_t>(in);
    (void)read_string(in);
    if (!opt.dump_path.empty()) {
        const std::uint64_t current_source_size = file_size_bytes(opt.dump_path);
        if (current_source_size != 0 && cached_source_size != current_source_size) {
            throw std::runtime_error("BHAC staged cache source file size mismatch");
        }
    }

    const int nxlone1 = read_pod<int>(in);
    const int nxlone2 = read_pod<int>(in);
    const int nxlone3 = read_pod<int>(in);
    const int spin_index = read_pod<int>(in);
    const int sfc = read_pod<int>(in);
    const int reverse_field = read_pod<int>(in);
    const DefaultReal x1_min = read_pod<DefaultReal>(in);
    const DefaultReal x1_max = read_pod<DefaultReal>(in);
    const DefaultReal x2_min = read_pod<DefaultReal>(in);
    const DefaultReal x2_max = read_pod<DefaultReal>(in);
    const DefaultReal x3_min = read_pod<DefaultReal>(in);
    const DefaultReal x3_max = read_pod<DefaultReal>(in);
    const DefaultReal hslope = read_pod<DefaultReal>(in);
    if (nxlone1 != opt.nxlone1 || nxlone2 != opt.nxlone2 || nxlone3 != opt.nxlone3 ||
        spin_index != opt.spin_index || sfc != opt.sfc || reverse_field != opt.reverse_field ||
        !same_real(x1_min, opt.x1_min) || !same_real(x1_max, opt.x1_max) ||
        !same_real(x2_min, opt.x2_min) || !same_real(x2_max, opt.x2_max) ||
        !same_real(x3_min, opt.x3_min) || !same_real(x3_max, opt.x3_max) ||
        !same_real(hslope, opt.hslope)) {
        throw std::runtime_error("BHAC staged cache options mismatch");
    }

    BHACStagedDump staged;
    staged.nblocks = read_pod<int>(in);
    staged.nvar = read_pod<int>(in);
    staged.nx1 = read_pod<int>(in);
    staged.nx2 = read_pod<int>(in);
    staged.nx3 = read_pod<int>(in);
    staged.spin = read_pod<DefaultReal>(in);
    staged.gamma = read_pod<DefaultReal>(in);
    staged.time = read_pod<DefaultReal>(in);
    staged.startx1 = read_pod<DefaultReal>(in);
    staged.startx2 = read_pod<DefaultReal>(in);
    staged.startx3 = read_pod<DefaultReal>(in);
    staged.stopx1 = read_pod<DefaultReal>(in);
    staged.stopx2 = read_pod<DefaultReal>(in);
    staged.stopx3 = read_pod<DefaultReal>(in);
    staged.hslope = read_pod<DefaultReal>(in);
    staged.index_nx = read_pod<int>(in);
    staged.index_ny = read_pod<int>(in);
    staged.index_nz = read_pod<int>(in);
    staged.index_xmin = read_pod<DefaultReal>(in);
    staged.index_xmax = read_pod<DefaultReal>(in);
    staged.index_ymin = read_pod<DefaultReal>(in);
    staged.index_ymax = read_pod<DefaultReal>(in);
    staged.index_zmin = read_pod<DefaultReal>(in);
    staged.index_zmax = read_pod<DefaultReal>(in);
    staged.index_dx = read_pod<DefaultReal>(in);
    staged.index_dy = read_pod<DefaultReal>(in);
    staged.index_dz = read_pod<DefaultReal>(in);

    staged.extents = read_vector<DefaultReal>(in);
    staged.cell_geom = read_vector<DefaultReal>(in);
    staged.prims = read_vector<float>(in);
    staged.levels = read_vector<int>(in);
    staged.index_offsets = read_vector<int>(in);
    staged.index_candidates = read_vector<int>(in);

    if (!(staged.nblocks > 0 && staged.nvar == 8 && staged.nx1 > 1 && staged.nx2 > 1 && staged.nx3 > 1)) {
        throw std::runtime_error("invalid BHAC staged cache dimensions");
    }
    const size_t nblocks = static_cast<size_t>(staged.nblocks);
    const size_t cells_per_block = static_cast<size_t>(staged.nx1) *
                                   static_cast<size_t>(staged.nx2) *
                                   static_cast<size_t>(staged.nx3);
    if (staged.extents.size() != nblocks * 6u ||
        staged.cell_geom.size() != nblocks * 6u ||
        staged.levels.size() != nblocks ||
        staged.prims.size() != nblocks * static_cast<size_t>(staged.nvar) * cells_per_block) {
        throw std::runtime_error("BHAC staged cache vector size mismatch");
    }
    const size_t nbin = static_cast<size_t>(staged.index_nx) *
                        static_cast<size_t>(staged.index_ny) *
                        static_cast<size_t>(staged.index_nz);
    if (!(staged.index_nx > 0 && staged.index_ny > 0 && staged.index_nz > 0) ||
        staged.index_offsets.size() != nbin + 1u) {
        throw std::runtime_error("BHAC staged cache AMR index mismatch");
    }
    return staged;
}

BHACStagedDump make_staged_shell_from_raw(const BHACRawStagedDump& raw) {
    BHACStagedDump staged;
    staged.nblocks = raw.nblocks;
    staged.nvar = 8;
    staged.nx1 = raw.nx1;
    staged.nx2 = raw.nx2;
    staged.nx3 = raw.nx3;
    staged.spin = raw.spin;
    staged.gamma = raw.gamma;
    staged.time = raw.time;
    staged.startx1 = raw.startx1;
    staged.startx2 = raw.startx2;
    staged.startx3 = raw.startx3;
    staged.stopx1 = raw.stopx1;
    staged.stopx2 = raw.stopx2;
    staged.stopx3 = raw.stopx3;
    staged.hslope = raw.hslope;
    staged.extents = raw.extents;
    staged.cell_geom = raw.cell_geom;
    staged.levels = raw.levels;
    staged.index_offsets = raw.index_offsets;
    staged.index_candidates = raw.index_candidates;
    staged.index_nx = raw.index_nx;
    staged.index_ny = raw.index_ny;
    staged.index_nz = raw.index_nz;
    staged.index_xmin = raw.index_xmin;
    staged.index_xmax = raw.index_xmax;
    staged.index_ymin = raw.index_ymin;
    staged.index_ymax = raw.index_ymax;
    staged.index_zmin = raw.index_zmin;
    staged.index_zmax = raw.index_zmax;
    staged.index_dx = raw.index_dx;
    staged.index_dy = raw.index_dy;
    staged.index_dz = raw.index_dz;
    return staged;
}

BHACAMRRadiationModel<DefaultReal> materialize_bhac_model_from_staged_impl(const BHACStagedDump& staged,
                                                                           const BHACLoadOptions& opt,
                                                                           bool pack_derived) {
    Kokkos::Timer total_timer;
    Kokkos::Timer pack_timer;
    BHACAMRRadiationModel<RealT> model;
    model.nblocks = staged.nblocks;
    model.nvar = staged.nvar;
    model.nx1 = staged.nx1;
    model.nx2 = staged.nx2;
    model.nx3 = staged.nx3;
    model.spin = staged.spin;
    model.gam = opt.gamma > DefaultReal(0) ? opt.gamma : staged.gamma;
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
    model.data_coordinate_system = static_cast<int>(CoordinateSystem::MKS);
    model.radial_coordinate_log = 1;
    model.native_coordinate_transform = 1;
    model.startx1 = staged.startx1;
    model.startx2 = staged.startx2;
    model.startx3 = staged.startx3;
    model.stopx1 = staged.stopx1;
    model.stopx2 = staged.stopx2;
    model.stopx3 = staged.stopx3;
    model.hslope = staged.hslope;
    const RealT rh = RealT(1) + Kokkos::sqrt(max_val(RealT(0), RealT(1) - model.spin * model.spin));
    model.r_in = opt.r_in > RealT(0) ? opt.r_in : std::max(RealT(0.9) * rh, Kokkos::exp(model.startx1));
    model.r_out = opt.r_out > RealT(0) ? opt.r_out : Kokkos::exp(model.stopx1);

    const size_t nblocks = static_cast<size_t>(model.nblocks);
    const size_t cells_per_block = static_cast<size_t>(model.nx1) *
                                   static_cast<size_t>(model.nx2) *
                                   static_cast<size_t>(model.nx3);
    const size_t values_per_block = static_cast<size_t>(model.nvar) * cells_per_block;
    model.extents = typename BHACAMRRadiationModel<RealT>::RealView("bhac_mks_extents", nblocks * 6u);
    model.cell_geom = typename BHACAMRRadiationModel<RealT>::RealView("bhac_mks_cell_geom", nblocks * 6u);
    model.prims = typename BHACAMRRadiationModel<RealT>::FloatView("bhac_mks_prims", nblocks * values_per_block);
    const size_t derived_per_block = static_cast<size_t>(BHACAMRRadiationModel<RealT>::NumDerived) * cells_per_block;
    model.derived = typename BHACAMRRadiationModel<RealT>::FloatView("bhac_mks_derived", nblocks * derived_per_block);
    model.levels = typename BHACAMRRadiationModel<RealT>::IntView("bhac_mks_levels", nblocks);
    model.index_offsets = typename BHACAMRRadiationModel<RealT>::IntView("bhac_mks_index_offsets", staged.index_offsets.size());
    model.index_candidates = typename BHACAMRRadiationModel<RealT>::IntView("bhac_mks_index_candidates", staged.index_candidates.size());

    using HostRealView = Kokkos::View<const RealT*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using HostFloatView = Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using HostIntView = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    if (!staged.extents.empty()) Kokkos::deep_copy(model.extents, HostRealView(staged.extents.data(), staged.extents.size()));
    if (!staged.cell_geom.empty()) Kokkos::deep_copy(model.cell_geom, HostRealView(staged.cell_geom.data(), staged.cell_geom.size()));
    if (!staged.prims.empty()) Kokkos::deep_copy(model.prims, HostFloatView(staged.prims.data(), staged.prims.size()));
    if (!staged.levels.empty()) Kokkos::deep_copy(model.levels, HostIntView(staged.levels.data(), staged.levels.size()));
    if (!staged.index_offsets.empty()) Kokkos::deep_copy(model.index_offsets, HostIntView(staged.index_offsets.data(), staged.index_offsets.size()));
    if (!staged.index_candidates.empty()) Kokkos::deep_copy(model.index_candidates, HostIntView(staged.index_candidates.data(), staged.index_candidates.size()));
    model.index_nx = staged.index_nx;
    model.index_ny = staged.index_ny;
    model.index_nz = staged.index_nz;
    model.index_xmin = staged.index_xmin;
    model.index_xmax = staged.index_xmax;
    model.index_ymin = staged.index_ymin;
    model.index_ymax = staged.index_ymax;
    model.index_zmin = staged.index_zmin;
    model.index_zmax = staged.index_zmax;
    model.index_dx = staged.index_dx;
    model.index_dy = staged.index_dy;
    model.index_dz = staged.index_dz;
    model.index_inv_dx = RealT(1) / model.index_dx;
    model.index_inv_dy = RealT(1) / model.index_dy;
    model.index_inv_dz = RealT(1) / model.index_dz;

    if (pack_derived) {
        BHACAMRRadiationModel<RealT> device_model = model;
        const size_t nx1s = static_cast<size_t>(model.nx1);
        const size_t nx2s = static_cast<size_t>(model.nx2);
        const size_t nvars = static_cast<size_t>(model.nvar);
        const size_t num_derived = static_cast<size_t>(BHACAMRRadiationModel<RealT>::NumDerived);
        const RealT rho_unit = model.rho_unit_cgs();
        const RealT b_unit = model.b_unit_cgs();
        const RealT trat_small = model.trat_small;
        const RealT trat_large = model.trat_large;
        const RealT beta_crit2 = max_val(model.beta_crit * model.beta_crit, RealT(1e-40));
        Kokkos::parallel_for(
            "KPOLARISBHACDerivedPack",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<size_t>>(0, nblocks * cells_per_block),
            KOKKOS_LAMBDA(const size_t n) {
            const size_t mb = n / cells_per_block;
            const size_t local = n - mb * cells_per_block;
            const int i = static_cast<int>(local % nx1s);
            const int j = static_cast<int>((local / nx1s) % nx2s);
            const int k = static_cast<int>(local / (nx1s * nx2s));
            const size_t ebase = mb * 6u;
            const RealT x1 = device_model.cell_geom(ebase + 0u) +
                             (RealT(i) + RealT(0.5)) / device_model.cell_geom(ebase + 3u);
            const RealT x2 = device_model.cell_geom(ebase + 1u) +
                             (RealT(j) + RealT(0.5)) / device_model.cell_geom(ebase + 4u);
            const RealT r = Kokkos::exp(x1);
            const RealT th = device_model.bhac_theta_from_x2(x2);
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
                RealT gcov_nat[ndim][ndim];
                RealT gcon0_nat[ndim];
                device_model.bhac_native_metric(x1, x2, r, th, gcov_nat, gcon0_nat);
                const Vec4<RealT> vnat(RealT(0), prim(P_U1), prim(P_U2), prim(P_U3));
                RealT spatial_norm = RealT(0);
                for (int a = 1; a < ndim; ++a) {
                    for (int b = 1; b < ndim; ++b) spatial_norm += gcov_nat[a][b] * vnat[a] * vnat[b];
                }
                const RealT vfac = Kokkos::sqrt(max_val(-RealT(1) / gcon0_nat[0] *
                                                        (RealT(1) + abs_val(spatial_norm)), RealT(0)));
                Vec4<RealT> ucon_nat;
                ucon_nat[0] = -vfac * gcon0_nat[0];
                for (int a = 1; a < ndim; ++a) ucon_nat[a] = vnat[a] - vfac * gcon0_nat[a];
                const Vec4<RealT> ucov_nat = device_model.lower_with_matrix(gcov_nat, ucon_nat);
                const Vec4<RealT> Bnat(RealT(0), prim(P_B1), prim(P_B2), prim(P_B3));
                RealT udotB = RealT(0);
                for (int a = 1; a < ndim; ++a) udotB += ucov_nat[a] * Bnat[a];
                Vec4<RealT> bcon_nat;
                bcon_nat[0] = udotB;
                const RealT inv_u0 = RealT(1) / max_val(ucon_nat[0], RealT(1e-300));
                for (int a = 1; a < ndim; ++a) bcon_nat[a] = (Bnat[a] + ucon_nat[a] * udotB) * inv_u0;
                const Vec4<RealT> bcov_nat = device_model.lower_with_matrix(gcov_nat, bcon_nat);
                RealT bsq = RealT(0);
                for (int mu = 0; mu < ndim; ++mu) bsq += bcon_nat[mu] * bcov_nat[mu];
                bsq = max_val(abs_val(bsq), RealT(1e-40));
                sigma = max_val(bsq / max_val(rho, RealT(1e-300)), RealT(1e-300));
                beta = max_val(uu * (device_model.gam - RealT(1)) / (RealT(0.5) * bsq), RealT(1e-300));
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
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedNe) * var_stride + cell) = static_cast<float>(ne);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedThetae) * var_stride + cell) = static_cast<float>(thetae);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedB) * var_stride + cell) = static_cast<float>(b_cgs);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedSigma) * var_stride + cell) = static_cast<float>(sigma);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedBeta) * var_stride + cell) = static_cast<float>(beta);
            });
        Kokkos::fence();
        report_timing(opt.timing, "bhac_direct_device_pack", pack_timer.seconds());
    }
    report_timing(opt.timing, "bhac_materialize_total", total_timer.seconds());
    return model;
}

BHACAMRRadiationModel<DefaultReal> materialize_bhac_model_from_staged(const BHACStagedDump& staged,
                                                                       const BHACLoadOptions& opt) {
    return materialize_bhac_model_from_staged_impl(staged, opt, true);
}

BHACAMRRadiationModel<DefaultReal> materialize_bhac_model_from_raw(const BHACRawStagedDump& raw,
                                                                    const BHACLoadOptions& opt) {
    BHACStagedDump staged = make_staged_shell_from_raw(raw);
    BHACAMRRadiationModel<RealT> model = materialize_bhac_model_from_staged_impl(staged, opt, false);
    Kokkos::Timer timer;

    const size_t nblocks = static_cast<size_t>(model.nblocks);
    const size_t cells_per_block = static_cast<size_t>(model.nx1) *
                                   static_cast<size_t>(model.nx2) *
                                   static_cast<size_t>(model.nx3);
    const size_t raw_values_per_block = static_cast<size_t>(raw.nw) * cells_per_block;
    if (raw.conserved.size() != nblocks * raw_values_per_block) {
        throw std::runtime_error("BHAC raw conserved vector size mismatch");
    }

    Kokkos::View<double*> conserved_device("bhac_raw_conserved", raw.conserved.size());
    using HostDoubleView = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    if (!raw.conserved.empty()) {
        Kokkos::deep_copy(conserved_device, HostDoubleView(raw.conserved.data(), raw.conserved.size()));
    }

    BHACAMRRadiationModel<RealT> device_model = model;
    const int raw_nw = raw.nw;
    const int reverse_field = opt.reverse_field;
    const RealT spin = model.spin;
    const RealT gam = model.gam;
    const RealT hslope = model.hslope;
    const size_t nx1s = static_cast<size_t>(model.nx1);
    const size_t nx2s = static_cast<size_t>(model.nx2);
    const size_t nvars = static_cast<size_t>(model.nvar);
    const size_t num_derived = static_cast<size_t>(BHACAMRRadiationModel<RealT>::NumDerived);
    const RealT rho_unit = model.rho_unit_cgs();
    const RealT b_unit = model.b_unit_cgs();
    const RealT trat_small = model.trat_small;
    const RealT trat_large = model.trat_large;
    const RealT beta_crit2 = max_val(model.beta_crit * model.beta_crit, RealT(1e-40));

    Kokkos::parallel_for(
        "KPOLARISBHACRawConvertPack",
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
            const RealT dx1 = RealT(1) / device_model.cell_geom(ebase + 3u);
            const RealT dx2 = RealT(1) / device_model.cell_geom(ebase + 4u);
            const RealT dx3 = RealT(1) / device_model.cell_geom(ebase + 5u);
            const RealT x1 = xmin + (RealT(i) + RealT(0.5)) * dx1;
            const RealT x2 = ymin + (RealT(j) + RealT(0.5)) * dx2;
            const RealT x3 = zmin + (RealT(k) + RealT(0.5)) * dx3;
            const size_t cell = (static_cast<size_t>(k) * nx2s + static_cast<size_t>(j)) * nx1s +
                                static_cast<size_t>(i);
            const size_t raw_block = mb * static_cast<size_t>(raw_nw) * cells_per_block;
            auto raw_val = [&](int v) -> RealT {
                return static_cast<RealT>(conserved_device(raw_block + static_cast<size_t>(v) * cells_per_block + cell));
            };

            RealT xbar0 = RealT(0);
            RealT xbar1 = RealT(0);
            RealT xbar2 = RealT(0);
            RealT norm = RealT(0);
            for (int qk = 0; qk < 3; ++qk) {
                const RealT ok = qk == 0 ? RealT(-1) : (qk == 1 ? RealT(0) : RealT(1));
                const RealT wk = qk == 1 ? RealT(4) : RealT(1);
                for (int qj = 0; qj < 3; ++qj) {
                    const RealT oj = qj == 0 ? RealT(-1) : (qj == 1 ? RealT(0) : RealT(1));
                    const RealT wj = qj == 1 ? RealT(4) : RealT(1);
                    for (int qi = 0; qi < 3; ++qi) {
                        const RealT oi = qi == 0 ? RealT(-1) : (qi == 1 ? RealT(0) : RealT(1));
                        const RealT wi = qi == 1 ? RealT(4) : RealT(1);
                        const RealT xq0 = x1 + RealT(0.5) * dx1 * oi;
                        const RealT xq1 = x2 + RealT(0.5) * dx2 * oj;
                        const RealT xq2 = x3 + RealT(0.5) * dx3 * ok;
                        const RealT rq = Kokkos::exp(xq0);
                        const RealT thq = xq1 + RealT(0.5) * hslope * Kokkos::sin(RealT(2) * xq1);
                        const RealT hfacq = RealT(1) + hslope * Kokkos::cos(RealT(2) * xq1);
                        const RealT cthq = Kokkos::cos(thq);
                        const RealT sthq = Kokkos::sin(thq);
                        const RealT rho2q = rq * rq + spin * spin * cthq * cthq;
                        const RealT det = abs_val(rq * hfacq * sthq) *
                                          Kokkos::sqrt(max_val(rho2q * (rho2q + RealT(2) * rq), RealT(0)));
                        const RealT w = wi * wj * wk * det;
                        norm += w;
                        xbar0 += w * xq0;
                        xbar1 += w * xq1;
                        xbar2 += w * xq2;
                    }
                }
            }
            if (norm > RealT(0)) {
                xbar0 /= norm;
                xbar1 /= norm;
                xbar2 /= norm;
            } else {
                xbar0 = x1;
                xbar1 = x2;
                xbar2 = x3;
            }

            RealT p[8];
            for (int v = 0; v < 8; ++v) p[v] = RealT(0);
            const int LFAC = raw_nw - 2;
            const int XI = raw_nw - 1;
            if (raw_nw > DS && Kokkos::exp(xbar0) >= RealT(1) && raw_val(LFAC) > RealT(0) && raw_val(XI) > RealT(0)) {
                const RealT r = Kokkos::exp(xbar0);
                const RealT th = xbar1 + RealT(0.5) * hslope * Kokkos::sin(RealT(2) * xbar1);
                const RealT hfac = RealT(1) + hslope * Kokkos::cos(RealT(2) * xbar1);
                const RealT cth = Kokkos::cos(th);
                const RealT sth = Kokkos::sin(th);
                const RealT s2 = sth * sth;
                const RealT a2 = spin * spin;
                const RealT rho2 = r * r + a2 * cth * cth;
                const RealT f = RealT(2) * r / rho2;
                const RealT delta = r * r - RealT(2) * r + a2;
                RealT gcov[4][4];
                RealT gcon[4][4];
                for (int mu = 0; mu < 4; ++mu) {
                    for (int nu = 0; nu < 4; ++nu) {
                        gcov[mu][nu] = RealT(0);
                        gcon[mu][nu] = RealT(0);
                    }
                }
                gcov[0][0] = RealT(-1) + f;
                gcov[0][1] = f * r;
                gcov[1][0] = gcov[0][1];
                gcov[0][3] = -RealT(2) * spin * r * s2 / rho2;
                gcov[3][0] = gcov[0][3];
                gcov[1][1] = (RealT(1) + f) * r * r;
                gcov[1][3] = -spin * s2 * (RealT(1) + f) * r;
                gcov[3][1] = gcov[1][3];
                gcov[2][2] = rho2 * hfac * hfac;
                gcov[3][3] = s2 * (rho2 + a2 * s2 * (RealT(1) + f));
                gcon[0][0] = -(RealT(1) + f);
                gcon[0][1] = f / max_val(r, RealT(1e-300));
                gcon[1][0] = gcon[0][1];
                gcon[1][1] = delta / (rho2 * max_val(r * r, RealT(1e-300)));
                gcon[1][3] = spin / (rho2 * max_val(r, RealT(1e-300)));
                gcon[3][1] = gcon[1][3];
                gcon[2][2] = RealT(1) / (rho2 * max_val(hfac * hfac, RealT(1e-300)));
                gcon[3][3] = RealT(1) / (rho2 * max_val(s2, RealT(1e-300)));

                const RealT B[3] = {raw_val(P_B1), raw_val(P_B2), raw_val(P_B3)};
                const RealT S[3] = {raw_val(S1), raw_val(S2), raw_val(S3)};
                const RealT BS = S[0] * B[0] + S[1] * B[1] + S[2] * B[2];
                RealT Bcov[4];
                RealT Scon[4];
                for (int a = 0; a < 4; ++a) {
                    Bcov[a] = RealT(0);
                    Scon[a] = RealT(0);
                }
                for (int a = 1; a < 4; ++a) {
                    for (int b = 1; b < 4; ++b) {
                        const RealT gamma_spatial = gcon[a][b] + gcon[0][a] * gcon[0][b] / (-gcon[0][0]);
                        Scon[b] += gamma_spatial * S[a - 1];
                        Bcov[b] += gcov[a][b] * B[a - 1];
                    }
                }
                RealT bsq_lab = RealT(0);
                for (int a = 1; a < 4; ++a) bsq_lab += Bcov[a] * B[a - 1];
                const RealT rho = raw_val(D) / raw_val(LFAC);
                RealT uu = (gam - RealT(1)) / gam *
                           (raw_val(XI) / (raw_val(LFAC) * raw_val(LFAC)) - rho) / (gam - RealT(1));
                if (!(uu > RealT(0)) && DS < raw_nw && raw_val(D) != RealT(0)) {
                    uu = (raw_val(DS) / raw_val(D)) *
                         Kokkos::pow(max_val(rho, RealT(1e-300)), gam - RealT(1)) / (gam - RealT(1));
                }
                if (rho > RealT(0) && uu > RealT(0)) {
                    const RealT bpol = reverse_field ? RealT(-1) : RealT(1);
                    p[P_RHO] = rho;
                    p[P_UU] = uu;
                    for (int a = 0; a < 3; ++a) {
                        const RealT Bi = B[a];
                        p[P_U1 + a] = (Scon[a + 1] / (raw_val(XI) + bsq_lab) +
                                       Bi * BS / (raw_val(XI) * (raw_val(XI) + bsq_lab))) *
                                      raw_val(LFAC);
                    }
                    p[P_B1] = bpol * B[0];
                    p[P_B2] = bpol * B[1];
                    p[P_B3] = bpol * B[2];
                }
            }

            const size_t var_stride = cells_per_block;
            const size_t prim_block = mb * nvars * var_stride;
            for (int v = 0; v < 8; ++v) {
                const float stored = static_cast<float>(p[v]);
                p[v] = static_cast<RealT>(stored);
                device_model.prims(prim_block + static_cast<size_t>(v) * var_stride + cell) = stored;
            }

            const RealT rho = p[P_RHO];
            const RealT uu = p[P_UU];
            RealT ne = RealT(0);
            RealT thetae = RealT(0);
            RealT b_cgs = RealT(0);
            RealT sigma = RealT(1e300);
            RealT beta = RealT(1e-300);
            if (rho > RealT(0) && uu > RealT(0)) {
                const RealT r = Kokkos::exp(x1);
                const RealT th = device_model.bhac_theta_from_x2(x2);
                RealT gcov_nat[ndim][ndim];
                RealT gcon0_nat[ndim];
                device_model.bhac_native_metric(x1, x2, r, th, gcov_nat, gcon0_nat);
                const Vec4<RealT> vnat(RealT(0), p[P_U1], p[P_U2], p[P_U3]);
                RealT spatial_norm = RealT(0);
                for (int a = 1; a < ndim; ++a) {
                    for (int b = 1; b < ndim; ++b) spatial_norm += gcov_nat[a][b] * vnat[a] * vnat[b];
                }
                const RealT vfac = Kokkos::sqrt(max_val(-RealT(1) / gcon0_nat[0] *
                                                        (RealT(1) + abs_val(spatial_norm)), RealT(0)));
                Vec4<RealT> ucon_nat;
                ucon_nat[0] = -vfac * gcon0_nat[0];
                for (int a = 1; a < ndim; ++a) ucon_nat[a] = vnat[a] - vfac * gcon0_nat[a];
                const Vec4<RealT> ucov_nat = device_model.lower_with_matrix(gcov_nat, ucon_nat);
                const Vec4<RealT> Bnat(RealT(0), p[P_B1], p[P_B2], p[P_B3]);
                RealT udotB = RealT(0);
                for (int a = 1; a < ndim; ++a) udotB += ucov_nat[a] * Bnat[a];
                Vec4<RealT> bcon_nat;
                bcon_nat[0] = udotB;
                const RealT inv_u0 = RealT(1) / max_val(ucon_nat[0], RealT(1e-300));
                for (int a = 1; a < ndim; ++a) bcon_nat[a] = (Bnat[a] + ucon_nat[a] * udotB) * inv_u0;
                const Vec4<RealT> bcov_nat = device_model.lower_with_matrix(gcov_nat, bcon_nat);
                RealT bsq = RealT(0);
                for (int mu = 0; mu < ndim; ++mu) bsq += bcon_nat[mu] * bcov_nat[mu];
                bsq = max_val(abs_val(bsq), RealT(1e-40));
                sigma = max_val(bsq / max_val(rho, RealT(1e-300)), RealT(1e-300));
                beta = max_val(uu * (device_model.gam - RealT(1)) / (RealT(0.5) * bsq), RealT(1e-300));
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
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedNe) * var_stride + cell) = static_cast<float>(ne);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedThetae) * var_stride + cell) = static_cast<float>(thetae);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedB) * var_stride + cell) = static_cast<float>(b_cgs);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedSigma) * var_stride + cell) = static_cast<float>(sigma);
            device_model.derived(derived_block + static_cast<size_t>(BHACAMRRadiationModel<RealT>::DerivedBeta) * var_stride + cell) = static_cast<float>(beta);
        });
    Kokkos::fence();
    report_timing(opt.timing, "bhac_raw_convert_pack", timer.seconds());
    return model;
}

BHACStagedDump make_staged_cache_from_raw_and_model(const BHACRawStagedDump& raw,
                                                    const BHACAMRRadiationModel<RealT>& model,
                                                    const BHACLoadOptions& opt) {
    Kokkos::Timer timer;
    BHACStagedDump staged = make_staged_shell_from_raw(raw);
    const size_t cells_per_block = static_cast<size_t>(staged.nx1) *
                                   static_cast<size_t>(staged.nx2) *
                                   static_cast<size_t>(staged.nx3);
    const size_t values = static_cast<size_t>(staged.nblocks) *
                          static_cast<size_t>(staged.nvar) * cells_per_block;
    staged.prims.resize(values);
    auto prims_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), model.prims);
    for (size_t i = 0; i < values; ++i) {
        staged.prims[i] = prims_host(i);
    }
    report_timing(opt.timing, "bhac_cache_stage_prims", timer.seconds());
    return staged;
}

BHACAMRRadiationModel<DefaultReal> load_bhac_model_from_dat(const BHACLoadOptions& opt) {
    Kokkos::Timer timer;
    if (opt.cache_mode != BHACCacheOff && opt.cache_path.empty()) {
        throw std::runtime_error("BHAC cache mode requires --bhac_cache");
    }
    BHACStagedDump staged;
    bool loaded_from_cache = false;
    if (!opt.cache_path.empty() &&
        (opt.cache_mode == BHACCacheRead || opt.cache_mode == BHACCacheReadWrite)) {
        Kokkos::Timer cache_timer;
        try {
            staged = read_bhac_staged_cache(opt.cache_path, opt);
            loaded_from_cache = true;
            report_timing(opt.timing, "bhac_cache_read", cache_timer.seconds());
        } catch (const std::exception& e) {
            if (opt.cache_mode == BHACCacheRead) {
                throw;
            }
            if (opt.timing) {
                std::cout << "timing bhac_cache_miss " << cache_timer.seconds() << " s\n";
                std::cout << "bhac_cache_miss_reason " << e.what() << "\n";
            }
        }
    }
    BHACAMRRadiationModel<DefaultReal> model;
    if (loaded_from_cache) {
        model = materialize_bhac_model_from_staged(staged, opt);
    } else {
        BHACRawStagedDump raw = read_bhac_raw_staged_dump(opt);
        model = materialize_bhac_model_from_raw(raw, opt);
        if (!opt.cache_path.empty() &&
            (opt.cache_mode == BHACCacheWrite || opt.cache_mode == BHACCacheReadWrite)) {
            Kokkos::Timer cache_timer;
            staged = make_staged_cache_from_raw_and_model(raw, model, opt);
            write_bhac_staged_cache(opt.cache_path, staged, opt);
            report_timing(opt.timing, "bhac_cache_write", cache_timer.seconds());
        }
    }
    report_timing(opt.timing, "bhac_load_total", timer.seconds());
    return model;
}

} // namespace kpolaris
