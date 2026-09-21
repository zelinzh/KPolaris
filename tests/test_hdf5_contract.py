#!/usr/bin/env python3
"""Small HDF5 contract tests for image, plotting, and optional trace tools."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

try:
    import h5py
    import numpy as np
except ImportError:
    print("h5py/numpy is not available; skipping HDF5 contract test")
    raise SystemExit(77)


def text_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def run(cmd: list[str], cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if expect_ok and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(cmd)}")
    if not expect_ok and result.returncode == 0:
        print(result.stdout)
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(cmd)}")
    return result


def assert_failed_with(
    cmd: list[str], cwd: Path, expected: str
) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd, expect_ok=False)
    combined = result.stdout + result.stderr
    if expected not in combined:
        raise AssertionError(
            f"failed command did not report {expected!r}: {' '.join(cmd)}\n{combined}"
        )
    return result


def replace_option(args: list[str], option: str, value: Path) -> list[str]:
    """Return command arguments with one path-valued option replaced."""
    prefix = f"--{option}="
    replaced = [
        f"{prefix}{value.resolve()}" if argument.startswith(prefix) else argument
        for argument in args
    ]
    if replaced == args:
        raise AssertionError(f"missing command option {prefix}")
    return replaced


def assert_close(actual: float, expected: float, rel: float = 5e-6, abs_tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(f"{actual!r} != {expected!r}")


def relative_l1(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.sum(np.abs(left))), 1.0e-300)
    return float(np.sum(np.abs(left - right))) / denominator


def compiled_grmhd_coordinates(value: str) -> set[str]:
    aliases = {
        "cartesian_ks": "cartesian_ks",
        "cartesian": "cartesian_ks",
        "ks_cartesian": "cartesian_ks",
        "spherical_ks": "spherical_ks",
        "ks": "spherical_ks",
        "ks_spherical": "spherical_ks",
        "sks": "spherical_ks",
        "boyer_lindquist": "boyer_lindquist",
        "bl": "boyer_lindquist",
        "fmks": "fmks",
        "mks": "mks",
        "modified_ks": "mks",
        "modified-kerr-schild": "mks",
        "native_mks": "mks",
    }
    requested = [item.strip().lower() for item in value.replace(",", ";").split(";")]
    if requested == ["all"]:
        return {"cartesian_ks", "spherical_ks", "boyer_lindquist", "fmks", "mks"}
    try:
        return {aliases[item] for item in requested if item}
    except KeyError as error:
        raise AssertionError(f"unknown compiled GRMHD coordinate {error.args[0]!r}") from error


def image_args(output: Path, params: Path | None = None) -> list[str]:
    # Freeze the legacy sampling position and W output basis for numerical goldens.
    # A separate check below verifies the new pixel-center default.
    args = [
        "--model=riaf",
        "--evpa_0=W",
        "--format=hdf5",
        f"--output={output}",
        "--nx=8",
        "--ny=8",
        "--radius=1000",
        "--coordinate=boyer_lindquist",
        "--camera=pinhole",
        "--use_pinhole_pixel_bias=1",
        "--dsource=8100",
        "--fovx_dsource=200",
        "--fovy_dsource=200",
        "--freq=230000000000",
        "--spin=0.11272767423795238",
        "--inclination_deg=9.253534048829724",
        "--riaf_ne_unit=37228296.05086979",
        "--riaf_te_unit=914444770722.201",
        "--riaf_disk_h=0.1690723996627777",
        "--riaf_mbh_solar=3117093.5764001394",
        "--riaf_pow_nth=-1.33442753722822",
        "--riaf_pow_T=-0.9859963040846038",
        "--riaf_keplerian_factor=0.8376738186088084",
        "--riaf_infall_factor=0.1623261813911916",
        "--adaptive_tolerance=1e-8",
        "--timing=0",
    ]
    if params is None:
        args.append("--parameter_output=none")
    else:
        args.append(f"--parameter_output={params}")
    return args


def torus_image_args(output: Path) -> list[str]:
    return [
        "--model=torus",
        "--evpa_0=W",
        "--format=hdf5",
        f"--output={output}",
        "--parameter_output=none",
        "--nx=8",
        "--ny=8",
        "--camera=parallel_plane",
        "--coordinate=cartesian_ks",
        "--radius=1000",
        "--inclination_deg=50",
        "--xspan=20",
        "--dsource=8100",
        "--spin=0.9",
        "--step=0.025",
        "--adaptive=1",
        "--adaptive_tolerance=1e-8",
        "--min_step=1e-10",
        "--max_step=5",
        "--max_steps=800000",
        "--max_radiation_step=1",
        "--max_radiation_depth=1",
        "--max_absorption_depth=1",
        "--max_faraday_depth=4",
        "--freq=230000000000",
        "--torus_l_lambda=0.78",
        "--torus_wwin=1",
        "--torus_kappa=1.3333333333333333",
        "--torus_omegac=1",
        "--torus_betac=10",
        "--torus_beta=10",
        "--torus_Rhigh=1",
        "--torus_bh_mass_solar=4000000",
        "--torus_mdot_cgs=1570000000000000",
        "--torus_mdot_code=0.003",
        "--torus_thetae_min=0.05",
        "--scalar_transport=0",
        "--timing=0",
    ]


def write_synthetic_iharm_fixture(path: Path) -> None:
    """Write a small deterministic, redistributable iHARM-format fixture."""
    n1 = n2 = n3 = 8
    r_in = 2.0
    r_out = 40.0
    with h5py.File(path, "w") as h5:
        header = h5.create_group("header")
        for name, value in (("n1", n1), ("n2", n2), ("n3", n3)):
            header.create_dataset(name, data=np.int32(value))
        for name, value in (("a", 0.5), ("gam", 4.0 / 3.0), ("t", 100.0)):
            header.create_dataset(name, data=np.float64(value))

        geom = header.create_group("geom")
        for name, value in (
            ("startx1", math.log(r_in)),
            ("startx2", 0.0),
            ("startx3", 0.0),
            ("dx1", (math.log(r_out) - math.log(r_in)) / n1),
            ("dx2", 1.0 / n2),
            ("dx3", 2.0 * math.pi / n3),
        ):
            geom.create_dataset(name, data=np.float64(value))
        fmks = geom.create_group("fmks")
        for name, value in (
            ("hslope", 0.3),
            ("mks_smooth", 0.5),
            ("poly_alpha", 14.0),
            ("poly_xt", 0.82),
            ("r_in", r_in),
            ("r_out", r_out),
        ):
            fmks.create_dataset(name, data=np.float64(value))

        prims = np.zeros((n1, n2, n3, 8), dtype=np.float32)
        radial_density = np.exp(-0.08 * np.arange(n1, dtype=np.float32))
        prims[..., 0] = radial_density[:, None, None]
        prims[..., 1] = 0.12 * radial_density[:, None, None]
        prims[..., 5] = 0.03
        prims[..., 7] = 0.08
        h5.create_dataset("prims", data=prims)
        h5.create_dataset("t", data=np.float64(100.0))


def write_synthetic_kharma_fixture(
    path: Path, meshblocks_x1: int = 1
) -> None:
    """Write the same analytic grid in KHARMA's meshblock PHDF layout."""
    n = 8
    if meshblocks_x1 < 1 or n % meshblocks_x1:
        raise ValueError("synthetic KHARMA x1 meshblock count must divide 8")
    nx1_mb = n // meshblocks_x1
    r_in = 2.0
    r_out = 40.0
    parameters = (
        "<parthenon/mesh>\n"
        f"nx1 = {n}\n"
        f"nx2 = {n}\n"
        f"nx3 = {n}\n"
        f"x1min = {math.log(r_in):.17g}\n"
        f"x1max = {math.log(r_out):.17g}\n"
        "x2min = 0\n"
        "x2max = 1\n"
        "x3min = 0\n"
        f"x3max = {2.0 * math.pi:.17g}\n"
        "<coordinates>\n"
        "a = 0.5\n"
        "transform = fmks\n"
        "r_in = 2\n"
        "r_out = 40\n"
        "hslope = 0.3\n"
        "mks_smooth = 0.5\n"
        "poly_alpha = 14\n"
        "poly_xt = 0.82\n"
        "<GRMHD>\n"
        f"gamma = {4.0 / 3.0:.17g}\n"
    )
    with h5py.File(path, "w") as h5:
        info = h5.create_group("Info")
        info.attrs["NumMeshBlocks"] = np.int64(meshblocks_x1)
        info.attrs["MeshBlockSize"] = np.array(
            [nx1_mb, n, n], dtype=np.int64
        )
        info.attrs["Time"] = np.float64(100.0)
        input_group = h5.create_group("Input")
        input_group.attrs["File"] = parameters
        blocks = h5.create_group("Blocks")
        blocks.create_dataset(
            "loc.lx123",
            data=np.array(
                [[block, 0, 0] for block in range(meshblocks_x1)],
                dtype=np.int64,
            ),
        )

        shape = (meshblocks_x1, n, n, nx1_mb)
        rho = np.empty(shape, dtype=np.float32)
        radial_density = np.exp(-0.08 * np.arange(n, dtype=np.float32))
        for block in range(meshblocks_x1):
            begin = block * nx1_mb
            rho[block, ...] = radial_density[
                begin : begin + nx1_mb
            ][None, None, :]
        h5.create_dataset("prims.rho", data=rho)
        h5.create_dataset("prims.u", data=0.12 * rho)
        uvec = np.zeros(
            (meshblocks_x1, 3, n, n, nx1_mb), dtype=np.float32
        )
        h5.create_dataset("prims.uvec", data=uvec)
        bvec = np.zeros_like(uvec)
        bvec[:, 0, ...] = 0.03
        bvec[:, 2, ...] = 0.08
        h5.create_dataset("prims.B", data=bvec)


def write_synthetic_athenak_fixture(
    path: Path, location_size: int = 8, variable_size: int = 4
) -> None:
    """Write a deterministic single-meshblock Athena binary v1.1 fixture."""
    if location_size not in (4, 8) or variable_size not in (4, 8):
        raise ValueError("AthenaK fixture sizes must be 4 or 8 bytes")
    n = 8
    nvar = 8
    parameter_block = (
        "<meshblock>\n"
        f"nx1 = {n}\n"
        f"nx2 = {n}\n"
        f"nx3 = {n}\n"
        "<mesh>\n"
        "x1min = -40\n"
        "x1max = 40\n"
        "x2min = -40\n"
        "x2max = 40\n"
        "x3min = -40\n"
        "x3max = 40\n"
        "<coord>\n"
        "a = 0.5\n"
        "<mhd>\n"
        f"gamma = {4.0 / 3.0:.17g}\n"
    ).encode("ascii")
    preheader = (
        "Athena binary output version=1.1\n"
        "pheader size=5\n"
        "time=100\n"
        "cycle=0\n"
        f"size of location={location_size}\n"
        f"size of variable={variable_size}\n"
        "nvars=8\n"
        "variables: rho u1 u2 u3 uu b1 b2 b3\n"
        f"header offset={len(parameter_block)}\n"
    ).encode("ascii")

    # AthenaK binary order is variable, k, j, i.  The constant Cartesian
    # primitive field makes the format/index contract deterministic while the
    # Kerr metric still produces spatially varying derived beta and sigma.
    prims = np.zeros(
        (nvar, n, n, n), dtype="<f4" if variable_size == 4 else "<f8"
    )
    prims[0, ...] = 1.0
    prims[4, ...] = 0.12
    prims[5, ...] = 0.03
    prims[7, ...] = 0.08
    with path.open("wb") as stream:
        stream.write(preheader)
        stream.write(parameter_block)
        stream.write(struct.pack("<10i", 0, n - 1, 0, n - 1, 0, n - 1, 0, 0, 0, 0))
        location_format = "<6f" if location_size == 4 else "<6d"
        stream.write(
            struct.pack(
                location_format, -40.0, 40.0, -40.0, 40.0, -40.0, 40.0
            )
        )
        stream.write(prims.tobytes(order="C"))


def write_synthetic_bhac_fixture(path: Path) -> None:
    """Write a deterministic single-leaf BHAC AMR dump."""
    n = 8
    cells = n * n * n
    nvar = 13
    gamma = 4.0 / 3.0

    # BHAC stores conserved variables as variable-major float64 blocks.  With
    # Lorentz factor one and zero momentum, XI=rho+gamma*u recovers rho=1 and
    # u=0.12 in the loader's conserved-to-primitive conversion.
    conserved = np.zeros((nvar, cells), dtype="<f8")
    conserved[0, ...] = 1.0
    conserved[5, ...] = 0.03
    conserved[7, ...] = 0.08
    conserved[nvar - 2, ...] = 1.0
    conserved[nvar - 1, ...] = 1.0 + gamma * 0.12

    with path.open("wb") as stream:
        stream.write(conserved.tobytes(order="C"))
        stream.write(struct.pack("<i", 1))  # one root node, which is a leaf
        stream.write(struct.pack("<3i", n, n, n))
        stream.write(struct.pack("<4d", gamma, 2.0, 2.0, 0.5))
        stream.write(
            struct.pack(
                "<8id",
                1,  # nleafs
                1,  # levmax
                3,  # ndim
                3,  # ndir
                nvar,
                0,  # no staggered variables
                4,  # neqpar
                0,  # iteration
                100.0,
            )
        )


def write_synthetic_hamr_fixture(directory: Path) -> None:
    """Write a deterministic one-block H-AMR split-binary fixture."""
    n = 8
    nvar = 9
    gamma = 4.0 / 3.0
    x1_min = math.log(2.0)
    x1_max = math.log(40.0)

    directory.mkdir(parents=True, exist_ok=True)
    header = bytearray(0x114)
    struct.pack_into("<d", header, 0x000, 100.0)
    struct.pack_into("<i", header, 0x008, 1)  # blocks per new_dump file
    struct.pack_into("<i", header, 0x00C, 1)  # total active blocks
    struct.pack_into("<3i", header, 0x040, n, n, n)
    struct.pack_into("<i", header, 0x04C, 72)  # eight roots, nine slots each
    # Two roots per dimension make the default Morton decoder well-defined;
    # block id zero occupies the first root and spans the physical domain used
    # by this fixture.
    struct.pack_into("<3i", header, 0x050, 2, 2, 2)
    struct.pack_into("<3d", header, 0x05C, x1_min, -1.0, 0.0)
    struct.pack_into(
        "<3d",
        header,
        0x074,
        (x1_max - x1_min) / n,
        2.0 / n,
        2.0 * math.pi / n,
    )
    struct.pack_into("<d", header, 0x094, 0.5)
    struct.pack_into("<d", header, 0x09C, gamma)
    struct.pack_into("<2d", header, 0x0AC, 2.0, 40.0)
    struct.pack_into("<d", header, 0x0C4, 1.0)
    struct.pack_into("<3i", header, 0x108, 0, 1, 0)
    (directory / "parameters").write_bytes(header)

    # H-AMR raw data are cell-major float32 values.  The fields used by the
    # loader are rho, uu, unused, u1..u3, and B1..B3.
    raw = np.zeros((n * n * n, nvar), dtype="<f4")
    raw[:, 0] = 1.0
    raw[:, 1] = 0.12
    raw[:, 6] = 0.03
    raw[:, 8] = 0.08
    (directory / "new_dump0000").write_bytes(raw.tobytes(order="C"))


def synthetic_iharm_transport_args(dump: Path) -> list[str]:
    return [
        "--model=iharm",
        f"--iharm_dump={dump.resolve()}",
        "--coordinate=fmks",
        "--camera=pinhole",
        "--use_pinhole_pixel_bias=1",  # retain historical fixture sampling
        "--nx=2",
        "--ny=2",
        "--radius=100",
        "--inclination_deg=17",
        "--dsource=16900000",
        "--fovx_dsource=160",
        "--fovy_dsource=160",
        "--inner_radius=-1",
        "--outer_radius=38",
        "--step=0.025",
        "--adaptive=1",
        "--adaptive_tolerance=1e-8",
        "--min_step=1e-10",
        "--max_step=1",
        "--max_steps=800000",
        "--max_radiation_step=1",
        "--max_radiation_depth=1",
        "--max_absorption_depth=1",
        "--max_faraday_depth=4",
        "--freq=230000000000",
        "--iharm_M_unit=3e25",
        "--iharm_mbh_solar=6.2e9",
        "--iharm_trat_small=1",
        "--iharm_trat_large=20",
        "--iharm_beta_crit=1",
        "--iharm_sigma_cut=100",
        "--iharm_interpolate_derived_scalars=1",
        "--timing=0",
    ]


def synthetic_kharma_transport_args(dump: Path) -> list[str]:
    return [
        argument.replace("--model=iharm", "--model=kharma")
        .replace("--iharm_dump=", "--kharma_dump=")
        .replace("--iharm_", "--kharma_")
        for argument in synthetic_iharm_transport_args(dump)
    ]


def synthetic_athenak_transport_args(dump: Path) -> list[str]:
    return [
        "--model=athenak",
        f"--athenak_dump={dump.resolve()}",
        "--coordinate=cartesian_ks",
        "--camera=parallel_plane",
        "--nx=2",
        "--ny=2",
        "--radius=100",
        "--inclination_deg=17",
        "--xspan=20",
        "--dsource=16900000",
        "--inner_radius=-1",
        "--outer_radius=38",
        "--step=0.025",
        "--adaptive=1",
        "--adaptive_tolerance=1e-8",
        "--min_step=1e-10",
        "--max_step=1",
        "--max_steps=800000",
        "--max_radiation_step=1",
        "--max_radiation_depth=1",
        "--max_absorption_depth=1",
        "--max_faraday_depth=4",
        "--freq=230000000000",
        "--athenak_M_unit=3e25",
        "--athenak_mbh_solar=6.2e9",
        "--athenak_trat_small=1",
        "--athenak_trat_large=20",
        "--athenak_beta_crit=1",
        "--athenak_sigma_cut=100",
        "--timing=0",
    ]


def synthetic_bhac_transport_args(dump: Path) -> list[str]:
    return [
        "--model=bhac",
        f"--bhac_dump={dump.resolve()}",
        "--coordinate=spherical_ks",
        "--camera=pinhole",
        "--use_pinhole_pixel_bias=1",  # retain historical fixture sampling
        "--nx=2",
        "--ny=2",
        "--radius=100",
        "--inclination_deg=17",
        "--dsource=16900000",
        "--fovx_dsource=160",
        "--fovy_dsource=160",
        "--inner_radius=-1",
        "--outer_radius=38",
        "--step=0.025",
        "--adaptive=1",
        "--adaptive_tolerance=1e-8",
        "--min_step=1e-10",
        "--max_step=1",
        "--max_steps=800000",
        "--max_radiation_step=1",
        "--max_radiation_depth=1",
        "--max_absorption_depth=1",
        "--max_faraday_depth=4",
        "--freq=230000000000",
        "--bhac_M_unit=3e25",
        "--bhac_mbh_solar=6.2e9",
        "--bhac_trat_small=1",
        "--bhac_trat_large=20",
        "--bhac_beta_crit=1",
        f"--bhac_gamma={4.0 / 3.0:.17g}",
        "--bhac_sigma_cut=100",
        "--bhac_r_in=2",
        "--bhac_r_out=40",
        "--bhac_hslope=0.3",
        f"--bhac_x1_min={math.log(2.0):.17g}",
        f"--bhac_x1_max={math.log(40.0):.17g}",
        "--bhac_x2_min=0",
        f"--bhac_x2_max={math.pi:.17g}",
        "--bhac_x3_min=0",
        f"--bhac_x3_max={2.0 * math.pi:.17g}",
        "--bhac_sfc=0",
        "--timing=0",
    ]


def synthetic_hamr_transport_args(directory: Path) -> list[str]:
    return [
        "--model=hamr",
        f"--hamr_dump={directory.resolve()}",
        "--coordinate=spherical_ks",
        "--camera=pinhole",
        "--use_pinhole_pixel_bias=1",  # retain historical fixture sampling
        "--nx=2",
        "--ny=2",
        "--radius=100",
        "--inclination_deg=17",
        "--dsource=16900000",
        "--fovx_dsource=160",
        "--fovy_dsource=160",
        "--inner_radius=-1",
        "--outer_radius=38",
        "--step=0.025",
        "--adaptive=1",
        "--adaptive_tolerance=1e-8",
        "--min_step=1e-10",
        "--max_step=1",
        "--max_steps=800000",
        "--max_radiation_step=1",
        "--max_radiation_depth=1",
        "--max_absorption_depth=1",
        "--max_faraday_depth=4",
        "--freq=230000000000",
        "--hamr_M_unit=3e25",
        "--hamr_mbh_solar=6.2e9",
        "--hamr_trat_small=1",
        "--hamr_trat_large=20",
        "--hamr_beta_crit=1",
        "--hamr_sigma_cut=100",
        "--hamr_hslope=0.3",
        "--hamr_id_order=root_slot",
        "--hamr_root_order=morton",
        "--timing=0",
    ]


def check_image_contract(path: Path, expect_golden: bool = False) -> None:
    with h5py.File(path, "r") as h5:
        assert text_attr(h5.attrs["schema"]) == "kpolaris_image"
        assert int(h5.attrs["schema_version"]) == 1
        assert text_attr(h5.attrs["code"]) == "KPolaris"
        assert text_attr(h5.attrs["code_version"])
        assert text_attr(h5.attrs["code_revision"])
        assert int(h5.attrs["code_source_dirty"]) in {-1, 0, 1}
        fingerprint = text_attr(h5.attrs["code_source_fingerprint"])
        assert fingerprint == "unknown" or (
            len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
        )
        assert text_attr(h5.attrs["compiler_id"])
        assert text_attr(h5.attrs["compiler_version"])
        assert text_attr(h5.attrs["build_type"])
        header = h5["header"].attrs
        for name in (
            "code",
            "code_version",
            "code_revision",
            "code_source_dirty",
            "compiler_id",
            "compiler_version",
            "build_type",
        ):
            assert h5.attrs[name] == header[name]
        assert int(h5.attrs["nfreq"]) == 1
        assert h5["grid/x"].shape == (int(h5.attrs["nx"]),)
        assert h5["grid/y"].shape == (int(h5.attrs["ny"]),)
        assert h5["grid/frequency_hz"].shape == (1,)
        for name in ("grid/x", "grid/y", "grid/frequency_hz"):
            assert h5[name].dtype == np.dtype("float64")
        assert "intensity_to_flux_jy_per_pixel" in h5.attrs
        scale = float(h5.attrs["intensity_to_flux_jy_per_pixel"])
        results = h5["results"].attrs
        flux = float(results["flux_I_Jy"])
        freq = h5["frame_0/freq_0"]
        image = freq["I"][...]
        for name in ("I", "Q", "U", "V", "I_inv", "Q_inv", "U_inv", "V_inv"):
            assert freq[name].shape == (int(h5.attrs["ny"]), int(h5.attrs["nx"]))
            assert freq[name].dtype == np.dtype("float64")
        diagnostics = h5["frame_0/diagnostics"]
        integer_diagnostics = {"reason", "pass_a_steps", "steps", "total_steps"}
        for name in (
            "reason", "pass_a_steps", "steps", "total_steps", "closure_x",
            "closure_k", "final_null", "frame_error", "det_r", "overlap_r11",
            "overlap_r12", "overlap_r21", "overlap_r22", "basis_identity_error",
            "basis_rotation_angle",
        ):
            assert diagnostics[name].shape == image.shape
            expected_dtype = np.dtype("int32") if name in integer_diagnostics else np.dtype("float64")
            assert diagnostics[name].dtype == expected_dtype
        assert math.isfinite(scale) and scale > 0.0
        assert math.isfinite(flux)
        assert np.isfinite(image).all()
        assert abs(float(image.sum()) * scale - flux) <= max(1e-10, abs(flux) * 1e-10)
        assert "sum_I_nu_cgs" not in h5["results"].attrs
        assert "sum_I_nu_cgs_by_freq" not in h5["results"]
        assert "sum_I_nu_cgs" not in h5["frame_0/freq_0"].attrs
        assert "flux_I_Jy_by_freq" in h5["results"]
        if expect_golden:
            assert image.shape == (8, 8)
            # Fixed-parameter regression for the BL pinhole-camera KS->BL
            # Jacobian and composed adaptive domain endpoints.
            assert_close(flux, 2.3672187308355426, rel=2e-6, abs_tol=1e-10)
            assert_close(float(results["flux_Q_Jy"]), -0.009037507606537341, rel=2e-6, abs_tol=1e-11)
            assert_close(float(results["flux_U_Jy"]), 0.002284126131173937, rel=2e-6, abs_tol=1e-11)
            assert_close(float(results["flux_V_Jy"]), -0.02250919032159511, rel=2e-6, abs_tol=1e-11)
            assert_close(float(image.sum()), 0.0016114226118512818, rel=2e-6, abs_tol=1e-14)
            assert_close(float(image.max()), 0.0002874691367846926, rel=2e-6, abs_tol=1e-14)
            assert_close(float(image[4, 4]), 0.0001724585679855508, rel=2e-6, abs_tol=1e-14)


def check_invalid_pixel_contract(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        reason = h5["frame_0/diagnostics/reason"][...]
        invalid = reason != 1
        assert invalid.any()
        for name in ("I", "Q", "U", "V", "I_inv", "Q_inv", "U_inv", "V_inv"):
            values = h5[f"frame_0/freq_0/{name}"][...]
            assert np.count_nonzero(values[invalid]) == 0


def check_torus_golden(path: Path) -> None:
    check_image_contract(path)
    with h5py.File(path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "torus"
        assert text_attr(h5.attrs["coordinate"]) == "cartesian_ks"
        assert int(h5["results"].attrs["returned"]) == 64
        results = h5["results"].attrs
        image = h5["frame_0/freq_0/I"][...]
        assert_close(float(results["flux_I_Jy"]), 6.045089718945156, rel=2e-6, abs_tol=1e-10)
        assert_close(float(results["flux_Q_Jy"]), -0.12238392198279589, rel=2e-6, abs_tol=1e-11)
        assert_close(float(results["flux_U_Jy"]), 0.3783525605607396, rel=2e-6, abs_tol=1e-11)
        assert_close(float(results["flux_V_Jy"]), -0.1562912859650497, rel=2e-6, abs_tol=1e-11)
        assert_close(float(image.sum()), 0.004327417706395372, rel=2e-6, abs_tol=1e-12)
        assert_close(float(image.max()), 0.0013972657697387486, rel=2e-6, abs_tol=1e-12)


def compare_single_frequency_images(left: Path, right: Path) -> None:
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        for stokes in ("I", "Q", "U", "V"):
            av = a[f"frame_0/freq_0/{stokes}"][...]
            bv = b[f"frame_0/freq_0/{stokes}"][...]
            assert np.allclose(av, bv, rtol=1e-12, atol=1e-30)
            key = f"flux_{stokes}_Jy"
            assert_close(float(a["results"].attrs[key]), float(b["results"].attrs[key]), rel=1e-12, abs_tol=1e-20)


def check_multifrequency_image_contract(
    path: Path,
    expected_shape: tuple[int, int] = (4, 4),
    expected_frequencies: tuple[float, float] = (100000000000.0, 230000000000.0),
) -> None:
    with h5py.File(path, "r") as h5:
        assert text_attr(h5.attrs["schema"]) == "kpolaris_image"
        assert int(h5.attrs["nfreq"]) == 2
        assert list(h5["grid/frequency_hz"][...]) == list(expected_frequencies)
        fluxes = h5["results/flux_I_Jy_by_freq"][...]
        assert fluxes.shape == (2,)
        assert np.isfinite(fluxes).all()
        for i, expected_freq in enumerate(expected_frequencies):
            group = h5[f"frame_0/freq_{i}"]
            assert float(group.attrs["frequency_hz"]) == expected_freq
            image = group["I"][...]
            assert image.shape == expected_shape
            assert np.isfinite(image).all()
            scale = float(group.attrs["intensity_to_flux_jy_per_pixel"])
            assert_close(float(image.sum()) * scale, float(fluxes[i]), rel=1e-10, abs_tol=1e-12)


def check_analysis_contract(
    path: Path,
    analysis_path: str = "frame_0/analysis",
    image_path: str = "frame_0/freq_0",
) -> None:
    required = [
        "radiating_path_length",
        "emission_weight",
        "emission_weighted_radius",
        "emission_weighted_optical_depth_to_camera",
        "absorption_depth",
        "absorption_operator_depth",
        "faraday_rotation_depth",
        "faraday_conversion_depth",
        "faraday_operator_depth",
        "dominant_emission_radius",
        "dominant_emission_region",
        "dominant_ne_cgs",
        "dominant_thetae",
        "dominant_b_cgs",
        "dominant_beta",
        "dominant_sigma",
        "emission_weighted_ne_cgs",
        "emission_weighted_thetae",
        "emission_weighted_b_cgs",
        "emission_weighted_beta",
        "emission_weighted_sigma",
        "photon_ring_winding_estimate",
        "radiation_substeps",
    ]
    with h5py.File(path, "r") as h5:
        assert analysis_path in h5
        analysis = h5[analysis_path]
        image = h5[image_path]
        expected_shape = image["I"].shape
        assert text_attr(analysis.attrs["schema"]) == "kpolaris_physical_analysis"
        assert int(analysis.attrs["schema_version"]) == 2
        for name in required:
            assert name in analysis
            data = analysis[name][...]
            assert data.shape == expected_shape
            assert np.isfinite(data).all()
            expected_dtype = np.dtype("int32") if name in {
                "dominant_emission_region", "radiation_substeps"
            } else np.dtype("float64")
            assert analysis[name].dtype == expected_dtype

        assert int(analysis.attrs["observer_weighted_available"]) == 1
        radial_bins = int(analysis.attrs["radial_bins"])
        assert 1 <= radial_bins <= 24
        assert 0.0 < float(analysis.attrs["formation_fraction"]) < 1.0
        edges = analysis["radial_bin_edges"][...]
        centers = analysis["radial_bin_centers"][...]
        assert edges.shape == (radial_bins + 1,)
        assert centers.shape == (radial_bins,)
        assert np.isfinite(edges).all() and np.isfinite(centers).all()
        assert np.all(np.diff(edges) > 0.0)
        assert np.all((centers > edges[:-1]) & (centers < edges[1:]))

        radial_names = (
            "radial_stokes_I_inv_contribution",
            "radial_stokes_Q_inv_contribution",
            "radial_stokes_U_inv_contribution",
            "radial_stokes_V_inv_contribution",
            "radial_absorption_depth",
            "radial_faraday_rotation_depth",
            "radial_faraday_conversion_depth",
            "radial_faraday_operator_depth",
        )
        for name in radial_names:
            assert name in analysis
            assert analysis[name].shape == (radial_bins, *expected_shape)
            assert analysis[name].dtype == np.dtype("float64")
            assert np.isfinite(analysis[name][...]).all()

        derived_names = (
            "observer_weighted_radius_I",
            "observer_weighted_radius_linear",
            "observer_weighted_radius_circular",
            "intensity_formation_radius_low",
            "intensity_formation_radius_median",
            "intensity_formation_radius_high",
            "linear_formation_radius_low",
            "linear_formation_radius_median",
            "linear_formation_radius_high",
            "circular_formation_radius_low",
            "circular_formation_radius_median",
            "circular_formation_radius_high",
            "los_linear_coherence",
            "los_circular_coherence",
            "contribution_closure_max_abs",
            "contribution_closure_relative_l1",
            "foreground_absorption_depth",
            "foreground_faraday_rotation_depth",
            "foreground_faraday_conversion_depth",
            "foreground_faraday_operator_depth",
            "foreground_faraday_operator_fraction",
        )
        for name in derived_names:
            assert analysis[name].shape == expected_shape
            assert analysis[name].dtype == np.dtype("float64")
            assert np.isfinite(analysis[name][...]).all()

        for stokes in "IQUV":
            radial_sum = analysis[
                f"radial_stokes_{stokes}_inv_contribution"
            ][...].sum(axis=0)
            assert np.allclose(
                radial_sum, image[f"{stokes}_inv"][...],
                rtol=5.0e-12, atol=1.0e-300,
            )
        depth_pairs = (
            ("radial_absorption_depth", "absorption_depth"),
            ("radial_faraday_rotation_depth", "faraday_rotation_depth"),
            ("radial_faraday_conversion_depth", "faraday_conversion_depth"),
            ("radial_faraday_operator_depth", "faraday_operator_depth"),
        )
        for radial_name, total_name in depth_pairs:
            assert np.allclose(
                analysis[radial_name][...].sum(axis=0),
                analysis[total_name][...], rtol=5.0e-12, atol=1.0e-12,
            )
        for prefix in ("intensity", "linear", "circular"):
            low = analysis[f"{prefix}_formation_radius_low"][...]
            median = analysis[f"{prefix}_formation_radius_median"][...]
            high = analysis[f"{prefix}_formation_radius_high"][...]
            assert np.all(low <= median)
            assert np.all(median <= high)
        for name in ("los_linear_coherence", "los_circular_coherence"):
            values = analysis[name][...]
            assert np.all(values >= 0.0)
            assert np.all(values <= 1.0 + 5.0e-12)
        assert np.nanmax(analysis["contribution_closure_relative_l1"][...]) < 1.0e-10
        fractions = analysis["foreground_faraday_operator_fraction"][...]
        assert np.all(fractions >= 0.0)
        assert np.all(fractions <= 1.0 + 5.0e-12)


def check_loader_csv_metadata(
    image_exe: str,
    transport_args: list[str],
    workdir: Path,
    model: str,
    expected_header: str,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    csv_path = workdir / f"synthetic_{model}_metadata.csv"
    run(
        [
            image_exe,
            *transport_args,
            "--format=csv",
            f"--output={csv_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    text = csv_path.read_text(encoding="utf-8")
    assert expected_header in text
    for prefix in forbidden_prefixes:
        assert f"# {prefix}_" not in text


def check_spherical_ks_resample_contract(
    image_exe: str,
    transport_args: list[str],
    workdir: Path,
    model: str,
) -> None:
    """Exercise both public native-to-Spherical-KS loader fast paths."""
    modes: tuple[tuple[str, tuple[str, ...], float], ...] = (
        (
            "spherical_ks_primitives",
            (),
            269.2754525609998,
        ),
        (
            "spherical_ks_precomputed",
            (
                f"--{model}_resample_n1=8",
                f"--{model}_resample_n2=8",
                f"--{model}_resample_n3=8",
                f"--{model}_resample_r_in=2.5",
                f"--{model}_resample_r_out=35",
            ),
            264.6981298740525,
        ),
    )
    fluxes: dict[str, float] = {}
    for mode, grid_args, expected_flux in modes:
        output = workdir / f"synthetic_{model}_resample_{mode}.h5"
        run(
            [
                image_exe,
                *transport_args,
                f"--{model}_resample={mode}",
                *grid_args,
                "--format=hdf5",
                "--analysis_mode=1",
                f"--output={output}",
                "--parameter_output=none",
            ],
            workdir,
        )
        check_image_contract(output)
        check_analysis_contract(output)
        with h5py.File(output, "r") as h5:
            assert text_attr(h5.attrs["coordinate"]) == "spherical_ks"
            assert int(h5["results"].attrs["returned"]) == 4
            assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
            for stokes in "IQUV":
                assert np.isfinite(h5[f"frame_0/freq_0/{stokes}"][...]).all()
            flux = float(h5["results"].attrs["flux_I_Jy"])
            assert_close(flux, expected_flux, rel=2e-6, abs_tol=1e-9)
            fluxes[mode] = flux

    # The two public paths intentionally interpolate different quantities, but
    # should remain close for this smooth analytic fixture.
    assert abs(
        fluxes["spherical_ks_primitives"]
        - fluxes["spherical_ks_precomputed"]
    ) / fluxes["spherical_ks_primitives"] < 0.02

    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--{model}_resample=spherical_ks_primitives",
            f"--{model}_resample_n1=1",
            "--format=hdf5",
            f"--output={workdir / f'synthetic_{model}_invalid_primitive_resample.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        f"invalid Spherical KS primitive resample grid for {model}",
    )
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--{model}_resample=spherical_ks_precomputed",
            f"--{model}_resample_r_in=40",
            f"--{model}_resample_r_out=2",
            "--format=hdf5",
            f"--output={workdir / f'synthetic_{model}_invalid_precomputed_resample.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        f"invalid Spherical KS resample grid for {model}",
    )
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--{model}_resample_spherical_ks_primitives=1",
            f"--{model}_resample_spherical_ks_precomputed=1",
            "--format=hdf5",
            f"--output={workdir / f'synthetic_{model}_ambiguous_resample.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        f"choose only one {model} Spherical KS resample mode",
    )

    # Model-specific image binaries compile out the on-the-fly scalar branch.
    # Reject an incompatible runtime toggle before any empty derived view can
    # reach a transport kernel.
    display_name = "iHARM" if model == "iharm" else "KHARMA"
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--{model}_interpolate_derived_scalars=0",
            "--format=hdf5",
            f"--output={workdir / f'synthetic_{model}_derived_scalars_disabled.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        f"this {display_name} build requires --{model}_interpolate_derived_scalars=1",
    )


def check_loader_dump_time_probe(
    image_exe: str,
    transport_args: list[str],
    fixture: Path,
    workdir: Path,
    model: str,
) -> None:
    """Read both supported dump-time spellings through the public probe path."""
    second_fixture = workdir / f"synthetic_{model}_time_200.h5"
    shutil.copyfile(fixture, second_fixture)
    with h5py.File(second_fixture, "r+") as h5:
        if model == "iharm":
            del h5["t"]
            h5["header/t"][...] = 200.0
        else:
            del h5["Info"].attrs["Time"]
            h5["Info"].attrs["time"] = 200.0

    probe = run(
        [
            image_exe,
            *transport_args,
            "--slow_light_time_probe=1",
            f"--slow_light_dump_list={fixture},{second_fixture}",
            "--slow_light_observation_time=150",
            "--parameter_output=none",
        ],
        workdir,
    )
    assert "KPolaris slow-light time probe" in probe.stdout
    assert "dump_time_matching checked" in probe.stdout
    assert "dump_count 2" in probe.stdout
    assert "dump_time_min 100" in probe.stdout
    assert "dump_time_max 200" in probe.stdout


def check_synthetic_iharm_contract(
    image_exe: str,
    workdir: Path,
    trace_exe: str | None,
    analysis_compare_script: str | None,
    csv_output_enabled: bool,
    grmhd_coordinates: set[str],
) -> Path:
    fixture = workdir / "synthetic_iharm_fixture.h5"
    image_path = workdir / "synthetic_iharm_analysis.h5"
    effective_path = workdir / "synthetic_iharm_effective.par"
    write_synthetic_iharm_fixture(fixture)
    transport_args = synthetic_iharm_transport_args(fixture)
    run(
        [
            image_exe,
            *transport_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={image_path}",
            f"--parameter_output={effective_path}",
        ],
        workdir,
    )
    check_image_contract(image_path)
    check_analysis_contract(image_path)
    with h5py.File(image_path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "iharm"
        assert text_attr(h5.attrs["coordinate"]) == "fmks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        radiation = h5["parameters/radiation"].attrs
        assert int(radiation["iharm_n1"]) == 8
        assert int(radiation["iharm_n2"]) == 8
        assert int(radiation["iharm_n3"]) == 8
        assert int(radiation["grmhd_n1"]) == 8
        assert not any(name.startswith("kharma_") for name in radiation.keys())
        assert not any(name.startswith("kharma_") for name in h5.attrs.keys())
        assert not any(
            name.startswith("kharma_")
            for name in h5["frame_0/freq_0"].attrs.keys()
        )
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            269.0930186784772,
            rel=2e-6,
            abs_tol=1e-9,
        )
        analysis = h5["frame_0/analysis"]
        for name in (
            "dominant_beta",
            "dominant_sigma",
            "emission_weighted_beta",
            "emission_weighted_sigma",
        ):
            values = analysis[name][...]
            assert np.isfinite(values).all()
            assert np.all(values > 0.0)

    effective_text = effective_path.read_text(encoding="utf-8")
    assert "# derived_iharm_n1=8\n" in effective_text
    assert "# derived_kharma_n1=" not in effective_text
    if csv_output_enabled:
        check_loader_csv_metadata(
            image_exe,
            transport_args,
            workdir,
            "iharm",
            "# iharm_n1,8\n",
            ("kharma",),
        )

    # Real iHARM archives use both gamma/gam and mmks/fmks group spellings.
    # The aliases describe the same data here and must therefore produce an
    # identical image and analysis payload.
    alias_fixture = workdir / "synthetic_iharm_gamma_mmks.h5"
    shutil.copyfile(fixture, alias_fixture)
    with h5py.File(alias_fixture, "r+") as h5:
        h5.move("/header/gam", "/header/gamma")
        h5.move("/header/geom/fmks", "/header/geom/mmks")
    alias_image = workdir / "synthetic_iharm_gamma_mmks_analysis.h5"
    run(
        [
            image_exe,
            *replace_option(transport_args, "iharm_dump", alias_fixture),
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={alias_image}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(alias_image)
    check_analysis_contract(alias_image)
    compare_single_frequency_images(image_path, alias_image)
    compare_analysis_maps(image_path, alias_image)

    check_loader_dump_time_probe(
        image_exe, transport_args, fixture, workdir, "iharm"
    )

    # Exercise every iHARM coordinate backend present in this build.  The
    # analysis image above deliberately takes a separate driver path, so these
    # fast-light images also test the compiled backend translation units and
    # their runtime dispatcher.  A minimal fmks-only build remains valid.
    coordinate_fluxes = {
        "fmks": 269.0930186784772,
        "cartesian_ks": 269.1037101134578,
        "spherical_ks": 269.16221229297616,
        "mks": 269.16221229297616,
    }
    coordinate_images: dict[str, Path] = {}
    coordinate_multifrequency_images: dict[str, Path] = {}
    multifrequency_supported: bool | None = None
    for coordinate in ("fmks", "cartesian_ks", "spherical_ks", "mks"):
        if coordinate not in grmhd_coordinates:
            continue
        coordinate_path = workdir / f"synthetic_iharm_coordinate_{coordinate}.h5"
        coordinate_args = [
            f"--coordinate={coordinate}"
            if argument.startswith("--coordinate=")
            else argument
            for argument in transport_args
        ]
        run(
            [
                image_exe,
                *coordinate_args,
                "--format=hdf5",
                f"--output={coordinate_path}",
                "--parameter_output=none",
            ],
            workdir,
        )
        check_image_contract(coordinate_path)
        with h5py.File(coordinate_path, "r") as h5:
            assert text_attr(h5.attrs["coordinate"]) == coordinate
            assert int(h5["results"].attrs["returned"]) == 4
            assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
            for stokes in "IQUV":
                assert np.isfinite(h5[f"frame_0/freq_0/{stokes}"][...]).all()
            assert_close(
                float(h5["results"].attrs["flux_I_Jy"]),
                coordinate_fluxes[coordinate],
                rel=2e-6,
                abs_tol=1e-9,
            )
        coordinate_images[coordinate] = coordinate_path

        if multifrequency_supported is False:
            continue
        multifrequency_path = (
            workdir / f"synthetic_iharm_coordinate_{coordinate}_multifrequency.h5"
        )
        multifrequency_cmd = [
            image_exe,
            *coordinate_args,
            "--format=hdf5",
            "--freq_list=100000000000,230000000000",
            f"--output={multifrequency_path}",
            "--parameter_output=none",
        ]
        multifrequency = subprocess.run(
            multifrequency_cmd, cwd=workdir, text=True, capture_output=True
        )
        if multifrequency.returncode == 0:
            multifrequency_supported = True
            check_multifrequency_image_contract(
                multifrequency_path, expected_shape=(2, 2)
            )
            with h5py.File(multifrequency_path, "r") as h5:
                assert text_attr(h5.attrs["coordinate"]) == coordinate
                assert int(h5["results"].attrs["returned"]) == 4
                assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
            coordinate_multifrequency_images[coordinate] = multifrequency_path
        elif "multi-frequency shared-geometry transport is disabled" in (
            multifrequency.stdout + multifrequency.stderr
        ):
            multifrequency_supported = False
        else:
            print(multifrequency.stdout)
            print(multifrequency.stderr, file=sys.stderr)
            raise AssertionError(f"command failed: {' '.join(multifrequency_cmd)}")

    if "spherical_ks" in coordinate_images and "mks" in coordinate_images:
        with h5py.File(coordinate_images["spherical_ks"], "r") as spherical, h5py.File(
            coordinate_images["mks"], "r"
        ) as native_mks:
            for stokes in "IQUV":
                assert np.array_equal(
                    spherical[f"frame_0/freq_0/{stokes}"][...],
                    native_mks[f"frame_0/freq_0/{stokes}"][...],
                )
            assert np.array_equal(
                spherical["frame_0/diagnostics/reason"][...],
                native_mks["frame_0/diagnostics/reason"][...],
            )

    if (
        "spherical_ks" in coordinate_multifrequency_images
        and "mks" in coordinate_multifrequency_images
    ):
        with h5py.File(
            coordinate_multifrequency_images["spherical_ks"], "r"
        ) as spherical, h5py.File(
            coordinate_multifrequency_images["mks"], "r"
        ) as native_mks:
            for frequency_index in range(2):
                for stokes in "IQUV":
                    assert np.array_equal(
                        spherical[f"frame_0/freq_{frequency_index}/{stokes}"][...],
                        native_mks[f"frame_0/freq_{frequency_index}/{stokes}"][...],
                    )
            assert np.array_equal(
                spherical["frame_0/diagnostics/reason"][...],
                native_mks["frame_0/diagnostics/reason"][...],
            )

    assert_failed_with(
        [
            image_exe,
            *transport_args,
            "--coordinate=boyer_lindquist",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_iharm_boyer_lindquist.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "iharm currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks",
    )

    check_spherical_ks_resample_contract(
        image_exe, transport_args, workdir, "iharm"
    )

    # Validate both explicit iHARM primitive-array format diagnostics.  These
    # fixtures remain otherwise valid so a failure cannot be attributed to a
    # missing header field or an HDF5 open error.
    iharm_corruptions: tuple[tuple[str, tuple[int, ...], str], ...] = (
        (
            "rank3",
            (8, 8, 8),
            "iharm /prims dataset must be rank 4",
        ),
        (
            "seven_primitives",
            (8, 8, 8, 7),
            "iharm /prims dimensions do not match header or have fewer than 8 primitives",
        ),
        (
            "header_mismatch",
            (7, 8, 8, 8),
            "iharm /prims dimensions do not match header or have fewer than 8 primitives",
        ),
    )
    for suffix, shape, expected_error in iharm_corruptions:
        corrupt_fixture = workdir / f"synthetic_iharm_{suffix}.h5"
        shutil.copyfile(fixture, corrupt_fixture)
        with h5py.File(corrupt_fixture, "r+") as h5:
            del h5["prims"]
            h5.create_dataset("prims", data=np.zeros(shape, dtype=np.float32))
        assert_failed_with(
            [
                image_exe,
                *replace_option(transport_args, "iharm_dump", corrupt_fixture),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_iharm_{suffix}_output.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            expected_error,
        )

    iharm_header_corruptions: tuple[tuple[str, str, float | int, str], ...] = (
        ("zero_n1", "/header/n1", 0, "iharm header dimensions must be positive"),
        (
            "nonfinite_spin",
            "/header/a",
            math.inf,
            "iharm header spin must be finite and lie in [-1, 1]",
        ),
        (
            "unit_gamma",
            "/header/gam",
            1.0,
            "iharm header gamma must be finite and greater than 1",
        ),
        (
            "nonfinite_start",
            "/header/geom/startx1",
            math.nan,
            "iharm grid starts must be finite",
        ),
        (
            "zero_dx2",
            "/header/geom/dx2",
            0.0,
            "iharm grid spacings must be finite and positive",
        ),
        (
            "negative_dx3",
            "/header/geom/dx3",
            -1.0,
            "iharm grid spacings must be finite and positive",
        ),
        (
            "reversed_radial_bounds",
            "/header/geom/fmks/r_out",
            2.0,
            "iharm radial bounds must be finite, positive, and increasing",
        ),
        (
            "nonfinite_hslope",
            "/header/geom/fmks/hslope",
            math.nan,
            "iharm FMKS/MMKS mapping parameters must be finite",
        ),
        (
            "zero_poly_xt",
            "/header/geom/fmks/poly_xt",
            0.0,
            "iharm FMKS/MMKS mapping parameters must be finite",
        ),
    )
    for suffix, dataset, value, expected_error in iharm_header_corruptions:
        corrupt_fixture = workdir / f"synthetic_iharm_header_{suffix}.h5"
        shutil.copyfile(fixture, corrupt_fixture)
        with h5py.File(corrupt_fixture, "r+") as h5:
            h5[dataset][...] = value
        assert_failed_with(
            [
                image_exe,
                *replace_option(transport_args, "iharm_dump", corrupt_fixture),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_iharm_header_{suffix}_output.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            expected_error,
        )

    iharm_nonfinite_primitives: tuple[
        tuple[str, tuple[int, int, int, int], float], ...
    ] = (
        ("nan_velocity", (0, 0, 0, 2), math.nan),
        ("infinite_field", (0, 0, 0, 7), math.inf),
    )
    for suffix, index, value in iharm_nonfinite_primitives:
        corrupt_fixture = workdir / f"synthetic_iharm_{suffix}.h5"
        shutil.copyfile(fixture, corrupt_fixture)
        with h5py.File(corrupt_fixture, "r+") as h5:
            h5["prims"][index] = value
        assert_failed_with(
            [
                image_exe,
                *replace_option(transport_args, "iharm_dump", corrupt_fixture),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_iharm_{suffix}_output.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            "iharm /prims first eight primitive fields must be finite",
        )

    if trace_exe is None or analysis_compare_script is None:
        return image_path
    trace_path = workdir / "synthetic_iharm_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs",
            f"--output={trace_path}",
        ],
        workdir,
    )
    oracle_json = workdir / "synthetic_iharm_analysis_trace.json"
    oracle = run(
        [
            sys.executable,
            analysis_compare_script,
            str(image_path),
            str(trace_path),
            f"--json-output={oracle_json}",
        ],
        workdir,
    )
    assert "analysis_trace_comparison=pass" in oracle.stdout
    payload = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failed_fields"] == []
    assert len(payload["fields"]) == 23
    assert max(
        float(row.get("l1_relative", 0.0)) for row in payload["fields"].values()
    ) < 3.0e-15
    return image_path


def check_slow_light_vacuum_tail(
    image_exe: str, fixture: Path, transport_args: list[str], workdir: Path,
) -> None:
    """Cover emission with fluid snapshots, then finish the outgoing vacuum."""
    common = [
        image_exe, *transport_args, "--format=hdf5", "--slow_light=1",
        "--slow_light_observation_time=200", "--parameter_output=none",
        f"--slow_light_dump_list={fixture.resolve()},{fixture.resolve()}",
    ]
    probe = run([*common, "--slow_light_time_probe=1",
                 "--slow_light_time_list=0,170"], workdir)
    assert "dump_time_covers_required 1" in probe.stdout
    assert "dump_time_covers_runtime 1" in probe.stdout
    # This camera is at r=100, emission ends at r=38, and the emitting samples
    # span t=52..132. Missing plasma must not be replaced by vacuum, including
    # when the final available state is still on the incoming vacuum segment.
    for end in (100, 10):
        for mode in ('snapshot', 'block'):
            output = workdir / f"kharma_missing_fluid_{mode}_t{end}.h5"
            run([*common, f"--slow_light_step_mode={mode}", "--analysis_mode=0",
                 f"--slow_light_time_list=0,{end}", f"--output={output}"], workdir)
            with h5py.File(output) as h5:
                assert np.all(h5["frame_0/diagnostics/reason"][...] == 6)
        # Default decoupled stepping rejects missing physical support instead
        # of writing a partially completed image. Keep both contracts explicit.
        output = workdir / f"kharma_missing_fluid_default_t{end}.h5"
        output.unlink(missing_ok=True)
        rejected = run([*common, "--analysis_mode=0", f"--slow_light_time_list=0,{end}",
                        f"--output={output}"], workdir, expect_ok=False)
        assert 'outside supplied physical time support' in rejected.stderr
        assert not output.exists()
    for nfreq in (1, 2):
        frequencies = [] if nfreq == 1 else ["--freq_list=100000000000,230000000000"]
        for analysis in (0, 1):
            outputs = []
            for end in (300, 170):
                output = workdir / f"kharma_vacuum_tail_nf{nfreq}_a{analysis}_t{end}.h5"
                result = subprocess.run([
                    *common, *frequencies, f"--analysis_mode={analysis}",
                    f"--slow_light_time_list=0,{end}", f"--output={output}",
                ], cwd=workdir, text=True, capture_output=True)
                if result.returncode:
                    if nfreq == 2 and "multi-frequency shared-geometry transport is disabled" in (
                        result.stdout + result.stderr
                    ):
                        return
                    raise AssertionError(result.stdout + result.stderr)
                outputs.append(output)
            with h5py.File(outputs[0]) as full, h5py.File(outputs[1]) as tail:
                assert np.all(tail["frame_0/diagnostics/reason"][...] == 1)
                assert np.max(tail["frame_0/diagnostics/closure_x"][...]) < 1.e-4
                for f in range(nfreq):
                    for channel in "IQUV":
                        key = f"frame_0/freq_{f}/{channel}"
                        assert relative_l1(full[key][...], tail[key][...]) < 2.e-5
                if analysis:
                    for f in range(nfreq):
                        key = "frame_0/analysis" if nfreq == 1 else f"frame_0/freq_{f}/analysis"
                        for name in full[key]:
                            a, b = full[key][name][...], tail[key][name][...]
                            if np.issubdtype(a.dtype, np.integer):
                                assert np.array_equal(a, b), name
                            else:
                                # Winding sums sampled midpoint azimuths; a new
                                # vacuum boundary changes the final midpoint.
                                rtol = 1.e-3 if name == "photon_ring_winding_estimate" else 2.e-5
                                assert np.allclose(a, b, rtol=rtol, atol=1.e-10), name


def check_synthetic_kharma_contract(
    image_exe: str,
    workdir: Path,
    trace_exe: str | None,
    analysis_compare_script: str | None,
    csv_output_enabled: bool,
) -> Path:
    fixture = workdir / "synthetic_kharma_fixture.phdf"
    image_path = workdir / "synthetic_kharma_analysis.h5"
    effective_path = workdir / "synthetic_kharma_effective.par"
    write_synthetic_kharma_fixture(fixture)
    transport_args = synthetic_kharma_transport_args(fixture)
    check_slow_light_vacuum_tail(image_exe, fixture, transport_args, workdir)
    run(
        [
            image_exe,
            *transport_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={image_path}",
            f"--parameter_output={effective_path}",
        ],
        workdir,
    )
    check_image_contract(image_path)
    check_analysis_contract(image_path)
    with h5py.File(image_path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "kharma"
        assert text_attr(h5.attrs["coordinate"]) == "fmks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        radiation = h5["parameters/radiation"].attrs
        assert int(radiation["kharma_n1"]) == 8
        assert int(radiation["kharma_n2"]) == 8
        assert int(radiation["kharma_n3"]) == 8
        assert int(radiation["grmhd_n1"]) == 8
        assert int(radiation["kharma_reverse_field"]) == 0
        assert not any(name.startswith("iharm_") for name in radiation.keys())
        assert not any(name.startswith("iharm_") for name in h5.attrs.keys())
        assert not any(
            name.startswith("iharm_")
            for name in h5["frame_0/freq_0"].attrs.keys()
        )
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            269.0930186784772,
            rel=2e-6,
            abs_tol=1e-9,
        )
        analysis = h5["frame_0/analysis"]
        for name in (
            "dominant_beta",
            "dominant_sigma",
            "emission_weighted_beta",
            "emission_weighted_sigma",
        ):
            values = analysis[name][...]
            assert np.isfinite(values).all()
            assert np.all(values > 0.0)

    effective_text = effective_path.read_text(encoding="utf-8")
    assert "# derived_kharma_n1=8\n" in effective_text
    assert "# derived_iharm_n1=" not in effective_text
    if csv_output_enabled:
        check_loader_csv_metadata(
            image_exe,
            transport_args,
            workdir,
            "kharma",
            "# kharma_n1,8\n",
            ("iharm",),
        )

    # Repartitioning the same analytic grid into two logical x1 meshblocks
    # must leave both the image and the physical-analysis maps unchanged.
    multiblock_fixture = workdir / "synthetic_kharma_two_blocks.phdf"
    write_synthetic_kharma_fixture(multiblock_fixture, meshblocks_x1=2)
    multiblock_image = workdir / "synthetic_kharma_two_blocks_analysis.h5"
    run(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", multiblock_fixture
            ),
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={multiblock_image}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(multiblock_image)
    check_analysis_contract(multiblock_image)
    compare_single_frequency_images(image_path, multiblock_image)
    compare_analysis_maps(image_path, multiblock_image)

    # Parthenon archives may store /Input/File as either a variable- or
    # fixed-length HDF5 string.  Both forms must decode identically.
    fixed_string_fixture = workdir / "synthetic_kharma_fixed_input.phdf"
    shutil.copyfile(fixture, fixed_string_fixture)
    with h5py.File(fixed_string_fixture, "r+") as h5:
        parameters = text_attr(h5["Input"].attrs["File"])
        del h5["Input"].attrs["File"]
        h5["Input"].attrs.create("File", np.bytes_(parameters))
    fixed_string_image = workdir / "synthetic_kharma_fixed_input_analysis.h5"
    run(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", fixed_string_fixture
            ),
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={fixed_string_image}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(fixed_string_image)
    check_analysis_contract(fixed_string_image)
    compare_single_frequency_images(image_path, fixed_string_image)
    compare_analysis_maps(image_path, fixed_string_image)

    # Exercise the supported native MKS branch independently of FMKS/MMKS.
    mks_fixture = workdir / "synthetic_kharma_mks.phdf"
    shutil.copyfile(fixture, mks_fixture)
    with h5py.File(mks_fixture, "r+") as h5:
        parameters = text_attr(h5["Input"].attrs["File"])
        h5["Input"].attrs.modify(
            "File", parameters.replace("transform = fmks", "transform = mks")
        )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "kharma_dump", mks_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_mks_as_fmks.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA dump declares transform=mks; use --coordinate=mks, spherical_ks, or cartesian_ks, not fmks",
    )
    mks_image = workdir / "synthetic_kharma_mks_analysis.h5"
    run(
        [
            image_exe,
            *[
                "--coordinate=mks"
                if argument.startswith("--coordinate=")
                else argument
                for argument in replace_option(
                    transport_args, "kharma_dump", mks_fixture
                )
            ],
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={mks_image}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(mks_image)
    check_analysis_contract(mks_image)
    with h5py.File(mks_image, "r") as h5:
        assert text_attr(h5.attrs["coordinate"]) == "mks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            268.56679753468,
            rel=2e-6,
            abs_tol=1e-9,
        )

    check_loader_dump_time_probe(
        image_exe, transport_args, fixture, workdir, "kharma"
    )

    check_spherical_ks_resample_contract(
        image_exe, transport_args, workdir, "kharma"
    )

    # Exercise every KHARMA new-format validation branch that can be reached
    # from a well-formed HDF5 container.  Keeping one mutation per fixture also
    # makes each expected diagnostic unambiguous.
    invalid_info_fixture = workdir / "synthetic_kharma_invalid_info.phdf"
    shutil.copyfile(fixture, invalid_info_fixture)
    with h5py.File(invalid_info_fixture, "r+") as h5:
        h5["Info"].attrs["MeshBlockSize"] = np.array([8, 8], dtype=np.int64)
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "kharma_dump", invalid_info_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_invalid_info.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "invalid KHARMA Info NumMeshBlocks/MeshBlockSize",
    )

    empty_input_fixture = workdir / "synthetic_kharma_empty_input.phdf"
    shutil.copyfile(fixture, empty_input_fixture)
    with h5py.File(empty_input_fixture, "r+") as h5:
        h5["Input"].attrs["File"] = ""
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "kharma_dump", empty_input_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_empty_input.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA /Input File attribute is empty",
    )

    missing_dimensions_fixture = workdir / "synthetic_kharma_missing_dimensions.phdf"
    shutil.copyfile(fixture, missing_dimensions_fixture)
    with h5py.File(missing_dimensions_fixture, "r+") as h5:
        parameters = text_attr(h5["Input"].attrs["File"])
        h5["Input"].attrs["File"] = parameters.replace("nx3 = 8\n", "")
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", missing_dimensions_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_missing_dimensions.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "failed to parse KHARMA mesh dimensions from /Input/File",
    )

    unsupported_transform_fixture = (
        workdir / "synthetic_kharma_unsupported_transform.phdf"
    )
    shutil.copyfile(fixture, unsupported_transform_fixture)
    with h5py.File(unsupported_transform_fixture, "r+") as h5:
        parameters = text_attr(h5["Input"].attrs["File"])
        h5["Input"].attrs["File"] = parameters.replace(
            "transform = fmks", "transform = cartesian"
        )
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", unsupported_transform_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_unsupported_transform.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "unsupported KHARMA coordinate transform: cartesian",
    )

    kharma_parameter_corruptions: tuple[
        tuple[str, str, str, str], ...
    ] = (
        (
            "zero_x2_extent",
            "x2max = 1\n",
            "x2max = 0\n",
            "KHARMA grid spacings must be finite and positive",
        ),
        (
            "nonfinite_spin",
            "a = 0.5\n",
            "a = nan\n",
            "KHARMA spin must be finite and lie in [-1, 1]",
        ),
        (
            "unit_gamma",
            f"gamma = {4.0 / 3.0:.17g}\n",
            "gamma = 1\n",
            "KHARMA gamma must be finite and greater than 1",
        ),
        (
            "reversed_radial_bounds",
            "r_out = 40\n",
            "r_out = 1\n",
            "KHARMA radial bounds must be finite, positive, and increasing",
        ),
        (
            "nonfinite_hslope",
            "hslope = 0.3\n",
            "hslope = nan\n",
            "KHARMA coordinate hslope must be finite",
        ),
        (
            "zero_poly_xt",
            "poly_xt = 0.82\n",
            "poly_xt = 0\n",
            "KHARMA FMKS/MMKS mapping parameters must be finite",
        ),
    )
    for suffix, old, new, expected_error in kharma_parameter_corruptions:
        corrupt_fixture = workdir / f"synthetic_kharma_{suffix}.phdf"
        shutil.copyfile(fixture, corrupt_fixture)
        with h5py.File(corrupt_fixture, "r+") as h5:
            parameters = text_attr(h5["Input"].attrs["File"])
            if old not in parameters:
                raise AssertionError(f"missing KHARMA parameter text {old!r}")
            h5["Input"].attrs.modify("File", parameters.replace(old, new))
        assert_failed_with(
            [
                image_exe,
                *replace_option(
                    transport_args, "kharma_dump", corrupt_fixture
                ),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_kharma_{suffix}.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            expected_error,
        )

    missing_dataset_fixture = workdir / "synthetic_kharma_missing_b.phdf"
    shutil.copyfile(fixture, missing_dataset_fixture)
    with h5py.File(missing_dataset_fixture, "r+") as h5:
        del h5["prims.B"]
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "kharma_dump", missing_dataset_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_missing_b.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA new-format phdf requires Blocks/loc.lx123 and prims.rho/u/uvec/B",
    )

    size_mismatch_fixture = workdir / "synthetic_kharma_size_mismatch.phdf"
    shutil.copyfile(fixture, size_mismatch_fixture)
    with h5py.File(size_mismatch_fixture, "r+") as h5:
        h5["Info"].attrs["NumMeshBlocks"] = np.int64(2)
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "kharma_dump", size_mismatch_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_size_mismatch.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA dataset dimensions do not match the documented meshblock layout",
    )

    incomplete_tiling_fixture = (
        workdir / "synthetic_kharma_incomplete_tiling.phdf"
    )
    shutil.copyfile(fixture, incomplete_tiling_fixture)
    with h5py.File(incomplete_tiling_fixture, "r+") as h5:
        parameters = text_attr(h5["Input"].attrs["File"])
        h5["Input"].attrs.modify(
            "File", parameters.replace("nx1 = 8\n", "nx1 = 16\n")
        )
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", incomplete_tiling_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_incomplete_tiling.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA meshblocks must tile the declared uniform global grid exactly",
    )

    zero_meshblock_fixture = workdir / "synthetic_kharma_zero_meshblock.phdf"
    shutil.copyfile(fixture, zero_meshblock_fixture)
    with h5py.File(zero_meshblock_fixture, "r+") as h5:
        h5["Info"].attrs["MeshBlockSize"] = np.array(
            [0, 8, 8], dtype=np.int64
        )
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", zero_meshblock_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_zero_meshblock.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "invalid KHARMA Info NumMeshBlocks/MeshBlockSize",
    )

    for suffix, location, expected_error in (
        (
            "outside_location",
            np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.int64),
            "KHARMA logical meshblock location lies outside the global grid",
        ),
        (
            "duplicate_location",
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
            "KHARMA logical meshblock locations contain a duplicate",
        ),
    ):
        corrupt_fixture = workdir / f"synthetic_kharma_{suffix}.phdf"
        write_synthetic_kharma_fixture(corrupt_fixture, meshblocks_x1=2)
        with h5py.File(corrupt_fixture, "r+") as h5:
            h5["Blocks/loc.lx123"][...] = location
        assert_failed_with(
            [
                image_exe,
                *replace_option(
                    transport_args, "kharma_dump", corrupt_fixture
                ),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_kharma_{suffix}.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            expected_error,
        )

    wrong_shape_fixture = workdir / "synthetic_kharma_wrong_shape.phdf"
    shutil.copyfile(fixture, wrong_shape_fixture)
    with h5py.File(wrong_shape_fixture, "r+") as h5:
        rho = h5["prims.rho"][...]
        del h5["prims.rho"]
        h5.create_dataset("prims.rho", data=rho.reshape(1, 8, 64))
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", wrong_shape_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_wrong_shape.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "KHARMA dataset dimensions do not match the documented meshblock layout",
    )

    empty_meshblock_size = workdir / "synthetic_kharma_empty_meshblock_size.phdf"
    shutil.copyfile(fixture, empty_meshblock_size)
    with h5py.File(empty_meshblock_size, "r+") as h5:
        del h5["Info"].attrs["MeshBlockSize"]
        h5["Info"].attrs.create(
            "MeshBlockSize", np.array([], dtype=np.int64)
        )
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", empty_meshblock_size
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_empty_meshblock_size.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "empty KHARMA attribute: MeshBlockSize",
    )

    scalar_dataset_fixture = workdir / "synthetic_kharma_scalar_rho.phdf"
    shutil.copyfile(fixture, scalar_dataset_fixture)
    with h5py.File(scalar_dataset_fixture, "r+") as h5:
        del h5["prims.rho"]
        h5.create_dataset("prims.rho", data=np.float32(1.0))
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "kharma_dump", scalar_dataset_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_kharma_scalar_rho.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "invalid KHARMA dataset rank: /prims.rho",
    )

    kharma_nonfinite_primitives: tuple[
        tuple[str, tuple[int, ...], float], ...
    ] = (
        ("prims.rho", (0, 0, 0, 0), math.nan),
        ("prims.u", (0, 0, 0, 0), math.inf),
        ("prims.uvec", (0, 0, 0, 0, 0), math.nan),
        ("prims.B", (0, 0, 0, 0, 0), -math.inf),
    )
    for dataset, index, value in kharma_nonfinite_primitives:
        suffix = dataset.replace(".", "_")
        corrupt_fixture = workdir / f"synthetic_kharma_nonfinite_{suffix}.phdf"
        shutil.copyfile(fixture, corrupt_fixture)
        with h5py.File(corrupt_fixture, "r+") as h5:
            h5[dataset][index] = value
        assert_failed_with(
            [
                image_exe,
                *replace_option(
                    transport_args, "kharma_dump", corrupt_fixture
                ),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_kharma_nonfinite_{suffix}.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            f"KHARMA dataset contains nonfinite primitive values: /{dataset}",
        )

    if trace_exe is None or analysis_compare_script is None:
        return image_path
    trace_path = workdir / "synthetic_kharma_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs",
            f"--output={trace_path}",
        ],
        workdir,
    )
    oracle_json = workdir / "synthetic_kharma_analysis_trace.json"
    oracle = run(
        [
            sys.executable,
            analysis_compare_script,
            str(image_path),
            str(trace_path),
            f"--json-output={oracle_json}",
        ],
        workdir,
    )
    assert "analysis_trace_comparison=pass" in oracle.stdout
    payload = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failed_fields"] == []
    assert len(payload["fields"]) == 23
    assert max(
        float(row.get("l1_relative", 0.0)) for row in payload["fields"].values()
    ) < 3.0e-15
    return image_path


def check_synthetic_athenak_contract(
    image_exe: str,
    workdir: Path,
    trace_exe: str | None,
    analysis_compare_script: str | None,
    csv_output_enabled: bool,
) -> Path:
    fixture = workdir / "synthetic_athenak_fixture.bin"
    image_path = workdir / "synthetic_athenak_analysis.h5"
    effective_path = workdir / "synthetic_athenak_effective.par"
    write_synthetic_athenak_fixture(fixture)
    transport_args = synthetic_athenak_transport_args(fixture)
    run(
        [
            image_exe,
            *transport_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={image_path}",
            f"--parameter_output={effective_path}",
        ],
        workdir,
    )
    check_image_contract(image_path)
    check_analysis_contract(image_path)
    with h5py.File(image_path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "athenak"
        assert text_attr(h5.attrs["coordinate"]) == "cartesian_ks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        radiation = h5["parameters/radiation"].attrs
        assert text_attr(radiation["athenak_backend"]) == "direct_cks_meshblocks"
        assert int(radiation["athenak_nblocks"]) == 1
        assert int(radiation["athenak_meshblock_nx1"]) == 8
        assert int(radiation["athenak_meshblock_nx2"]) == 8
        assert int(radiation["athenak_meshblock_nx3"]) == 8
        # Domain-endpoint baseline, independently separated from the sampling
        # optimization (which changes real-snapshot IQUV only at ~1e-15).
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            0.00028529517097446443,
            rel=2e-6,
            abs_tol=1e-13,
        )
        analysis = h5["frame_0/analysis"]
        for name in (
            "dominant_beta",
            "dominant_sigma",
            "emission_weighted_beta",
            "emission_weighted_sigma",
        ):
            values = analysis[name][...]
            assert np.isfinite(values).all()
            assert np.all(values > 0.0)

    effective_text = effective_path.read_text(encoding="utf-8")
    assert "# derived_athenak_backend=direct_cks_meshblocks\n" in effective_text
    assert "# derived_athenak_nblocks=1\n" in effective_text
    assert "# derived_athenak_meshblock_nx1=8\n" in effective_text
    if csv_output_enabled:
        check_loader_csv_metadata(
            image_exe,
            transport_args,
            workdir,
            "athenak",
            "# athenak_backend,direct_cks_meshblocks\n",
            ("iharm", "kharma", "bhac", "hamr"),
        )

    # Athena binary v1.1 permits independently selected 32/64-bit location
    # and variable storage.  The primary fixture uses the common 64/32 layout;
    # this complementary 32/64 file exercises both other supported readers.
    alternate_fixture = workdir / "synthetic_athenak_loc32_var64.bin"
    alternate_path = workdir / "synthetic_athenak_loc32_var64.h5"
    write_synthetic_athenak_fixture(
        alternate_fixture, location_size=4, variable_size=8
    )
    alternate_args = replace_option(
        transport_args, "athenak_dump", alternate_fixture
    )
    run(
        [
            image_exe,
            *alternate_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={alternate_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(alternate_path)
    check_analysis_contract(alternate_path)
    with h5py.File(image_path, "r") as primary, h5py.File(
        alternate_path, "r"
    ) as alternate:
        assert int(alternate["results"].attrs["returned"]) == 4
        assert np.all(alternate["frame_0/diagnostics/reason"][...] == 1)
        assert_close(
            float(alternate["results"].attrs["flux_I_Jy"]),
            float(primary["results"].attrs["flux_I_Jy"]),
            rel=2e-6,
            abs_tol=1e-13,
        )
        for stokes in "IQUV":
            assert relative_l1(
                primary[f"frame_0/freq_0/{stokes}"][...],
                alternate[f"frame_0/freq_0/{stokes}"][...],
            ) < 2.0e-6

    def mutated_athenak_fixture(
        suffix: str, old: bytes, new: bytes
    ) -> Path:
        if len(old) != len(new):
            raise AssertionError("AthenaK mutation must preserve binary offsets")
        data = fixture.read_bytes()
        if data.count(old) != 1:
            raise AssertionError(f"AthenaK mutation marker {old!r} is not unique")
        path = workdir / f"synthetic_athenak_{suffix}.bin"
        path.write_bytes(data.replace(old, new, 1))
        return path

    athenak_header_corruptions = (
        (
            "unsupported_version",
            b"Athena binary output version=1.1",
            b"Athena binary output version=1.0",
            "unsupported AthenaK binary header",
        ),
        (
            "invalid_preheader_size",
            b"pheader size=5",
            b"pheader sizeX5",
            "invalid AthenaK preheader size line",
        ),
        (
            "invalid_variable_count",
            b"nvars=8",
            b"nvarsX8",
            "invalid AthenaK variable count line",
        ),
        (
            "invalid_header_offset",
            b"header offset=",
            b"header offsetX",
            "invalid AthenaK header offset line",
        ),
        (
            "invalid_meshblock_dimensions",
            b"nx1 = 8",
            b"nx1 = 1",
            "invalid AthenaK nvar/meshblock dimensions",
        ),
        (
            "unsupported_location_size",
            b"size of location=8",
            b"size of location=3",
            "unsupported AthenaK floating-point byte size",
        ),
        (
            "unsupported_variable_size",
            b"size of variable=4",
            b"size of variable=2",
            "unsupported AthenaK variable size",
        ),
    )
    for suffix, old, new, expected_error in athenak_header_corruptions:
        corrupt_fixture = mutated_athenak_fixture(suffix, old, new)
        assert_failed_with(
            [
                image_exe,
                *replace_option(transport_args, "athenak_dump", corrupt_fixture),
                "--format=hdf5",
                f"--output={workdir / f'synthetic_athenak_{suffix}.h5'}",
                "--parameter_output=none",
            ],
            workdir,
            expected_error,
        )

    meshblock_mismatch_fixture = workdir / "synthetic_athenak_meshblock_mismatch.bin"
    meshblock_mismatch = bytearray(fixture.read_bytes())
    offset_marker = b"header offset="
    marker_position = meshblock_mismatch.index(offset_marker)
    offset_end = meshblock_mismatch.index(b"\n", marker_position)
    parameter_size = int(
        meshblock_mismatch[marker_position + len(offset_marker) : offset_end]
    )
    binary_offset = offset_end + 1 + parameter_size
    struct.pack_into("<i", meshblock_mismatch, binary_offset + 4, 6)
    meshblock_mismatch_fixture.write_bytes(meshblock_mismatch)
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "athenak_dump", meshblock_mismatch_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_athenak_meshblock_mismatch.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "AthenaK meshblock size mismatch",
    )

    truncated_fixture = workdir / "synthetic_athenak_truncated.bin"
    shutil.copyfile(fixture, truncated_fixture)
    with truncated_fixture.open("r+b") as stream:
        stream.truncate(truncated_fixture.stat().st_size - 1)
    truncated_args = [
        f"--athenak_dump={truncated_fixture.resolve()}"
        if argument.startswith("--athenak_dump=")
        else argument
        for argument in transport_args
    ]
    assert_failed_with(
        [
            image_exe,
            *truncated_args,
            "--format=hdf5",
            f"--output={workdir / 'synthetic_athenak_truncated.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "AthenaK truncated meshblock payload or trailing data",
    )

    # A valid first block must not hide a truncated second block or trailing data.
    raw = fixture.read_bytes()
    huge_dims = raw.replace(b"nx1 = 8", b"nx1 = 2147483647").replace(b"nx2 = 8", b"nx2 = 2147483647")
    huge_dims = huge_dims.replace(f"header offset={parameter_size}".encode(), f"header offset={parameter_size + 18}".encode(), 1)
    nonfinite = bytearray(raw)
    struct.pack_into("<f", nonfinite, binary_offset + 40 + 48, float("nan"))
    reversed_extent = bytearray(raw)
    struct.pack_into("<d", reversed_extent, binary_offset + 40, 41.0)
    corruptions = {
        "trailing_bytes": (raw + b"x", "truncated meshblock payload or trailing data"),
        "partial_second_block": (raw + raw[binary_offset:-1], "truncated meshblock payload or trailing data"),
        "nonfinite_primitive": (nonfinite, "non-finite AthenaK primitive field"),
        "reversed_extent": (reversed_extent, "invalid AthenaK meshblock extent"),
        "negative_header_offset": (raw.replace(f"header offset={parameter_size}".encode(), b"header offset=-1", 1), "invalid AthenaK header offset"),
        "oversized_dimensions": (huge_dims, "payload size overflow"),
    }
    for suffix, (payload, diagnostic) in corruptions.items():
        corrupt = workdir / f"synthetic_athenak_{suffix}.bin"
        corrupt.write_bytes(payload)
        assert_failed_with(
            [image_exe, *replace_option(transport_args, "athenak_dump", corrupt),
             f"--output={workdir / f'athenak_{suffix}.h5'}", "--parameter_output=none"],
            workdir, diagnostic,
        )

    if trace_exe is None or analysis_compare_script is None:
        return image_path
    trace_path = workdir / "synthetic_athenak_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs",
            f"--output={trace_path}",
        ],
        workdir,
    )
    oracle_json = workdir / "synthetic_athenak_analysis_trace.json"
    oracle = run(
        [
            sys.executable,
            analysis_compare_script,
            str(image_path),
            str(trace_path),
            f"--json-output={oracle_json}",
        ],
        workdir,
    )
    assert "analysis_trace_comparison=pass" in oracle.stdout
    payload = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failed_fields"] == []
    assert len(payload["fields"]) == 23
    assert max(
        float(row.get("l1_relative", 0.0)) for row in payload["fields"].values()
    ) < 3.0e-15
    return image_path


def check_synthetic_bhac_contract(
    image_exe: str,
    workdir: Path,
    trace_exe: str | None,
    analysis_compare_script: str | None,
    csv_output_enabled: bool,
) -> Path:
    fixture = workdir / "synthetic_bhac_fixture.dat"
    image_path = workdir / "synthetic_bhac_analysis.h5"
    effective_path = workdir / "synthetic_bhac_effective.par"
    write_synthetic_bhac_fixture(fixture)
    transport_args = synthetic_bhac_transport_args(fixture)
    run(
        [
            image_exe,
            *transport_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={image_path}",
            f"--parameter_output={effective_path}",
        ],
        workdir,
    )
    check_image_contract(image_path)
    check_analysis_contract(image_path)
    with h5py.File(image_path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "bhac"
        assert text_attr(h5.attrs["coordinate"]) == "spherical_ks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        radiation = h5["parameters/radiation"].attrs
        assert text_attr(radiation["bhac_backend"]) == "direct_bhac_mks_amr"
        assert int(radiation["bhac_nblocks"]) == 1
        assert int(radiation["bhac_meshblock_nx1"]) == 8
        assert int(radiation["bhac_meshblock_nx2"]) == 8
        assert int(radiation["bhac_meshblock_nx3"]) == 8
        assert_close(float(radiation["bhac_gamma"]), 4.0 / 3.0, rel=1e-15)
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            265.9757192903112,
            rel=2e-6,
            abs_tol=1e-9,
        )
        analysis = h5["frame_0/analysis"]
        for name in (
            "dominant_beta",
            "dominant_sigma",
            "emission_weighted_beta",
            "emission_weighted_sigma",
        ):
            values = analysis[name][...]
            assert np.isfinite(values).all()
            assert np.all(values > 0.0)

    effective_text = effective_path.read_text(encoding="utf-8")
    assert "# derived_bhac_backend=direct_bhac_mks_amr\n" in effective_text
    assert "# derived_bhac_nblocks=1\n" in effective_text
    assert "# derived_bhac_meshblock_nx1=8\n" in effective_text
    if csv_output_enabled:
        check_loader_csv_metadata(
            image_exe,
            transport_args,
            workdir,
            "bhac",
            "# bhac_backend,direct_bhac_mks_amr\n",
            ("hamr",),
        )

    # The staged cache is a production path, not merely a performance detail:
    # accepting a cache created for a different dump or geometry would silently
    # change the image.  Exercise write and read independently, prove that a
    # read-only cache is self-contained, and reject each serialized contract
    # class with a specific diagnostic.
    cache_path = workdir / "synthetic_bhac.staged"
    cache_write_path = workdir / "synthetic_bhac_cache_write.h5"
    cache_write_args = [
        *transport_args,
        f"--bhac_cache={cache_path}",
        "--bhac_cache_mode=write",
    ]
    run(
        [
            image_exe,
            *cache_write_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={cache_write_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    assert_nonempty_file(cache_path)
    check_image_contract(cache_write_path)
    check_analysis_contract(cache_write_path)
    compare_single_frequency_images(image_path, cache_write_path)
    compare_analysis_maps(image_path, cache_write_path)

    missing_dump = workdir / "synthetic_bhac_removed_source.dat"
    cache_read_path = workdir / "synthetic_bhac_cache_read.h5"
    cache_read_args = [
        *replace_option(transport_args, "bhac_dump", missing_dump),
        f"--bhac_cache={cache_path}",
        "--bhac_cache_mode=read",
    ]
    run(
        [
            image_exe,
            *cache_read_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={cache_read_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(cache_read_path)
    check_analysis_contract(cache_read_path)
    compare_single_frequency_images(image_path, cache_read_path)
    compare_analysis_maps(image_path, cache_read_path)

    cache_bytes = cache_path.read_bytes()

    def mutated_cache(name: str, mutate) -> Path:
        path = workdir / f"synthetic_bhac_cache_{name}.staged"
        data = bytearray(cache_bytes)
        mutate(data)
        path.write_bytes(data)
        return path

    def staged_header_offset(data: bytearray) -> int:
        # 16-byte magic, five uint32 ABI words, uint64 source size, then a
        # uint64-length source path, six int options, and seven Real options.
        source_path_length = struct.unpack_from("<Q", data, 44)[0]
        return 52 + int(source_path_length) + 6 * 4 + 7 * 8

    invalid_magic_cache = mutated_cache(
        "invalid_magic", lambda data: data.__setitem__(0, data[0] ^ 0xFF)
    )
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--bhac_cache={invalid_magic_cache}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_invalid_cache_magic.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "invalid BHAC staged cache magic",
    )

    def invalidate_cache_version(data: bytearray) -> None:
        struct.pack_into("<I", data, 16, 0)

    invalid_abi_cache = mutated_cache("invalid_abi", invalidate_cache_version)
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--bhac_cache={invalid_abi_cache}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_invalid_cache_abi.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC staged cache ABI/version mismatch",
    )

    mismatched_source = workdir / "synthetic_bhac_cache_size_mismatch.dat"
    mismatched_source.write_bytes(fixture.read_bytes() + b"\0")
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "bhac_dump", mismatched_source),
            f"--bhac_cache={cache_path}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_cache_size_mismatch.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC staged cache source file size mismatch",
    )

    assert_failed_with(
        [
            image_exe,
            *transport_args,
            "--bhac_reverse_field=1",
            f"--bhac_cache={cache_path}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_cache_options_mismatch.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC staged cache options mismatch",
    )

    def mismatch_cache_vectors(data: bytearray) -> None:
        struct.pack_into("<i", data, staged_header_offset(data), 2)

    invalid_vectors_cache = mutated_cache(
        "invalid_vectors", mismatch_cache_vectors
    )
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--bhac_cache={invalid_vectors_cache}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_invalid_cache_vectors.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC staged cache vector size mismatch",
    )

    def invalidate_cache_index(data: bytearray) -> None:
        struct.pack_into("<i", data, staged_header_offset(data) + 100, 0)

    invalid_index_cache = mutated_cache("invalid_index", invalidate_cache_index)
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            f"--bhac_cache={invalid_index_cache}",
            "--bhac_cache_mode=read",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_invalid_cache_index.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC staged cache AMR index mismatch",
    )

    # A footer smaller than the fixed BHAC metadata record must be rejected
    # before any mesh or primitive data are interpreted.
    too_small_fixture = workdir / "synthetic_bhac_too_small.dat"
    too_small_fixture.write_bytes(fixture.read_bytes()[:39])
    too_small_args = [
        f"--bhac_dump={too_small_fixture.resolve()}"
        if argument.startswith("--bhac_dump=")
        else argument
        for argument in transport_args
    ]
    assert_failed_with(
        [
            image_exe,
            *too_small_args,
            "--format=hdf5",
            f"--output={workdir / 'synthetic_bhac_too_small.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "BHAC dump is too small",
    )

    if trace_exe is None or analysis_compare_script is None:
        return image_path
    trace_path = workdir / "synthetic_bhac_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs",
            f"--output={trace_path}",
        ],
        workdir,
    )
    oracle_json = workdir / "synthetic_bhac_analysis_trace.json"
    oracle = run(
        [
            sys.executable,
            analysis_compare_script,
            str(image_path),
            str(trace_path),
            f"--json-output={oracle_json}",
        ],
        workdir,
    )
    assert "analysis_trace_comparison=pass" in oracle.stdout
    payload = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failed_fields"] == []
    assert len(payload["fields"]) == 23
    assert max(
        float(row.get("l1_relative", 0.0)) for row in payload["fields"].values()
    ) < 5.0e-15
    return image_path


def check_synthetic_hamr_contract(
    image_exe: str,
    workdir: Path,
    trace_exe: str | None,
    analysis_compare_script: str | None,
    csv_output_enabled: bool,
) -> Path:
    fixture = workdir / "synthetic_hamr_fixture"
    image_path = workdir / "synthetic_hamr_analysis.h5"
    effective_path = workdir / "synthetic_hamr_effective.par"
    if fixture.exists():
        shutil.rmtree(fixture)
    write_synthetic_hamr_fixture(fixture)
    transport_args = synthetic_hamr_transport_args(fixture)
    run(
        [
            image_exe,
            *transport_args,
            "--format=hdf5",
            "--analysis_mode=1",
            f"--output={image_path}",
            f"--parameter_output={effective_path}",
        ],
        workdir,
    )
    check_image_contract(image_path)
    check_analysis_contract(image_path)
    with h5py.File(image_path, "r") as h5:
        assert text_attr(h5.attrs["model"]) == "hamr"
        assert text_attr(h5.attrs["coordinate"]) == "spherical_ks"
        assert int(h5["results"].attrs["returned"]) == 4
        assert np.all(h5["frame_0/diagnostics/reason"][...] == 1)
        radiation = h5["parameters/radiation"].attrs
        assert text_attr(radiation["hamr_backend"]) == "direct_hamr_mks_amr"
        assert int(radiation["hamr_nblocks"]) == 1
        assert int(radiation["hamr_meshblock_nx1"]) == 8
        assert int(radiation["hamr_meshblock_nx2"]) == 8
        assert int(radiation["hamr_meshblock_nx3"]) == 8
        assert_close(float(radiation["hamr_gamma"]), 4.0 / 3.0, rel=1e-15)
        assert_close(float(radiation["hamr_hslope"]), 0.3, rel=1e-15)
        assert_close(float(radiation["hamr_internal_hslope"]), 0.7, rel=1e-15)
        assert text_attr(radiation["hamr_hslope_source"]) == "explicit"
        for attrs in (
            h5.attrs,
            radiation,
            h5["frame_0/freq_0"].attrs,
        ):
            assert not any(name.startswith("bhac_") for name in attrs.keys())
        assert_close(
            float(h5["results"].attrs["flux_I_Jy"]),
            266.98263670547124,
            rel=2e-6,
            abs_tol=1e-9,
        )
        analysis = h5["frame_0/analysis"]
        for name in (
            "dominant_beta",
            "dominant_sigma",
            "emission_weighted_beta",
            "emission_weighted_sigma",
        ):
            values = analysis[name][...]
            assert np.isfinite(values).all()
            assert np.all(values > 0.0)

    effective_text = effective_path.read_text(encoding="utf-8")
    assert "# derived_hamr_backend=direct_hamr_mks_amr\n" in effective_text
    assert "# derived_hamr_internal_hslope=0.69999999999999996\n" in effective_text
    assert "# derived_hamr_nblocks=1\n" in effective_text
    assert "# derived_hamr_meshblock_nx1=8\n" in effective_text
    assert "# derived_bhac_" not in effective_text
    if csv_output_enabled:
        check_loader_csv_metadata(
            image_exe,
            transport_args,
            workdir,
            "hamr",
            "# hamr_backend,direct_hamr_mks_amr\n",
            ("bhac",),
        )
        csv_text = (workdir / "synthetic_hamr_metadata.csv").read_text(
            encoding="utf-8"
        )
        assert "# hamr_hslope,0.30000000000000004\n" in csv_text
        assert "# hamr_internal_hslope,0.69999999999999996\n" in csv_text

    # The emitting endpoints precede fluid time zero in this camera setup;
    # include negative times instead of relying on the old endpoint clamp.
    # A time-independent plasma must reduce to fast light.  The two-dump case
    # isolates interpolation, while the three-dump case crosses an internal
    # time-window boundary and verifies that asynchronous prefetch is exactly
    # equivalent to the serial loader schedule.
    slow_two_path = workdir / "synthetic_hamr_slow_static_two_dump.h5"
    fixture_text = str(fixture.resolve())
    slow_common_args = [
        *transport_args,
        "--format=hdf5",
        "--analysis_mode=1",
        "--slow_light=1",
        "--slow_light_observation_time=100",
        "--parameter_output=none",
    ]
    run(
        [
            image_exe,
            *slow_common_args,
            f"--slow_light_dump_list={fixture_text},{fixture_text}",
            "--slow_light_time_list=-100,200",
            f"--output={slow_two_path}",
        ],
        workdir,
    )
    check_image_contract(slow_two_path)
    check_analysis_contract(slow_two_path)
    with h5py.File(image_path, "r") as fast, h5py.File(slow_two_path, "r") as slow:
        assert int(slow.attrs["slow_light"]) == 1
        assert int(slow.attrs["slow_light_prefetch"]) == 1
        assert_close(float(slow.attrs["slow_light_observation_time"]), 100.0)
        assert text_attr(slow.attrs["slow_light_time_list"]) == "-100,200"
        assert np.all(slow["frame_0/diagnostics/reason"][...] == 1)
        for stokes in "IQUV":
            fast_values = fast[f"frame_0/freq_0/{stokes}"][...]
            slow_values = slow[f"frame_0/freq_0/{stokes}"][...]
            assert relative_l1(fast_values, slow_values) < 1.0e-12
    compare_analysis_maps(image_path, slow_two_path)

    slow_three_paths: list[Path] = []
    for prefetch in (0, 1):
        slow_path = workdir / f"synthetic_hamr_slow_static_prefetch_{prefetch}.h5"
        slow_three_paths.append(slow_path)
        run(
            [
                image_exe,
                *slow_common_args,
                f"--slow_light_prefetch={prefetch}",
                (
                    f"--slow_light_dump_list={fixture_text},{fixture_text},"
                    f"{fixture_text}"
                ),
                "--slow_light_time_list=-100,50,200",
                f"--output={slow_path}",
            ],
            workdir,
        )
        check_image_contract(slow_path)
        check_analysis_contract(slow_path)
        with h5py.File(slow_path, "r") as slow:
            assert int(slow.attrs["slow_light_prefetch"]) == prefetch
            assert np.all(slow["frame_0/diagnostics/reason"][...] == 1)

    with h5py.File(slow_three_paths[0], "r") as serial, h5py.File(
        slow_three_paths[1], "r"
    ) as prefetched, h5py.File(image_path, "r") as fast:
        for stokes in "IQUV":
            serial_values = serial[f"frame_0/freq_0/{stokes}"][...]
            prefetched_values = prefetched[f"frame_0/freq_0/{stokes}"][...]
            fast_values = fast[f"frame_0/freq_0/{stokes}"][...]
            assert np.array_equal(serial_values, prefetched_values)
            assert relative_l1(fast_values, prefetched_values) < 1.0e-6
    compare_analysis_maps(slow_three_paths[0], slow_three_paths[1])

    fast_multi_path = workdir / "synthetic_hamr_fast_multifrequency.h5"
    slow_multi_path = workdir / "synthetic_hamr_slow_multifrequency.h5"
    multifrequency_args = [
        *transport_args,
        "--format=hdf5",
        "--freq_list=100000000000,230000000000",
        "--parameter_output=none",
    ]
    fast_multi = subprocess.run(
        [image_exe, *multifrequency_args, f"--output={fast_multi_path}"],
        cwd=workdir,
        text=True,
        capture_output=True,
    )
    if fast_multi.returncode == 0:
        run(
            [
                image_exe,
                *multifrequency_args,
                "--slow_light=1",
                f"--slow_light_dump_list={fixture_text},{fixture_text}",
                "--slow_light_time_list=-100,200",
                "--slow_light_observation_time=100",
                f"--output={slow_multi_path}",
            ],
            workdir,
        )
        check_multifrequency_image_contract(fast_multi_path, expected_shape=(2, 2))
        check_multifrequency_image_contract(slow_multi_path, expected_shape=(2, 2))
        with h5py.File(fast_multi_path, "r") as fast, h5py.File(
            slow_multi_path, "r"
        ) as slow:
            assert int(slow.attrs["slow_light"]) == 1
            assert np.all(slow["frame_0/diagnostics/reason"][...] == 1)
            for diagnostic in ("reason", "pass_a_steps", "steps", "total_steps"):
                assert np.array_equal(
                    fast[f"frame_0/diagnostics/{diagnostic}"][...],
                    slow[f"frame_0/diagnostics/{diagnostic}"][...],
                )
            for frequency_index in range(2):
                for stokes in "IQUV":
                    fast_values = fast[
                        f"frame_0/freq_{frequency_index}/{stokes}"
                    ][...]
                    slow_values = slow[
                        f"frame_0/freq_{frequency_index}/{stokes}"
                    ][...]
                    assert relative_l1(fast_values, slow_values) < 1.0e-12
    elif "multi-frequency shared-geometry transport is disabled" in (
        fast_multi.stdout + fast_multi.stderr
    ):
        print("H-AMR multi-frequency slow-light contract skipped: capacity is one")
    else:
        print(fast_multi.stdout)
        print(fast_multi.stderr, file=sys.stderr)
        raise AssertionError("H-AMR fast multi-frequency command failed")

    # The loader accepts a dump directory, its parameters file, or any
    # new_dump* file next to parameters.  All three entry points must resolve
    # to the same ordered chunk set and therefore the same image.
    for entry_name, entry_path in (
        ("parameters_entry", fixture / "parameters"),
        ("data_entry", fixture / "new_dump0000"),
    ):
        entry_image = workdir / f"synthetic_hamr_{entry_name}.h5"
        run(
            [
                image_exe,
                *replace_option(transport_args, "hamr_dump", entry_path),
                "--format=hdf5",
                f"--output={entry_image}",
                "--parameter_output=none",
            ],
            workdir,
        )
        check_image_contract(entry_image)
        compare_single_frequency_images(image_path, entry_image)

    # Block zero maps to root coordinate (0,0,0) in every documented linear
    # ordering, so these runs isolate the dispatcher without changing physics.
    for root_order in ("x1x2x3", "x1x3x2", "x3x2x1"):
        ordered_image = workdir / f"synthetic_hamr_root_order_{root_order}.h5"
        run(
            [
                image_exe,
                *transport_args,
                f"--hamr_root_order={root_order}",
                "--format=hdf5",
                f"--output={ordered_image}",
                "--parameter_output=none",
            ],
            workdir,
        )
        check_image_contract(ordered_image)
        compare_single_frequency_images(image_path, ordered_image)

    def mutated_hamr_fixture(suffix: str, mutate_parameters) -> Path:
        path = workdir / f"synthetic_hamr_{suffix}"
        if path.exists():
            shutil.rmtree(path)
        shutil.copytree(fixture, path)
        parameters = bytearray((path / "parameters").read_bytes())
        mutate_parameters(parameters)
        (path / "parameters").write_bytes(parameters)
        return path

    def set_block_id(data: bytearray, block_id: int) -> None:
        struct.pack_into("<i", data, 0x108, block_id)

    child_fixture = mutated_hamr_fixture(
        "child_id", lambda data: set_block_id(data, 8)
    )
    child_images: list[Path] = []
    for id_order in ("root_major", "child_major"):
        child_image = workdir / f"synthetic_hamr_id_order_{id_order}.h5"
        child_images.append(child_image)
        run(
            [
                image_exe,
                *replace_option(transport_args, "hamr_dump", child_fixture),
                f"--hamr_id_order={id_order}",
                "--format=hdf5",
                f"--output={child_image}",
                "--parameter_output=none",
            ],
            workdir,
        )
        check_image_contract(child_image)
    compare_single_frequency_images(child_images[0], child_images[1])

    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", child_fixture),
            "--hamr_id_order=invalid",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_invalid_id_order.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "unsupported H-AMR id_order: invalid",
    )
    assert_failed_with(
        [
            image_exe,
            *transport_args,
            "--hamr_root_order=invalid",
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_invalid_root_order.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "unsupported H-AMR root_order: invalid",
    )

    def zero_chunk_blocks(data: bytearray) -> None:
        struct.pack_into("<i", data, 0x008, 0)

    corrupt_header_fixture = mutated_hamr_fixture(
        "corrupt_header", zero_chunk_blocks
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", corrupt_header_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_corrupt_header.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "unsupported or corrupt H-AMR parameters header",
    )

    def require_missing_second_record(data: bytearray) -> None:
        struct.pack_into("<i", data, 0x00C, 2)

    missing_table_fixture = mutated_hamr_fixture(
        "missing_block_table", require_missing_second_record
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", missing_table_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_missing_block_table.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR parameters do not contain the expected block-id table",
    )

    def deactivate_block(data: bytearray) -> None:
        struct.pack_into("<i", data, 0x10C, 0)

    inactive_block_fixture = mutated_hamr_fixture(
        "inactive_block", deactivate_block
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", inactive_block_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_inactive_block.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR block-id table contains an inactive block",
    )

    outside_slot_fixture = mutated_hamr_fixture(
        "outside_block_slot", lambda data: set_block_id(data, 72)
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", outside_slot_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_outside_block_slot.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR block id is outside max_block_slots",
    )

    no_data_fixture = workdir / "synthetic_hamr_no_data"
    if no_data_fixture.exists():
        shutil.rmtree(no_data_fixture)
    no_data_fixture.mkdir()
    shutil.copyfile(fixture / "parameters", no_data_fixture / "parameters")
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", no_data_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_no_data.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR dump directory contains no new_dump* files",
    )

    too_few_variables_fixture = workdir / "synthetic_hamr_too_few_variables"
    if too_few_variables_fixture.exists():
        shutil.rmtree(too_few_variables_fixture)
    shutil.copytree(fixture, too_few_variables_fixture)
    (too_few_variables_fixture / "new_dump0000").write_bytes(
        (fixture / "new_dump0000").read_bytes()[: 8 * 8 * 8 * 8 * 4]
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "hamr_dump", too_few_variables_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_too_few_variables.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR data file has fewer variables than expected",
    )

    invalid_second_chunk_fixture = workdir / "synthetic_hamr_invalid_second_chunk"
    if invalid_second_chunk_fixture.exists():
        shutil.rmtree(invalid_second_chunk_fixture)
    shutil.copytree(fixture, invalid_second_chunk_fixture)
    (invalid_second_chunk_fixture / "new_dump0001").write_bytes(b"\0")
    assert_failed_with(
        [
            image_exe,
            *replace_option(
                transport_args, "hamr_dump", invalid_second_chunk_fixture
            ),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_invalid_second_chunk.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR data file size is not an integer number of blocks",
    )

    def two_blocks_per_chunk(data: bytearray) -> None:
        struct.pack_into("<i", data, 0x008, 2)

    too_many_blocks_fixture = mutated_hamr_fixture(
        "too_many_blocks", two_blocks_per_chunk
    )
    raw_block = (fixture / "new_dump0000").read_bytes()
    (too_many_blocks_fixture / "new_dump0000").write_bytes(raw_block + raw_block)
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", too_many_blocks_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_too_many_blocks.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR data files contain more blocks than parameters",
    )

    def declare_two_blocks(data: bytearray) -> None:
        struct.pack_into("<i", data, 0x00C, 2)
        data.extend(b"\0" * 12)
        struct.pack_into("<3i", data, 0x114, 1, 1, 0)

    missing_data_block_fixture = mutated_hamr_fixture(
        "missing_data_block", declare_two_blocks
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", missing_data_block_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_missing_data_block.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR data files do not match parameters total_blocks",
    )

    def one_root_grid(data: bytearray) -> None:
        struct.pack_into("<3i", data, 0x050, 1, 1, 1)

    unsupported_morton_fixture = mutated_hamr_fixture(
        "unsupported_morton", one_root_grid
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", unsupported_morton_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_unsupported_morton.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "unsupported H-AMR Morton root grid dimensions",
    )

    def invalid_morton_code(data: bytearray) -> None:
        struct.pack_into("<3i", data, 0x050, 2, 1, 2)
        set_block_id(data, 18)

    invalid_morton_fixture = mutated_hamr_fixture(
        "invalid_morton_code", invalid_morton_code
    )
    assert_failed_with(
        [
            image_exe,
            *replace_option(transport_args, "hamr_dump", invalid_morton_fixture),
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_invalid_morton_code.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR Morton root id decodes outside root grid",
    )

    small_header_fixture = workdir / "synthetic_hamr_small_header"
    if small_header_fixture.exists():
        shutil.rmtree(small_header_fixture)
    shutil.copytree(fixture, small_header_fixture)
    with (small_header_fixture / "parameters").open("r+b") as stream:
        stream.truncate(0x107)
    small_header_args = [
        f"--hamr_dump={small_header_fixture.resolve()}"
        if argument.startswith("--hamr_dump=")
        else argument
        for argument in transport_args
    ]
    assert_failed_with(
        [
            image_exe,
            *small_header_args,
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_small_header.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR parameters file is too small",
    )

    truncated_data_fixture = workdir / "synthetic_hamr_truncated_data"
    if truncated_data_fixture.exists():
        shutil.rmtree(truncated_data_fixture)
    shutil.copytree(fixture, truncated_data_fixture)
    data_path = truncated_data_fixture / "new_dump0000"
    with data_path.open("r+b") as stream:
        stream.truncate(data_path.stat().st_size - 1)
    truncated_data_args = [
        f"--hamr_dump={truncated_data_fixture.resolve()}"
        if argument.startswith("--hamr_dump=")
        else argument
        for argument in transport_args
    ]
    assert_failed_with(
        [
            image_exe,
            *truncated_data_args,
            "--format=hdf5",
            f"--output={workdir / 'synthetic_hamr_truncated_data.h5'}",
            "--parameter_output=none",
        ],
        workdir,
        "H-AMR data file size does not match parameters block geometry",
    )

    if trace_exe is None or analysis_compare_script is None:
        return image_path
    trace_path = workdir / "synthetic_hamr_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs,stokes",
            f"--output={trace_path}",
        ],
        workdir,
    )
    oracle_json = workdir / "synthetic_hamr_analysis_trace.json"
    oracle = run(
        [
            sys.executable,
            analysis_compare_script,
            str(image_path),
            str(trace_path),
            f"--json-output={oracle_json}",
        ],
        workdir,
    )
    assert "analysis_trace_comparison=pass" in oracle.stdout
    payload = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["failed_fields"] == []
    assert len(payload["fields"]) == 23
    assert max(
        float(row.get("l1_relative", 0.0)) for row in payload["fields"].values()
    ) < 32 * np.finfo(np.float64).eps  # cumulative trace path differences round differently

    slow_trace_path = workdir / "synthetic_hamr_slow_trace.h5"
    run(
        [
            trace_exe,
            *transport_args,
            "--trace_mode=image",
            "--trace_precision=double",
            "--trace_layout=ragged",
            "--trace_compression=0",
            "--trace_stride=1",
            "--max_trace_samples=4096",
            "--trace_fields=lambda,coords,plasma,coeffs,stokes",
            "--slow_light=1",
            f"--slow_light_dump_list={fixture_text},{fixture_text}",
            "--slow_light_time_list=-100,200",
            "--slow_light_observation_time=100",
            f"--output={slow_trace_path}",
        ],
        workdir,
    )
    with h5py.File(trace_path, "r") as fast, h5py.File(
        slow_trace_path, "r"
    ) as slow:
        assert int(slow.attrs["slow_light"]) == 1
        assert_close(float(slow.attrs["slow_light_observation_time"]), 100.0)
        for group_name in ("rays", "trace"):
            assert set(fast[group_name].keys()) == set(slow[group_name].keys())
            for dataset_name in fast[group_name].keys():
                fast_values = fast[f"{group_name}/{dataset_name}"][...]
                slow_values = slow[f"{group_name}/{dataset_name}"][...]
                if np.issubdtype(fast_values.dtype, np.integer):
                    assert np.array_equal(fast_values, slow_values)
                else:
                    assert relative_l1(fast_values, slow_values) < 1.0e-12
    return image_path


def compare_analysis_maps(left: Path, right: Path) -> None:
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        a_group = a["frame_0/analysis"]
        b_group = b["frame_0/analysis"]
        assert set(a_group.keys()) == set(b_group.keys())
        for name in a_group.keys():
            left_values = a_group[name][...]
            right_values = b_group[name][...]
            if np.issubdtype(left_values.dtype, np.integer):
                assert np.array_equal(left_values, right_values)
            elif name in {
                "contribution_closure_max_abs",
                "contribution_closure_relative_l1",
            }:
                assert np.nanmax(np.abs(left_values)) < 1.0e-10
                assert np.nanmax(np.abs(right_values)) < 1.0e-10
            else:
                assert np.allclose(left_values, right_values, rtol=1e-12, atol=1e-30)


def assert_nonempty_file(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0


def check_version_output(executable: str, expected_version: str, workdir: Path) -> None:
    result = run([executable, "--version"], workdir)
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"KPolaris {expected_version}"
    fields = dict(line.split(" ", 1) for line in lines[1:])
    assert fields["revision"]
    assert fields["source_dirty"] in {"unknown", "false", "true"}
    assert fields["source_fingerprint"] == "unknown" or len(
        fields["source_fingerprint"]
    ) == 64
    assert fields["compiler"]
    assert fields["build_type"]


def check_cli_failures(
    image_exe: str,
    trace_exe: str | None,
    workdir: Path,
    compiled_riaf_emission_type: int,
) -> None:
    image_base = [image_exe, "--model=riaf", "--coordinate=boyer_lindquist"]
    assert_failed_with([*image_base, "not-an-option"], workdir, "expected --key=value")
    assert_failed_with(
        [*image_base, "--definitely_unknown=1"], workdir, "unknown option"
    )
    assert_failed_with([*image_base, "--nx=0"], workdir, "nx and ny must be positive")
    assert_failed_with([*image_base, "--spin=2"], workdir, "spin must be finite")
    assert_failed_with(
        [*image_base, "--adaptive_tolerance=0"],
        workdir,
        "adaptive_tolerance must be finite and positive",
    )
    assert_failed_with(
        [*image_base, f"--parameter_file={workdir / 'missing.par'}"],
        workdir,
        "failed to open parameter file",
    )
    if compiled_riaf_emission_type > 0:
        conflicting_type = 4 if compiled_riaf_emission_type == 1 else 1
        assert_failed_with(
            [*image_base, f"--emission_type={conflicting_type}"],
            workdir,
            "conflicts with compile-time emission type",
        )

    malformed_params = workdir / "malformed.par"
    malformed_params.write_text("nx=\n", encoding="utf-8")
    assert_failed_with(
        [*image_base, f"--parameter_file={malformed_params}"],
        workdir,
        "empty value",
    )

    if trace_exe is not None:
        assert_failed_with(
            [trace_exe, "--trace_compression=10"],
            workdir,
            "trace_compression must lie in [0, 9]",
        )
        assert_failed_with(
            [trace_exe, "--trace_stride=0"],
            workdir,
            "trace_stride must be positive",
        )
        corrupt_dump = workdir / "corrupt_iharm.h5"
        with h5py.File(corrupt_dump, "w") as h5:
            h5.attrs["deliberately_corrupt"] = 1
        corrupt = run(
            [
                trace_exe,
                "--model=iharm",
                "--coordinate=fmks",
                f"--iharm_dump={corrupt_dump}",
                f"--output={workdir / 'corrupt_trace.h5'}",
            ],
            workdir,
            expect_ok=False,
        )
        if "error:" not in (corrupt.stdout + corrupt.stderr):
            raise AssertionError("corrupt iHARM dump did not produce a controlled error")


def check_trace_contract(
    trace_exe: str,
    workdir: Path,
    trace_diagnostics_script: Path | None = None,
    trace_bins_script: Path | None = None,
) -> None:
    slow_multi = run(
        [
            trace_exe,
            "--model=riaf",
            "--freq_list=100000000000,230000000000",
            "--slow_light=1",
            "--trace_mode=single",
            "--ix=0",
            "--iy=0",
            "--nx=2",
            "--ny=2",
            "--max_trace_samples=8",
            "--trace_stride=8",
            f"--output={workdir / 'trace_bad.h5'}",
        ],
        workdir,
        expect_ok=False,
    )
    assert "multi-frequency slow-light trace" in (slow_multi.stderr + slow_multi.stdout)

    trace_path = workdir / "trace_single.h5"
    trace_parameters = workdir / "trace_single.effective.par"
    run(
        [
            trace_exe,
            "--model=riaf",
            "--freq_list=230000000000",
            "--use_pinhole_pixel_bias=1",  # frozen legacy trace golden
            "--trace_mode=single",
            "--ix=0",
            "--iy=0",
            "--nx=2",
            "--ny=2",
            "--radius=100",
            "--max_trace_samples=8",
            "--trace_stride=8",
            "--trace_fields=lambda,coords,coeffs,stokes",
            f"--parameter_output={trace_parameters}",
            f"--output={trace_path}",
        ],
        workdir,
    )
    with h5py.File(trace_path, "r") as h5:
        assert text_attr(h5.attrs["schema"]) == "kpolaris_trace"
        assert int(h5.attrs["schema_version"]) == 2
        assert text_attr(h5.attrs["code"]) == "KPolaris"
        assert text_attr(h5.attrs["code_version"])
        assert text_attr(h5.attrs["code_revision"])
        assert int(h5.attrs["code_source_dirty"]) in {-1, 0, 1}
        assert len(text_attr(h5.attrs["code_source_fingerprint"])) == 64
        assert text_attr(h5.attrs["compiler_id"])
        assert text_attr(h5.attrs["compiler_version"])
        assert text_attr(h5.attrs["build_type"])
        assert text_attr(h5.attrs["effective_parameter_file"]) == str(trace_parameters)
        assert text_attr(h5["parameters"].attrs["effective_parameter_file"]) == str(trace_parameters)
        assert int(h5.attrs["nfreq"]) == 1
        assert float(h5.attrs["frequency_hz"]) == 230000000000.0
        assert "sample_offset" in h5["rays"]
        assert "lambda" in h5["trace"]
        assert text_attr(h5.attrs["trace_layout"]) == "ragged"
        assert h5["rays/sample_count"].dtype == np.dtype("int32")
        assert h5["rays/sample_offset"].dtype == np.dtype("int64")
        assert h5["trace/lambda"].dtype in {np.dtype("float32"), np.dtype("float64")}
        assert list(h5["rays/sample_count"][...]) == [8]
        assert list(h5["rays/sample_offset"][...]) == [0, 8]
        assert_close(float(h5["trace/lambda"][0]), 0.0010000000474974513, rel=1e-6, abs_tol=1e-12)
        assert_close(float(h5["trace/r"][0]), 1.4158214330673218, rel=1e-6, abs_tol=1e-9)
        assert_close(float(h5["trace/theta"][0]), 2.199409008026123, rel=1e-6, abs_tol=1e-9)
        assert list(h5["rays/pass_a_steps"][...]) == [271]
        assert list(h5["rays/pass_b_steps"][...]) == [280]
        for basis in ("propagated", "observed"):
            for stokes in "IQUV":
                name = f"final_{basis}_{stokes}_inv"
                assert h5[f"rays/{name}"].shape == (1,)
                assert np.isfinite(h5[f"rays/{name}"][...]).all()
        derived = h5["derived"]
        assert int(derived.attrs["available"]) == 1
        assert list(derived["complete"][...]) == [0]
        assert "radius-sorted" in text_attr(derived.attrs["emissivity_formation_definition"])
        radii = h5["trace/r"][...].astype(float)
        weights = np.maximum(h5["trace/jI"][...].astype(float), 0) * np.abs(
            h5["trace/dlambda"][...].astype(float))
        order = np.argsort(radii)
        cumulative = np.cumsum(weights[order])
        if cumulative[-1] > 0:
            expected_radii = radii[order][np.searchsorted(
                cumulative, np.array([0.05, 0.5, 0.95]) * cumulative[-1])]
            for name, expected in zip(("low", "median", "high"), expected_radii):
                assert_close(float(derived[f"emissivity_formation_radius_{name}"][0]),
                             float(expected), rel=1e-12)
        assert derived["linear_fraction"].shape == (8,)
        assert derived["evpa_unwrapped_rad"].shape == (8,)
        assert np.isfinite(derived["faraday_operator_depth"][...]).all()
        assert np.all(
            derived["emissivity_formation_radius_low"][...]
            <= derived["emissivity_formation_radius_median"][...]
        )
        assert np.all(
            derived["emissivity_formation_radius_median"][...]
            <= derived["emissivity_formation_radius_high"][...]
        )

    assert_nonempty_file(trace_parameters)
    effective_text = trace_parameters.read_text(encoding="utf-8")
    for expected in (
        "# KPolaris effective trace parameter file",
        "# code_revision=",
        "model=riaf",
        "trace_mode=single",
        "trace_stride=8",
        "freq_list=230000000000",
        f"parameter_output={trace_parameters}",
    ):
        assert expected in effective_text
    for image_only in ("analysis_mode=", "split_transport="):
        assert image_only not in effective_text

    replay_path = workdir / "trace_single_replay.h5"
    run(
        [
            trace_exe,
            f"--parameter_file={trace_parameters}",
            f"--output={replay_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    with h5py.File(trace_path, "r") as original, h5py.File(replay_path, "r") as replay:
        assert "effective_parameter_file" not in replay.attrs
        for basis in ("propagated", "observed"):
            for stokes in "IQUV":
                name = f"rays/final_{basis}_{stokes}_inv"
                assert np.allclose(original[name][...], replay[name][...], rtol=1.0e-6, atol=1.0e-12)

    source_fov_path = workdir / "trace_source_fov.h5"
    run(
        [
            trace_exe,
            "--model=riaf",
            "--coordinate=boyer_lindquist",
            "--camera=pinhole",
            "--nx=1",
            "--ny=1",
            "--ix=0",
            "--iy=0",
            "--dsource=8100",
            "--fovx_dsource=40",
            "--fovy_dsource=20",
            "--max_trace_samples=1",
            f"--output={source_fov_path}",
        ],
        workdir,
    )
    with h5py.File(source_fov_path, "r") as h5:
        camera = h5["parameters/camera"].attrs
        assert_close(float(camera["fovx_dsource"]), 40.0, rel=1e-12)
        assert_close(float(camera["fovy_dsource"]), 20.0, rel=1e-12)
        assert float(camera["image_width_x_M"]) > 0.0
        assert float(camera["image_width_y_M"]) > 0.0

    if trace_diagnostics_script is not None:
        diag_path = workdir / "trace_single_diag.png"
        run(
            [
                sys.executable,
                str(trace_diagnostics_script),
                str(trace_path),
                "--field",
                "r",
                "--output",
                str(diag_path),
            ],
            workdir,
        )
        assert_nonempty_file(diag_path)
        history_path = workdir / "trace_single_history.png"
        run(
            [
                sys.executable,
                str(trace_diagnostics_script),
                str(trace_path),
                "--history-ray-index=0",
                "--output",
                str(history_path),
            ],
            workdir,
        )
        assert_nonempty_file(history_path)

    if trace_bins_script is not None:
        bins_prefix = workdir / "trace_single_bins"
        run(
            [
                sys.executable,
                str(trace_bins_script),
                str(trace_path),
                "--output-prefix",
                str(bins_prefix),
                "--r-range=0,10",
                "--theta-range=0,pi",
                "--bins=8",
                "--panels=dI,jI,path_length",
            ],
            workdir,
        )
        assert_nonempty_file(workdir / "trace_single_bins_rtheta.png")
        assert_nonempty_file(workdir / "trace_single_bins_meridional.png")

    multi_trace_path = workdir / "trace_multifrequency.h5"
    run(
        [
            trace_exe,
            "--model=riaf",
            "--freq_list=100000000000,230000000000",
            "--use_pinhole_pixel_bias=1",  # frozen legacy trace golden
            "--trace_mode=single",
            "--ix=0",
            "--iy=0",
            "--nx=2",
            "--ny=2",
            "--radius=100",
            "--max_trace_samples=8",
            "--trace_stride=8",
            "--trace_fields=lambda,coords,coeffs,stokes",
            f"--output={multi_trace_path}",
        ],
        workdir,
    )
    with h5py.File(multi_trace_path, "r") as h5:
        assert text_attr(h5.attrs["schema"]) == "kpolaris_trace"
        assert int(h5.attrs["schema_version"]) == 3
        assert int(h5.attrs["nfreq"]) == 2
        assert list(h5["grid/frequency_hz"][...]) == [100000000000.0, 230000000000.0]
        assert "sample_offset" in h5["rays"]
        assert text_attr(h5.attrs["trace_layout"]) == "ragged"
        assert h5["rays/sample_count"].dtype == np.dtype("int32")
        assert h5["rays/sample_offset"].dtype == np.dtype("int64")
        assert "lambda" in h5["trace/shared"]
        assert "r" in h5["trace/shared"]
        assert float(h5["trace/freq_0"].attrs["frequency_hz"]) == 100000000000.0
        assert float(h5["trace/freq_1"].attrs["frequency_hz"]) == 230000000000.0
        assert "SI" in h5["trace/freq_0"]
        assert "SI" in h5["trace/freq_1"]
        for fi in (0, 1):
            terminal = h5[f"trace/freq_{fi}/rays"]
            for basis in ("propagated", "observed"):
                for stokes in "IQUV":
                    assert terminal[f"final_{basis}_{stokes}_inv"].shape == (1,)
            assert list(terminal["sample_count"][...]) == list(h5["rays/sample_count"][...])
        assert np.array_equal(h5["trace/freq_0/rays/final_propagated_I_inv"][...],
                              h5["rays/final_propagated_I_inv"][...])
        assert not np.array_equal(h5["trace/freq_1/rays/final_propagated_I_inv"][...],
                                  h5["rays/final_propagated_I_inv"][...])
        assert int(h5["trace/freq_0/derived"].attrs["available"]) == 1
        assert int(h5["trace/freq_1/derived"].attrs["available"]) == 1
        assert h5["trace/freq_0/derived/linear_fraction"].shape == (8,)
        assert h5["trace/freq_1/derived/linear_fraction"].shape == (8,)
        assert list(h5["rays/sample_count"][...]) == [8]
        assert list(h5["rays/sample_offset"][...]) == [0, 8]
        assert_close(float(h5["trace/shared/lambda"][0]), 0.0010000000474974513, rel=1e-6, abs_tol=1e-12)
        assert_close(float(h5["trace/shared/r"][0]), 1.4158214330673218, rel=1e-6, abs_tol=1e-9)
        assert list(h5["rays/pass_a_steps"][...]) == [271]
        assert list(h5["rays/pass_b_steps"][...]) == [280]

    if trace_diagnostics_script is not None:
        diag_path = workdir / "trace_multifrequency_diag.png"
        run(
            [
                sys.executable,
                str(trace_diagnostics_script),
                str(multi_trace_path),
                "--field",
                "r",
                "--freq-index=1",
                "--output",
                str(diag_path),
            ],
            workdir,
        )
        assert_nonempty_file(diag_path)
        history_path = workdir / "trace_multifrequency_history.png"
        run(
            [
                sys.executable,
                str(trace_diagnostics_script),
                str(multi_trace_path),
                "--history-ray-index=0",
                "--freq-index=1",
                "--output",
                str(history_path),
            ],
            workdir,
        )
        assert_nonempty_file(history_path)

    if trace_bins_script is not None:
        bins_prefix = workdir / "trace_multifrequency_bins"
        run(
            [
                sys.executable,
                str(trace_bins_script),
                str(multi_trace_path),
                "--output-prefix",
                str(bins_prefix),
                "--freq-index=1",
                "--r-range=0,10",
                "--theta-range=0,pi",
                "--bins=8",
                "--panels=dI,jI,path_length",
            ],
            workdir,
        )
        assert_nonempty_file(workdir / "trace_multifrequency_bins_rtheta.png")
        assert_nonempty_file(workdir / "trace_multifrequency_bins_meridional.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-exe", required=True)
    parser.add_argument("--plot-script", required=True)
    parser.add_argument("--analysis-script")
    parser.add_argument("--analysis-compare-script")
    parser.add_argument("--compare-script")
    parser.add_argument("--trace-diagnostics-script")
    parser.add_argument("--trace-bins-script")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--trace-exe")
    parser.add_argument("--torus-image-exe")
    parser.add_argument("--iharm-image-exe")
    parser.add_argument("--kharma-image-exe")
    parser.add_argument("--athenak-image-exe")
    parser.add_argument("--bhac-image-exe")
    parser.add_argument("--hamr-image-exe")
    parser.add_argument("--csv-output-enabled", action="store_true")
    parser.add_argument("--torus-parameter-file", type=Path)
    parser.add_argument("--compiled-riaf-emission-type", required=True, type=int)
    parser.add_argument("--compiled-grmhd-coordinates", default="fmks")
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    check_version_output(args.image_exe, args.expected_version, workdir)
    if args.trace_exe:
        check_version_output(args.trace_exe, args.expected_version, workdir)
    check_cli_failures(
        args.image_exe,
        args.trace_exe,
        workdir,
        args.compiled_riaf_emission_type,
    )

    image_path = workdir / "riaf_contract.h5"
    params_path = workdir / "riaf_contract.h5.params"
    run([args.image_exe, *image_args(image_path, params_path)], workdir)
    params_text = params_path.read_text(encoding="utf-8")
    for marker in (
        f"# code_version={args.expected_version}",
        "# code_revision=",
        "# code_source_dirty=",
        "# compiler_id=",
        "# compiler_version=",
        "# build_type=",
    ):
        assert marker in params_text
    check_image_contract(image_path, expect_golden=True)

    # Defaults represent pixel centers. Explicitly replaying that choice must
    # produce the same rays and Stokes, and outputs declare the actual basis.
    centered = workdir / 'centered_default.h5'
    centered_args = [x for x in image_args(centered) if not x.startswith('--use_pinhole_pixel_bias=')]
    run([args.image_exe, *centered_args], workdir)
    replay_centered = workdir / 'centered_explicit.h5'
    run([args.image_exe, *centered_args, '--use_pinhole_pixel_bias=0', f'--output={replay_centered}'], workdir)
    with h5py.File(centered) as h, h5py.File(replay_centered) as r:
        assert h.attrs['use_pinhole_pixel_bias'] == 0
        assert h.attrs['x_offset'] == 0
        assert text_attr(h.attrs['image_array_order']) == 'y,x'
        assert text_attr(h.attrs['polarization_conventions_version']) == '2'
        assert 'celestial position angle not assigned' in text_attr(h.attrs['sky_orientation'])
        assert 'V=-2Im' in text_attr(h.attrs['stokes_definition'])
        np.testing.assert_allclose(h['grid/x'][...], -h['grid/x'][...][::-1], atol=1e-15)
        for s in 'IQUV':
            np.testing.assert_array_equal(h[f'frame_0/freq_0/{s}'][...], r[f'frame_0/freq_0/{s}'][...])

    invalid_path = workdir / "riaf_contract_invalid_rays.h5"
    run(
        [
            args.image_exe,
            *image_args(invalid_path),
            "--nx=2",
            "--ny=2",
            "--max_steps=1",
        ],
        workdir,
    )
    check_invalid_pixel_contract(invalid_path)

    if args.torus_image_exe:
        check_version_output(args.torus_image_exe, args.expected_version, workdir)
        torus_path = workdir / "torus_contract.h5"
        run([args.torus_image_exe, *torus_image_args(torus_path)], workdir)
        check_torus_golden(torus_path)
        if args.torus_parameter_file:
            torus_recommended_path = workdir / "torus_recommended_contract.h5"
            run(
                [
                    args.torus_image_exe,
                    f"--parameter_file={args.torus_parameter_file}",
                    # This archived numerical reference uses the W output basis.
                    # Override the N default explicitly; do not alter the reference.
                    "--evpa_0=W",
                    "--nx=8",
                    "--ny=8",
                    "--dsource=8100",
                    f"--output={torus_recommended_path}",
                    "--parameter_output=none",
                    "--timing=0",
                ],
                workdir,
            )
            check_torus_golden(torus_recommended_path)
            compare_single_frequency_images(torus_path, torus_recommended_path)

    iharm_image_path = None
    if args.iharm_image_exe:
        check_version_output(args.iharm_image_exe, args.expected_version, workdir)
        iharm_image_path = check_synthetic_iharm_contract(
            args.iharm_image_exe,
            workdir,
            args.trace_exe,
            args.analysis_compare_script,
            args.csv_output_enabled,
            compiled_grmhd_coordinates(args.compiled_grmhd_coordinates),
        )
    if args.kharma_image_exe:
        check_version_output(args.kharma_image_exe, args.expected_version, workdir)
        kharma_image_path = check_synthetic_kharma_contract(
            args.kharma_image_exe,
            workdir,
            args.trace_exe,
            args.analysis_compare_script,
            args.csv_output_enabled,
        )
        if iharm_image_path is not None:
            compare_single_frequency_images(iharm_image_path, kharma_image_path)
            compare_analysis_maps(iharm_image_path, kharma_image_path)
            for mode in (
                "spherical_ks_primitives",
                "spherical_ks_precomputed",
            ):
                iharm_resampled = (
                    workdir / f"synthetic_iharm_resample_{mode}.h5"
                )
                kharma_resampled = (
                    workdir / f"synthetic_kharma_resample_{mode}.h5"
                )
                compare_single_frequency_images(
                    iharm_resampled, kharma_resampled
                )
                compare_analysis_maps(iharm_resampled, kharma_resampled)
    if args.athenak_image_exe:
        check_version_output(args.athenak_image_exe, args.expected_version, workdir)
        check_synthetic_athenak_contract(
            args.athenak_image_exe,
            workdir,
            args.trace_exe,
            args.analysis_compare_script,
            args.csv_output_enabled,
        )
    if args.bhac_image_exe:
        check_version_output(args.bhac_image_exe, args.expected_version, workdir)
        check_synthetic_bhac_contract(
            args.bhac_image_exe,
            workdir,
            args.trace_exe,
            args.analysis_compare_script,
            args.csv_output_enabled,
        )
    if args.hamr_image_exe:
        check_version_output(args.hamr_image_exe, args.expected_version, workdir)
        check_synthetic_hamr_contract(
            args.hamr_image_exe,
            workdir,
            args.trace_exe,
            args.analysis_compare_script,
            args.csv_output_enabled,
        )

    roundtrip_same_path = workdir / "riaf_contract_roundtrip_same.h5"
    run(
        [
            args.image_exe,
            f"--parameter_file={params_path}",
            f"--output={roundtrip_same_path}",
            "--parameter_output=none",
        ],
        workdir,
    )
    check_image_contract(roundtrip_same_path, expect_golden=True)
    compare_single_frequency_images(image_path, roundtrip_same_path)
    if args.compare_script:
        comparison_json = workdir / "riaf_contract_comparison.json"
        comparison = run(
            [
                sys.executable,
                args.compare_script,
                str(image_path),
                str(roundtrip_same_path),
                "--rtol-l1=1e-12",
                "--rtol-linf-peak=1e-12",
                "--compare-diagnostics",
                "--require-exact-discrete-diagnostics",
                f"--json-output={comparison_json}",
            ],
            workdir,
        )
        assert "worst_rel_L1=0.00000000000000000e+00" in comparison.stdout
        comparison_payload = json.loads(comparison_json.read_text(encoding="utf-8"))
        assert comparison_payload["diagnostics"]["reason"]["mismatched_values"] == 0
        assert comparison_payload["grid_comparison"]["grid_x"]["max_absolute_difference"] == 0.0

        rounded_grid_path = workdir / "riaf_contract_rounded_grid.h5"
        shutil.copyfile(roundtrip_same_path, rounded_grid_path)
        with h5py.File(rounded_grid_path, "r+") as h5:
            grid = h5["grid/x"]
            grid[0] = np.nextafter(grid[0], math.inf)
        rounded_comparison = run(
            [
                sys.executable,
                args.compare_script,
                str(image_path),
                str(rounded_grid_path),
                "--compare-diagnostics",
                "--require-exact-discrete-diagnostics",
                "--rtol-l1=1e-12",
                "--rtol-linf-peak=1e-12",
            ],
            workdir,
        )
        assert "grid_x_max_abs=" in rounded_comparison.stdout
        with h5py.File(rounded_grid_path, "r+") as h5:
            h5["grid/x"][0] += 1.0e-6
        assert_failed_with(
            [
                sys.executable,
                args.compare_script,
                str(image_path),
                str(rounded_grid_path),
            ],
            workdir,
            "grid mismatch in grid_x",
        )

    roundtrip_path = workdir / "riaf_contract_roundtrip.h5"
    run(
        [
            args.image_exe,
            f"--parameter_file={params_path}",
            f"--output={roundtrip_path}",
            "--parameter_output=none",
            "--nx=4",
            "--ny=4",
        ],
        workdir,
    )
    check_image_contract(roundtrip_path)

    multifreq_path = workdir / "riaf_contract_multifrequency.h5"
    multifreq_cmd = [
        args.image_exe,
        *image_args(multifreq_path),
        "--nx=4",
        "--ny=4",
        "--freq_list=100000000000,230000000000",
    ]
    multifreq = subprocess.run(multifreq_cmd, cwd=workdir, text=True, capture_output=True)
    if multifreq.returncode == 0:
        check_multifrequency_image_contract(multifreq_path)
    elif "multi-frequency shared-geometry transport is disabled" in (multifreq.stderr + multifreq.stdout):
        print("multi-frequency image contract skipped: KPOLARIS_MAX_FREQUENCIES=1")
    else:
        print(multifreq.stdout)
        print(multifreq.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(multifreq_cmd)}")

    # Analysis mode intentionally evaluates each frequency sequentially when
    # the compiled shared-geometry frequency capacity is one.  This keeps the
    # physical diagnostics available in every supported build.
    analysis_multifreq_path = workdir / "riaf_contract_multifrequency_analysis.h5"
    run(
        [
            args.image_exe,
            *image_args(analysis_multifreq_path),
            "--nx=2",
            "--ny=2",
            "--freq_list=100000000000,230000000000",
            "--analysis_mode=1",
            "--analysis_radial_bins=7",
        ],
        workdir,
    )
    check_multifrequency_image_contract(
        analysis_multifreq_path, expected_shape=(2, 2)
    )
    for frequency_index in range(2):
        check_analysis_contract(
            analysis_multifreq_path,
            analysis_path=f"frame_0/freq_{frequency_index}/analysis",
            image_path=f"frame_0/freq_{frequency_index}",
        )
    with h5py.File(analysis_multifreq_path, "r") as h5:
        assert "frame_0/analysis" not in h5
        assert text_attr(h5["header"].attrs["analysis_layout"]) == (
            "/frame_0/freq_N/analysis"
        )

    plot_path = workdir / "riaf_contract.png"
    plot = run([sys.executable, args.plot_script, str(image_path), "--output", str(plot_path)], workdir)
    assert "Flux [Jy]:" in plot.stdout
    assert_nonempty_file(plot_path)

    if args.analysis_script:
        analysis_path = workdir / "riaf_contract_analysis.h5"
        run([args.image_exe, *image_args(analysis_path), "--analysis_mode=1"], workdir)
        check_analysis_contract(analysis_path)
        analysis_plot_path = workdir / "riaf_contract_analysis.png"
        run([sys.executable, args.analysis_script, str(analysis_path), "--output", str(analysis_plot_path)], workdir)
        assert_nonempty_file(analysis_plot_path)
        analysis_multifreq_plot_path = workdir / "riaf_contract_analysis_multifrequency.png"
        run(
            [
                sys.executable,
                args.analysis_script,
                str(analysis_multifreq_path),
                "--freq-index=1",
                "--output",
                str(analysis_multifreq_plot_path),
            ],
            workdir,
        )
        assert_nonempty_file(analysis_multifreq_plot_path)

        if args.trace_exe and args.analysis_compare_script:
            analysis_trace_path = workdir / "riaf_contract_analysis_trace.h5"
            run(
                [
                    args.trace_exe,
                    *image_args(analysis_trace_path),
                    "--trace_mode=image",
                    "--trace_precision=double",
                    "--trace_layout=ragged",
                    "--trace_compression=0",
                    "--trace_stride=1",
                    "--max_trace_samples=4096",
                    "--trace_fields=lambda,coords,plasma,coeffs",
                    f"--output={analysis_trace_path}",
                ],
                workdir,
            )
            with h5py.File(analysis_trace_path, "r") as h5:
                for name in ("beta", "sigma"):
                    assert name in h5["trace"]
                    assert h5[f"trace/{name}"].dtype == np.dtype("float64")
                radiation = h5["parameters/radiation"].attrs
                assert text_attr(radiation["model"]) == "riaf"
                assert int(radiation["emission_type"]) == 1
                assert text_attr(radiation["emission_fit"]) == "symphony_pandya_thermal"
                assert float(radiation["riaf_ne_unit"]) == 37228296.05086979
                spacetime = h5["parameters/spacetime"].attrs
                assert text_attr(spacetime["coordinate"]) == "boyer_lindquist"
                assert float(spacetime["spin"]) == 0.11272767423795238
            oracle_json = workdir / "riaf_contract_analysis_trace.json"
            oracle = run(
                [
                    sys.executable,
                    args.analysis_compare_script,
                    str(analysis_path),
                    str(analysis_trace_path),
                    f"--json-output={oracle_json}",
                ],
                workdir,
            )
            assert "analysis_trace_comparison=pass" in oracle.stdout
            oracle_payload = json.loads(oracle_json.read_text(encoding="utf-8"))
            assert oracle_payload["passed"] is True
            assert oracle_payload["failed_fields"] == []
            assert len(oracle_payload["fields"]) == 23
            assert max(
                float(row.get("l1_relative", 0.0))
                for row in oracle_payload["fields"].values()
            ) < 3.0e-15

            corrupted_trace_path = workdir / "riaf_contract_analysis_trace_corrupted.h5"
            shutil.copyfile(analysis_trace_path, corrupted_trace_path)
            with h5py.File(corrupted_trace_path, "r+") as h5:
                emissivity = h5["trace/jI"]
                values = emissivity[...]
                index = int(np.argmax(np.abs(values)))
                assert values[index] != 0.0
                values[index] *= 2.0
                emissivity[...] = values
            assert_failed_with(
                [
                    sys.executable,
                    args.analysis_compare_script,
                    str(analysis_path),
                    str(corrupted_trace_path),
                ],
                workdir,
                "failed_fields=",
            )

            mismatched_metadata_path = workdir / "riaf_contract_analysis_trace_wrong_metadata.h5"
            shutil.copyfile(analysis_trace_path, mismatched_metadata_path)
            with h5py.File(mismatched_metadata_path, "r+") as h5:
                attrs = h5["parameters/radiation"].attrs
                attrs.modify("riaf_ne_unit", float(attrs["riaf_ne_unit"]) * 2.0)
            assert_failed_with(
                [
                    sys.executable,
                    args.analysis_compare_script,
                    str(analysis_path),
                    str(mismatched_metadata_path),
                ],
                workdir,
                "image and trace parameter 'riaf_ne_unit' differ",
            )

    if args.trace_exe:
        check_trace_contract(
            args.trace_exe,
            workdir,
            Path(args.trace_diagnostics_script) if args.trace_diagnostics_script else None,
            Path(args.trace_bins_script) if args.trace_bins_script else None,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
