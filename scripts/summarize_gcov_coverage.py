#!/usr/bin/env python3
"""Summarize project-only GCC gcov JSON coverage without third-party tools."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import subprocess
import tempfile


@dataclass
class BranchCoverage:
    count: int = 0
    is_throw: bool = False


@dataclass
class FileCoverage:
    line_counts: dict[int, int] = field(default_factory=dict)
    branch_counts: dict[tuple[int, int, bool, bool], BranchCoverage] = field(
        default_factory=dict
    )

    def merge(self, payload: dict[str, object]) -> None:
        for line in payload.get("lines", []):
            line_number = int(line["line_number"])
            count = int(line.get("count", 0))
            self.line_counts[line_number] = (
                self.line_counts.get(line_number, 0) + count
            )
            for index, branch in enumerate(line.get("branches", [])):
                is_throw = bool(branch.get("throw", False))
                fallthrough = bool(branch.get("fallthrough", False))
                key = (line_number, index, is_throw, fallthrough)
                entry = self.branch_counts.setdefault(
                    key, BranchCoverage(is_throw=is_throw)
                )
                entry.count += int(branch.get("count", 0))


def ratio(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def metrics(coverage: FileCoverage) -> dict[str, int | float]:
    lines_total = len(coverage.line_counts)
    lines_covered = sum(count > 0 for count in coverage.line_counts.values())
    raw_branches = list(coverage.branch_counts.values())
    branches = [branch for branch in raw_branches if not branch.is_throw]
    branches_covered = sum(branch.count > 0 for branch in branches)
    raw_branches_covered = sum(branch.count > 0 for branch in raw_branches)
    return {
        "lines_covered": lines_covered,
        "lines_total": lines_total,
        "line_percent": ratio(lines_covered, lines_total),
        "branches_covered": branches_covered,
        "branches_total": len(branches),
        "branch_percent": ratio(branches_covered, len(branches)),
        "raw_branches_covered": raw_branches_covered,
        "raw_branches_total": len(raw_branches),
        "raw_branch_percent": ratio(raw_branches_covered, len(raw_branches)),
    }


def merge_totals(files: dict[str, FileCoverage]) -> FileCoverage:
    total = FileCoverage()
    line_offset = 0
    for coverage in files.values():
        # Synthetic offsets make per-file line and branch keys unique while
        # retaining the same metric implementation used for individual files.
        for line_number, count in coverage.line_counts.items():
            total.line_counts[line_offset + line_number] = count
        for (line_number, index, is_throw, fallthrough), branch in (
            coverage.branch_counts.items()
        ):
            total.branch_counts[
                (line_offset + line_number, index, is_throw, fallthrough)
            ] = BranchCoverage(count=branch.count, is_throw=is_throw)
        line_offset += max(coverage.line_counts, default=0) + 1
    return total


def project_relative_path(
    filename: str, source_root: Path, source_prefixes: tuple[Path, ...]
) -> str | None:
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return None
    if not any(relative == prefix or prefix in relative.parents for prefix in source_prefixes):
        return None
    return relative.as_posix()


def collect_coverage(
    build_dir: Path,
    source_root: Path,
    source_prefixes: tuple[Path, ...],
    gcov: str,
) -> tuple[dict[str, FileCoverage], int]:
    files: dict[str, FileCoverage] = {}
    object_files = sorted(build_dir.glob("CMakeFiles/**/*.gcno"))
    if not object_files:
        raise RuntimeError(f"no project .gcno files found below {build_dir / 'CMakeFiles'}")

    for object_file in object_files:
        with tempfile.TemporaryDirectory(prefix="kpolaris-gcov-") as temporary:
            result = subprocess.run(
                [
                    gcov,
                    "--json-format",
                    "--branch-probabilities",
                    "--branch-counts",
                    str(object_file),
                ],
                cwd=temporary,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"gcov failed for {object_file}:\n{result.stdout}\n{result.stderr}"
                )
            reports = list(Path(temporary).glob("*.gcov.json.gz"))
            if len(reports) != 1:
                raise RuntimeError(
                    f"expected one gcov JSON report for {object_file}, got {len(reports)}"
                )
            with gzip.open(reports[0], "rt", encoding="utf-8") as stream:
                report = json.load(stream)
            for payload in report.get("files", []):
                relative = project_relative_path(
                    str(payload["file"]), source_root, source_prefixes
                )
                if relative is None:
                    continue
                files.setdefault(relative, FileCoverage()).merge(payload)
    return files, len(object_files)


def percent_text(value: float) -> str:
    return f"{value:.2f}%"


def markdown_report(payload: dict[str, object]) -> str:
    total = payload["totals"]
    lines = [
        "# KPolaris CPU coverage",
        "",
        (
            f"Project-only line coverage: **{total['lines_covered']}/"
            f"{total['lines_total']} ({percent_text(total['line_percent'])})**. "
            f"Branch coverage excluding compiler exception edges: "
            f"**{total['branches_covered']}/{total['branches_total']} "
            f"({percent_text(total['branch_percent'])})**. Raw gcov branch "
            f"coverage: **{total['raw_branches_covered']}/"
            f"{total['raw_branches_total']} "
            f"({percent_text(total['raw_branch_percent'])})**."
        ),
        "",
        "| File | Lines | Branches (no exception edges) | Raw branches |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["files"]:
        lines.append(
            f"| `{row['path']}` | {row['lines_covered']}/{row['lines_total']} "
            f"({percent_text(row['line_percent'])}) | "
            f"{row['branches_covered']}/{row['branches_total']} "
            f"({percent_text(row['branch_percent'])}) | "
            f"{row['raw_branches_covered']}/{row['raw_branches_total']} "
            f"({percent_text(row['raw_branch_percent'])}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-prefix",
        action="append",
        default=None,
        help="project-relative source tree to include; repeat as needed",
    )
    parser.add_argument("--gcov", default="gcov")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--min-line-percent", type=float, default=0.0)
    parser.add_argument(
        "--min-branch-percent",
        type=float,
        default=0.0,
        help="minimum branch coverage after compiler exception edges are excluded",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    source_root = args.source_root.resolve()
    source_prefixes = tuple(
        Path(value) for value in (args.source_prefix or ["src", "include", "tools"])
    )
    files, object_files = collect_coverage(
        build_dir, source_root, source_prefixes, args.gcov
    )
    if not files:
        raise RuntimeError("gcov reports contained no files in the selected source trees")

    rows: list[dict[str, object]] = []
    for path, coverage in sorted(files.items()):
        row: dict[str, object] = {"path": path}
        row.update(metrics(coverage))
        row["uncovered_lines"] = sorted(
            line for line, count in coverage.line_counts.items() if count == 0
        )
        rows.append(row)
    total = metrics(merge_totals(files))
    version = subprocess.run(
        [args.gcov, "--version"], text=True, capture_output=True, check=True
    ).stdout.splitlines()[0]
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "build_dir": str(build_dir),
        "source_prefixes": [path.as_posix() for path in source_prefixes],
        "gcov_version": version,
        "object_files": object_files,
        "totals": total,
        "files": rows,
    }

    rendered = markdown_report(payload)
    print(rendered, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered, encoding="utf-8")
    failures: list[str] = []
    if float(total["line_percent"]) < args.min_line_percent:
        failures.append(
            f"line coverage {total['line_percent']:.2f}% is below "
            f"{args.min_line_percent:.2f}%"
        )
    if float(total["branch_percent"]) < args.min_branch_percent:
        failures.append(
            f"branch coverage {total['branch_percent']:.2f}% is below "
            f"{args.min_branch_percent:.2f}%"
        )
    if failures:
        for failure in failures:
            print(f"coverage threshold failure: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
