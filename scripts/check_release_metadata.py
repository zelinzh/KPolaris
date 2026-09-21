#!/usr/bin/env python3
"""Check that KPolaris version and release-state metadata agree."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def require_match(pattern: str, text: str, label: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    if match is None:
        raise ValueError(f"could not find {label}")
    return match.group(1)


def git_tags(source_dir: Path) -> set[str] | None:
    # An extracted archive may sit inside an unrelated Git checkout. Its parent
    # tags cannot establish the release state of this source tree.
    if not (source_dir / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=source_dir,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    cmake_text = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    citation_text = (source_dir / "CITATION.cff").read_text(encoding="utf-8")
    changelog_text = (source_dir / "CHANGELOG.md").read_text(encoding="utf-8")
    version_template = (source_dir / "cmake" / "kpolaris_version.hpp.in").read_text(
        encoding="utf-8"
    )

    cmake_version = require_match(
        r"project\s*\(\s*KPolaris\s+VERSION\s+([^\s\)]+)",
        cmake_text,
        "CMake project version",
        re.IGNORECASE,
    )
    citation_version = require_match(
        r"(?m)^version:\s*[\"']?([^\s\"']+)", citation_text, "CITATION.cff version"
    )
    versions = {
        "expected": args.expected_version,
        "CMake": cmake_version,
        "CITATION.cff": citation_version,
    }
    # The manuscript is an optional companion, not a source-package dependency.
    bibliography = source_dir / "paper" / "references.bib"
    if bibliography.is_file():
        software_entry = require_match(
            r"(@misc\{KPolarisSoftware,.*?\n\})",
            bibliography.read_text(encoding="utf-8"),
            "KPolarisSoftware bibliography entry",
            re.DOTALL,
        )
        versions["bibliography"] = require_match(
            r"(?m)^\s*version\s*=\s*\{([^}]+)\}",
            software_entry,
            "KPolarisSoftware bibliography version",
        )
    if len(set(versions.values())) != 1:
        details = ", ".join(f"{key}={value}" for key, value in versions.items())
        raise ValueError(f"version mismatch: {details}")

    for token in (
        "@PROJECT_VERSION@",
        "@KPOLARIS_BUILD_SOURCE_REVISION@",
        "@KPOLARIS_BUILD_SOURCE_DIRTY@",
        "@CMAKE_CXX_COMPILER_ID@",
        "@CMAKE_CXX_COMPILER_VERSION@",
        "@CMAKE_BUILD_TYPE@",
    ):
        if token not in version_template:
            raise ValueError(f"generated version header is missing {token}")

    tags = git_tags(source_dir)
    matching_tag = tags is not None and bool(
        {args.expected_version, f"v{args.expected_version}"} & tags
    )
    has_release_date = re.search(r"(?m)^date-released:\s*\S+", citation_text) is not None
    has_dated_changelog = (
        re.search(
            rf"(?m)^## \[(?:v)?{re.escape(args.expected_version)}\]\s+-\s+\d{{4}}-\d{{2}}-\d{{2}}\s*$",
            changelog_text,
        )
        is not None
    )

    if has_release_date != has_dated_changelog:
        raise ValueError(
            "CITATION.cff release date and matching dated CHANGELOG entry must "
            "either both be present or both be absent"
        )

    if tags is None:
        if not has_release_date and "## [Unreleased]" not in changelog_text:
            raise ValueError("undated source archive requires an [Unreleased] changelog section")
    elif matching_tag:
        if not has_release_date or not has_dated_changelog:
            raise ValueError(
                "the current version has a Git tag but lacks a CITATION.cff release "
                "date or matching dated CHANGELOG entry"
            )
    elif has_release_date or has_dated_changelog:
        raise ValueError(
            "untagged current version must not claim a release date in CITATION.cff "
            "or CHANGELOG.md"
        )
    elif "## [Unreleased]" not in changelog_text:
        raise ValueError("untagged current version requires an [Unreleased] changelog section")

    tag_state = (
        "dated archive (Git tag not verified)" if has_release_date
        else "unreleased archive (no Git metadata)"
    ) if tags is None else (
        "tagged" if matching_tag else "unreleased"
    )
    print(f"release metadata OK: version={args.expected_version}, state={tag_state}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"release metadata error: {exc}", file=sys.stderr)
        raise SystemExit(1)
