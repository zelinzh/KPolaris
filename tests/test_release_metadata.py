#!/usr/bin/env python3
"""Exercise metadata in unreleased checkouts, tagged releases, and archives."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
CHECKER = SOURCE / "scripts" / "check_release_metadata.py"


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kpolaris-metadata-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        fixtures = {
            "CMakeLists.txt": "project(KPolaris VERSION 0.1.0 LANGUAGES CXX)\n",
            "CITATION.cff": "cff-version: 1.2.0\nversion: 0.1.0\n",
            "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n",
            "cmake/kpolaris_version.hpp.in": (
                SOURCE / "cmake/kpolaris_version.hpp.in"
            ).read_text(),
        }
        for name, content in fixtures.items():
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)

    def dates(self):
        citation = self.source / "CITATION.cff"
        citation.write_text(citation.read_text() + "\ndate-released: 2000-01-01\n")
        changelog = self.source / "CHANGELOG.md"
        changelog.write_text(changelog.read_text().replace(
            "## [Unreleased]", "## [0.1.0] - 2000-01-01", 1))

    def git(self, *args, directory=None):
        return subprocess.run(
            ["git", "-c", "user.name=Metadata Test",
             "-c", "user.email=metadata@example.invalid",
             "-c", "core.hooksPath=/dev/null", *args],
            cwd=directory or self.source, capture_output=True, text=True, check=True,
        )

    def repository(self, directory=None, tagged=False):
        if shutil.which("git") is None:
            self.skipTest("checkout/tag cases require git")
        self.git("init", directory=directory)
        self.git("commit", "--allow-empty", "-m", "Metadata fixture", directory=directory)
        if tagged:
            self.git("tag", "v0.1.0", directory=directory)

    def check(self, success, marker):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--source-dir", str(self.source),
             "--expected-version", "0.1.0"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        self.assertIn(marker, result.stdout + result.stderr)

    def test_unreleased_archive(self):
        self.check(True, "unreleased archive")

    def test_dated_archive(self):
        self.dates()
        self.check(True, "dated archive (Git tag not verified)")

    def test_optional_manuscript_version_is_checked(self):
        bibliography = self.source / "paper" / "references.bib"
        bibliography.parent.mkdir()
        bibliography.write_text("@misc{KPolarisSoftware,\n  version = {9.9.9}\n}\n")
        self.check(False, "version mismatch")

    def test_partial_archive_metadata_is_rejected(self):
        citation = self.source / "CITATION.cff"
        citation.write_text(citation.read_text() + "\ndate-released: 2000-01-01\n")
        self.check(False, "both be present or both be absent")

    def test_unreleased_checkout(self):
        self.repository()
        self.check(True, "state=unreleased")

    def test_untagged_dated_checkout_is_rejected(self):
        self.repository()
        self.dates()
        self.check(False, "untagged current version")

    def test_tagged_release(self):
        self.repository(tagged=True)
        self.dates()
        self.check(True, "state=tagged")

    def test_tag_without_release_metadata_is_rejected(self):
        self.repository(tagged=True)
        self.check(False, "lacks a CITATION.cff release date")

    def test_archive_ignores_unrelated_parent_repository(self):
        self.repository(directory=self.root)
        self.dates()
        self.check(True, "dated archive (Git tag not verified)")


if __name__ == "__main__":
    unittest.main()
