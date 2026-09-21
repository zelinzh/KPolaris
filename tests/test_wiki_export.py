#!/usr/bin/env python3
"""Check that the manual remains navigable after conversion to a GitHub Wiki."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("wiki", Path(__file__).resolve().parents[1] / "scripts/export_wiki.py")
wiki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiki)


class WikiExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        (self.root / "docs").mkdir(parents=True)
        (self.root / "README.md").write_text("# Code\n")
        (self.root / "docs/README.md").write_text(
            "# Manual\n[Install](quickstart.md#cpu)\n[Code](../README.md)\n"
            "[External](https://example.org)\n```text\n[Literal](missing.md)\n```\n")
        (self.root / "docs/quickstart.md").write_text("# Install\n[Home](README.md)\n")
        self.output = Path(self.temp.name) / "wiki"

    def test_pages_and_links(self):
        self.assertEqual(wiki.export(self.root, self.output, "example/KPolaris"), 4)
        text = (self.output / "Home.md").read_text()
        self.assertIn("[Install](https://github.com/example/KPolaris/wiki/Installation#cpu)", text)
        self.assertIn("https://github.com/example/KPolaris/blob/main/README.md", text)
        self.assertIn("[Literal](missing.md)", text)
        self.assertIn("[Home](https://github.com/example/KPolaris/wiki/Home)", (self.output / "Installation.md").read_text())
        self.assertTrue((self.output / "_Sidebar.md").is_file())

    def test_chinese_pages_and_cross_language_links(self):
        (self.root / "docs/zh").mkdir()
        (self.root / "docs/zh/README.md").write_text("# 中文手册\n[English](../README.md)\n")
        (self.root / "docs/README.md").write_text("# Manual\n[中文](zh/README.md)\n")
        self.assertEqual(wiki.export(self.root, self.output, "example/KPolaris"), 5)
        self.assertIn("/wiki/ZH-Home)", (self.output / "Home.md").read_text())
        self.assertIn("/wiki/Home)", (self.output / "ZH-Home.md").read_text())
        self.assertIn("[中文首页](https://github.com/example/KPolaris/wiki/ZH-Home)",
                      (self.output / "_Sidebar.md").read_text())
        self.assertIn("/wiki/ZH-Home)", (self.output / "_Footer.md").read_text())

    def test_final_repository_and_pinned_source_links(self):
        wiki.export(self.root, self.output, "new-owner/NewRepo", "v1.2.3")
        text = (self.output / "Home.md").read_text()
        self.assertIn("https://github.com/new-owner/NewRepo/blob/v1.2.3/README.md", text)
        self.assertIn("https://github.com/new-owner/NewRepo/wiki/Installation#cpu", text)
        footer = (self.output / "_Footer.md").read_text()
        self.assertIn("/tree/v1.2.3/docs", footer)
        self.assertIn("Documentation source: `v1.2.3`", footer)

    def test_additional_page_is_not_missing_from_navigation(self):
        (self.root / "docs/extra.md").write_text("# Extra guide\n")
        wiki.export(self.root, self.output, "example/KPolaris")
        self.assertIn("/wiki/extra)", (self.output / "_Sidebar.md").read_text())

    def test_missing_link_fails_before_writing(self):
        (self.root / "docs/quickstart.md").write_text("[Absent](missing.md)")
        with self.assertRaisesRegex(ValueError, "missing or unsafe"):
            wiki.export(self.root, self.output, "example/KPolaris")
        self.assertFalse(self.output.exists())

    def test_preserves_existing_output(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            wiki.export(self.root, self.output, "example/KPolaris")

    def test_rejects_escaping_local_link(self):
        (self.root / "docs/quickstart.md").write_text("[Outside](../../outside.md)")
        (self.root.parent / "outside.md").write_text("private")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            wiki.export(self.root, self.output, "example/KPolaris")


if __name__ == "__main__":
    unittest.main()
