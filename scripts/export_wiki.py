#!/usr/bin/env python3
"""Render the versioned user manual as GitHub Wiki pages; never upload files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit


PAGES = {
    "README.md": "Home",
    "quickstart.md": "Installation",
    "first_image.md": "First-image",
    "plotting.md": "Plotting",
    "cli.md": "Command-line-helper",
    "parameters.md": "Parameter-reference",
    "user_guide.md": "Image-workflows",
    "electron_distributions.md": "Electron-distributions",
    "diagnostics.md": "Physical-diagnostics",
    "slow_light.md": "Slow-light",
    "polarization_conventions.md": "Polarization-conventions",
    "hdf5_schema.md": "HDF5-format",
    "architecture.md": "Code-organization",
    "wiki.md": "Documentation",
}

SECTIONS = (
    ("Start here", (
        ("README.md", "Home"),
        ("quickstart.md", "Installation and first image"),
        ("first_image.md", "M87* example"),
        ("user_guide.md", "Image your GRMHD data"),
        ("plotting.md", "Plot and inspect results"),
    )),
    ("Scientific workflows", (
        ("slow_light.md", "Slow light and batches"),
        ("diagnostics.md", "Physical diagnostics"),
        ("polarization_conventions.md", "Camera and polarization conventions"),
        ("electron_distributions.md", "Electron distributions"),
    )),
    ("Reference", (
        ("parameters.md", "Solver parameters"),
        ("cli.md", "Command-line helper"),
        ("hdf5_schema.md", "HDF5 data format"),
        ("architecture.md", "Code organization"),
        ("wiki.md", "Documentation and Wiki"),
    )),
)

CHINESE_PAGES = (
    ("README.md", "中文首页"),
    ("user_guide.md", "安装、输入与成像"),
    ("first_image.md", "M87* 首次成图"),
    ("parameters.md", "参数设定"),
    ("slow_light.md", "慢光与批处理"),
    ("diagnostics.md", "物理诊断"),
    ("polarization_conventions.md", "方位与偏振约定"),
)


def wiki_url(repository: str, page: str) -> str:
    return f"https://github.com/{repository}/wiki/{quote(page, safe='-')}"


def render(text: str, source: Path, root: Path, pages: dict[Path, str], repository: str, ref: str) -> str:
    def link(match: re.Match) -> str:
        label, target = match.groups()
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or target.startswith("#"):
            return match.group(0)
        local = (source.parent / unquote(parts.path)).resolve()
        if not local.is_relative_to(root) or not local.is_file():
            raise ValueError(f"missing or unsafe documentation link in {source.name}: {target}")
        suffix = ("?" + parts.query if parts.query else "") + ("#" + parts.fragment if parts.fragment else "")
        if local in pages:
            destination = wiki_url(repository, pages[local])
        else:
            relative = quote(local.relative_to(root).as_posix(), safe="/")
            destination = f"https://github.com/{repository}/blob/{quote(ref, safe='')}/{relative}"
        return f"[{label}]({destination}{suffix})"

    # Examples are literal commands; rewrite links only in prose, not fenced code.
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            result.append(line)
        elif fence:
            result.append(line)
        else:
            result.append(re.sub(r"\[([^\]]*)\]\(([^)]+)\)", link, line))
    return "".join(result)


def export(root: Path, output: Path, repository: str, ref: str = "main") -> int:
    root, output = root.resolve(), output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be a GitHub owner/name")
    if not ref or any(ch.isspace() for ch in ref):
        raise ValueError("ref must be a nonempty Git revision without whitespace")
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    docs = root / "docs"
    if output.is_relative_to(docs):
        raise ValueError("Wiki output must not be inside docs")
    files = sorted(docs.glob("*.md")) + sorted((docs / "zh").glob("*.md"))
    if not files or not (docs / "README.md").is_file():
        raise ValueError("source must contain docs/README.md")
    pages = {
        p.resolve(): ("ZH-" if p.parent == docs / "zh" else "") + PAGES.get(p.name, p.stem)
        for p in files
    }
    if len(set(pages.values())) != len(files):
        raise ValueError("duplicate Wiki page names")
    rendered = {}
    for path in files:
        name = pages[path.resolve()]
        rendered[name + ".md"] = render(path.read_text(), path, root, pages, repository, ref)
    sidebar = ["**KPolaris user manual**"]
    listed = set()
    for section, entries in (*SECTIONS, ("中文文档", CHINESE_PAGES)):
        directory = docs / "zh" if section == "中文文档" else docs
        links = []
        for filename, label in entries:
            path = (directory / filename).resolve()
            if path in pages:
                links.append(f"- [{label}]({wiki_url(repository, pages[path])})\n")
                listed.add(path)
        if links:
            sidebar.append(f"**{section}**\n\n" + "".join(links).rstrip())
    extra = [path for path in pages if path not in listed]
    if extra:
        sidebar.append("**Additional pages**\n\n" + "\n".join(
            f"- [{pages[p].replace('-', ' ')}]({wiki_url(repository, pages[p])})" for p in extra))
    rendered["_Sidebar.md"] = "\n\n".join(sidebar) + "\n"
    rendered["_Footer.md"] = (
        f"[Source and versioned manual](https://github.com/{repository}/tree/{quote(ref, safe='')}/docs)"
        f" · [Citation](https://github.com/{repository}/blob/{quote(ref, safe='')}/README.md#citation)"
        f" · [English]({wiki_url(repository, 'Home')})"
        + (f" · [中文]({wiki_url(repository, 'ZH-Home')})" if (docs / "zh/README.md").resolve() in pages else "")
        + f"\n\nDocumentation source: `{ref}`. Maintained in the source repository's `docs/` directory.\n"
    )
    output.mkdir(parents=True)
    for name, content in rendered.items():
        (output / name).write_text(content)
    return len(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--repository", required=True, help="GitHub owner/name")
    parser.add_argument("--ref", default="main", help="code revision used for non-Wiki links")
    args = parser.parse_args()
    try:
        count = export(args.source_dir, args.output, args.repository, args.ref)
    except ValueError as error:
        parser.error(str(error))
    print(f"Wrote {count} Wiki pages to {args.output}; no upload performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
