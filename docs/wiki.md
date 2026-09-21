# Documentation and GitHub Wiki

The `docs/` directory is the versioned user manual. Read the copy shipped with
your checkout when reproducing a calculation. The README covers the first
successful image; the manual covers input models, physical conventions,
diagnostics, slow light, plotting and output fields.

The [GitHub Wiki](https://github.com/zelinzh/KPolaris/wiki) presents the same manual
with a navigation sidebar. To regenerate its pages from a source checkout:

```bash
python3 scripts/export_wiki.py --repository=OWNER/KPolaris --output=outputs/wiki
```

Replace `OWNER` with the repository owner. The exporter creates Markdown pages,
rewrites links and checks local file targets. English pages form the main manual;
the separate Chinese pages use the `ZH-` filename prefix, with their own navigation
group and links between the language homepages. It does not contact GitHub or
change repository visibility. `--ref` can select a code revision for links back
to the repository. Use a new output directory for each export.

For a release, generate from the finalized checkout and pin repository-file
links to its tag with `--ref=YOUR_RELEASE_TAG`. The Wiki has separate Git history;
the versioned `docs/` in the source release remain the reference for that release.
When moving to a new repository, update hard-coded project URLs in the source
manual and commands first, then pass the final `--repository`. The exporter
rewrites relative links; it intentionally preserves literal code examples and
already-absolute URLs.

To update an enabled Wiki, initialize its Home page on GitHub, clone its separate
`KPolaris.wiki.git` repository, copy the generated pages into that checkout,
review the changes, then commit and push. Keep changes in `docs/` and regenerate
the Wiki so that the two presentations agree. Do not copy build or run directories.

After creating the first Home page on the final repository, the import sequence
is as follows (replace the repository URL and generated-directory path):

```bash
git clone https://github.com/OWNER/KPolaris.wiki.git wiki-checkout
cp /path/to/generated-wiki/*.md wiki-checkout/
git -C wiki-checkout diff --check
git -C wiki-checkout add -- '*.md'
git -C wiki-checkout diff --cached --stat
git -C wiki-checkout commit -m "Add KPolaris user manual"
git -C wiki-checkout push origin HEAD
```

Include `Home.md`, `_Sidebar.md`, `_Footer.md` and all English/`ZH-` pages.
Copy only the exported pages into the Wiki; keep release checklists and export
verification outside it. See [GitHub's Wiki editing guide](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages).
