#!/usr/bin/env python3
"""
Find files in a built static site (MkDocs output) that no HTML or CSS file
references. Intended for `exploration-archive/<ver>/site/site/` roots — raw
figures, CSVs, or artifacts copied into docs/ sometimes end up in the build
without ever being linked.

Usage:
    python scripts/find_site_orphans.py <site_root>              # list orphans
    python scripts/find_site_orphans.py <site_root> --summary    # summarize bytes
    python scripts/find_site_orphans.py <site_root> --write      # write .gitignore

The --write mode creates `<site_root>/.gitignore` with one orphan per line,
using paths relative to the site root. Combined with the repo's whitelist
gitignore (`!exploration-archive/*/site/site/**`), nested re-ignore rules
pull orphans back out of the tracked set while leaving the files on disk.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urlparse

# Paths we never treat as orphans even if no HTML/CSS directly references
# them. These are dynamically loaded by MkDocs Material's JS or are protocol
# files the host (GitHub Pages) expects at fixed paths.
WHITELIST_FILES = {
    "index.html",
    "404.html",
    "sitemap.xml",
    "sitemap.xml.gz",
    ".gitignore",
}
WHITELIST_DIR_PREFIXES = (
    "search/",                 # Material search index + workers, fetched dynamically
    "assets/javascripts/",     # Material JS bundle + lunr language packs
    "assets/stylesheets/",     # palette + main CSS self-reference via url(), keep all
)

HTML_REF_ATTRS = {"href", "src", "data-src", "poster"}
CSS_URL_RE = re.compile(r"""url\(\s*['"]?([^'")\s]+)['"]?\s*\)""")
NON_LOCAL_SCHEMES = {"http", "https", "mailto", "data", "tel", "javascript", "ftp"}


class RefExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in HTML_REF_ATTRS and value:
                self.refs.add(value)
            elif name == "srcset" and value:
                # Pick the URL before any descriptor ("foo.png 2x, bar.png 3x")
                for part in value.split(","):
                    url = part.strip().split()[0] if part.strip() else ""
                    if url:
                        self.refs.add(url)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_html_refs(path: Path) -> set[str]:
    parser = RefExtractor()
    try:
        parser.feed(read_text(path))
    except Exception:
        return set()
    return parser.refs


def extract_css_refs(path: Path) -> set[str]:
    try:
        return set(CSS_URL_RE.findall(read_text(path)))
    except Exception:
        return set()


def resolve_ref(source_file: Path, ref: str, site_root: Path) -> Path | None:
    """Resolve an href/src reference relative to the file that contained it,
    returning a site-root-relative Path, or None if the ref is external/invalid."""
    ref = urldefrag(ref).url
    ref = ref.split("?", 1)[0]
    if not ref:
        return None
    parsed = urlparse(ref)
    if parsed.scheme in NON_LOCAL_SCHEMES:
        return None
    if ref.startswith(("#", "//")):
        return None
    if ref.startswith("/"):
        target = (site_root / ref.lstrip("/")).resolve()
    else:
        target = (source_file.parent / ref).resolve()
    try:
        return target.relative_to(site_root.resolve())
    except ValueError:
        return None


def is_whitelisted(rel: Path) -> bool:
    as_str = rel.as_posix()
    if rel.name in WHITELIST_FILES:
        return True
    return any(as_str.startswith(p) for p in WHITELIST_DIR_PREFIXES)


def find_orphans(site_root: Path) -> list[Path]:
    site_root = site_root.resolve()
    all_files: set[Path] = {
        p.resolve().relative_to(site_root)
        for p in site_root.rglob("*")
        if p.is_file()
    }

    referenced: set[Path] = set()
    for f in all_files:
        if is_whitelisted(f):
            referenced.add(f)

    for f in list(all_files):
        full = site_root / f
        if f.suffix.lower() in {".html", ".htm"}:
            refs = extract_html_refs(full)
        elif f.suffix.lower() == ".css":
            refs = extract_css_refs(full)
        else:
            continue
        for ref in refs:
            resolved = resolve_ref(full, ref, site_root)
            if resolved is None:
                continue
            target = site_root / resolved
            if target.is_dir():
                # Directory URLs (MkDocs use_directory_urls=true): ref "foo/" ==> foo/index.html
                referenced.add(resolved / "index.html")
            else:
                referenced.add(resolved)

    orphans = sorted(all_files - referenced)
    return orphans


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.1f}{unit}" if False else f"{n:.1f}{unit}" if n >= 1024 or unit != "B" else f"{n}{unit}"
        n /= 1024
    return str(n)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}T"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site_root", help="Path to built site root (contains index.html)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true", help="Print byte-count summary instead of paths")
    mode.add_argument("--write", action="store_true", help="Write orphans to <site_root>/.gitignore")
    args = ap.parse_args(argv)

    site_root = Path(args.site_root)
    if not site_root.is_dir() or not (site_root / "index.html").exists():
        print(f"error: {site_root} is not a built site root (missing index.html)", file=sys.stderr)
        return 2

    orphans = find_orphans(site_root)

    if args.summary:
        total = sum((site_root / o).stat().st_size for o in orphans)
        by_top: dict[str, int] = {}
        for o in orphans:
            top = o.parts[0] if len(o.parts) > 1 else "(root)"
            by_top[top] = by_top.get(top, 0) + (site_root / o).stat().st_size
        print(f"{len(orphans)} orphan files, total {fmt_bytes(total)}")
        for top, size in sorted(by_top.items(), key=lambda kv: -kv[1]):
            print(f"  {fmt_bytes(size):>8}  {top}/")
        return 0

    if args.write:
        gitignore = site_root / ".gitignore"
        if not orphans:
            if gitignore.exists():
                gitignore.unlink()
            print(f"{site_root}: no orphans; removed {gitignore}" if gitignore.exists() else f"{site_root}: no orphans")
            return 0
        body = (
            "# Auto-generated by scripts/find_site_orphans.py --write.\n"
            "# These files exist in the built site but are not referenced by any HTML or CSS.\n"
            "# Do not commit. Regenerate by rerunning the finder after each site rebuild.\n"
        )
        body += "\n".join(o.as_posix() for o in orphans) + "\n"
        gitignore.write_text(body)
        total = sum((site_root / o).stat().st_size for o in orphans)
        print(f"{site_root}: {len(orphans)} orphans ({fmt_bytes(total)}) written to {gitignore}")
        return 0

    for o in orphans:
        print(o.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
