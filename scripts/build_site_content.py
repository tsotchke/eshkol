#!/usr/bin/env python3
"""Project the documentation tree onto the website.

site/pages.json is the one declared list of documentation pages the website
publishes: each entry names a Markdown file, the slug it is published under,
the site view it belongs to (the /docs or the /tutorials page) and the sidebar
section it is listed in.  From that list this script generates, under
site/static/content/:

  <slug>.html       the page, rendered from its Markdown source by pandoc,
                    with links between published pages rewritten to stay on
                    the site and every other repository-relative link pointed
                    at the repository
  nav-<view>.html   the sidebar of each view, in list order
  pages.json        the index of what was published

The website (site/src/main.esk) loads nav-<view>.html into its sidebar, so a
page is added to the site by adding one entry to site/pages.json and rerunning
this script.  A listed file that does not exist is skipped with a warning when
its entry is marked "pending": true, and is published the first time the
script runs after the file lands.

Usage:
    scripts/build_site_content.py            # render with pandoc
    scripts/build_site_content.py --check    # no pandoc: verify that the list,
                                             # the fragments and the navigation
                                             # agree (exit 1 when they do not)
    scripts/build_site_content.py --self-test
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_PATH = ROOT / "site" / "pages.json"
CONTENT_DIR = ROOT / "site" / "static" / "content"
SCHEMA = "eshkol.site-pages.v1"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
# Names this script generates beside the page fragments.
INDEX_NAME = "pages.json"
NAV_PREFIX = "nav-"
# Slugs that would shadow a generated file or a graded repository README.
RESERVED_SLUGS = {"readme", "pages", "index"}
PANDOC_READER = "markdown+gfm_auto_identifiers"


class PageListError(Exception):
    """site/pages.json is malformed."""


def load_page_list(path: Path = PAGES_PATH, root: Path = ROOT) -> dict:
    """Read and validate the declared page list.

    Returns the document with two derived keys: "published" (entries whose
    file exists, in list order) and "absent" (entries whose file does not).
    Raises PageListError on a malformed list, a duplicate slug or file, an
    unknown view, a view default that is not a listed page of that view, or an
    absent file that is not marked pending.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PageListError(f"{path}: cannot read the page list: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise PageListError(f"{path}: schema must be {SCHEMA!r}")
    repository = doc.get("repository")
    branch = doc.get("branch")
    if not isinstance(repository, str) or not repository.startswith("https://github.com/"):
        raise PageListError(f"{path}: 'repository' must be a https://github.com/ URL")
    if not isinstance(branch, str) or not branch:
        raise PageListError(f"{path}: 'branch' must be a non-empty string")
    views = doc.get("views")
    pages = doc.get("pages")
    if not isinstance(views, dict) or not views:
        raise PageListError(f"{path}: 'views' must be a non-empty object")
    if not isinstance(pages, list) or not pages:
        raise PageListError(f"{path}: 'pages' must be a non-empty array")
    for name, view in views.items():
        if not SLUG_RE.match(name):
            raise PageListError(f"{path}: view name {name!r} is not a slug")
        if not isinstance(view, dict) or not str(view.get("route", "")).startswith("/"):
            raise PageListError(f"{path}: view {name!r} needs a 'route' starting with '/'")

    slugs: dict[str, str] = {}
    files: dict[str, str] = {}
    published, absent = [], []
    for position, page in enumerate(pages):
        where = f"{path}: pages[{position}]"
        if not isinstance(page, dict):
            raise PageListError(f"{where} is not an object")
        for key in ("file", "slug", "view", "section", "title"):
            if not isinstance(page.get(key), str) or not page[key].strip():
                raise PageListError(f"{where} needs a non-empty string {key!r}")
        unknown = set(page) - {"file", "slug", "view", "section", "title", "pending"}
        if unknown:
            raise PageListError(f"{where} has unknown key(s) {sorted(unknown)}")
        slug, file_rel = page["slug"], page["file"]
        if not SLUG_RE.match(slug) or slug in RESERVED_SLUGS or slug.startswith("nav"):
            raise PageListError(f"{where}: {slug!r} is not a usable slug")
        if slug in slugs:
            raise PageListError(f"{where}: slug {slug!r} is already used by {slugs[slug]}")
        if file_rel in files:
            raise PageListError(f"{where}: {file_rel} is already published as {files[file_rel]!r}")
        if os.path.isabs(file_rel) or ".." in Path(file_rel).parts or not file_rel.endswith(".md"):
            raise PageListError(f"{where}: 'file' must be a repository-relative .md path")
        if page["view"] not in views:
            raise PageListError(f"{where}: unknown view {page['view']!r}")
        if "pending" in page and page["pending"] is not True:
            raise PageListError(f"{where}: 'pending' may only be true; remove it otherwise")
        slugs[slug] = file_rel
        files[file_rel] = slug
        if (root / file_rel).is_file():
            published.append(page)
        elif page.get("pending"):
            absent.append(page)
        else:
            raise PageListError(
                f"{where}: {file_rel} does not exist and is not marked \"pending\": true"
            )

    for name, view in views.items():
        default = view.get("default")
        owners = [p for p in published if p["slug"] == default and p["view"] == name]
        if not owners:
            raise PageListError(
                f"{path}: view {name!r} default {default!r} is not a published page of that view"
            )
    doc["published"] = published
    doc["absent"] = absent
    return doc


def page_href(doc: dict, page: dict, fragment: str = "") -> str:
    route = doc["views"][page["view"]]["route"]
    return f"{route}#page={page['slug']}" + (f"&{fragment}" if fragment else "")


def content_url(page: dict) -> str:
    return f"/content/{page['slug']}.html"


_ATTR_RE = re.compile(r'(<(?P<tag>a|img)\b[^>]*?\s(?P<attr>href|src)=")(?P<url>[^"]*)(")', re.IGNORECASE)
_SCHEME_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*:|//|/)")


def rewrite_links(fragment: str, doc: dict, page: dict, root: Path = ROOT) -> tuple[str, list[str]]:
    """Keep links between published pages on the site; send the rest upstream."""
    by_file = {p["file"]: p for p in doc["published"]}
    source_dir = Path(page["file"]).parent
    unresolved: list[str] = []

    def replace(match: re.Match) -> str:
        head, url, tail = match.group(1), html.unescape(match.group("url")), match.group(5)
        is_image = match.group("tag").lower() == "img"
        if not url or _SCHEME_RE.match(url):
            return match.group(0)
        path_part, _, frag = url.partition("#")
        if not path_part:
            if is_image:
                return match.group(0)
            target = page
        else:
            rel = os.path.normpath(os.path.join(source_dir, path_part)).replace(os.sep, "/")
            if rel.startswith(".."):
                unresolved.append(url)
                return match.group(0)
            target = None if is_image else by_file.get(rel)
            if target is None:
                on_disk = root / rel
                if not on_disk.exists():
                    unresolved.append(url)
                    return match.group(0)
                if is_image:
                    raw = doc["repository"].replace(
                        "https://github.com/", "https://raw.githubusercontent.com/"
                    )
                    raw_url = raw + "/" + doc["branch"] + "/" + rel
                    return f"{head}{html.escape(raw_url, quote=True)}{tail}"
                kind = "tree" if on_disk.is_dir() else "blob"
                upstream = f"{doc['repository']}/{kind}/{doc['branch']}/{rel}"
                if frag:
                    upstream += f"#{frag}"
                return (
                    f'{head}{html.escape(upstream, quote=True)}{tail}'
                    ' target="_blank" rel="noopener noreferrer"'
                )
        href = html.escape(page_href(doc, target, frag), quote=True)
        return f'{head}{href}{tail} data-url="{content_url(target)}"'

    return _ATTR_RE.sub(replace, fragment), unresolved


def render_nav(doc: dict, view: str) -> str:
    """The sidebar of one view: sections in first-appearance order."""
    sections: dict[str, list[dict]] = {}
    for page in doc["published"]:
        if page["view"] == view:
            sections.setdefault(page["section"], []).append(page)
    out = ["<!-- Generated by scripts/build_site_content.py from site/pages.json. Do not edit. -->"]
    for title, pages in sections.items():
        out.append('<div class="docs-sidebar-group">')
        out.append(f'<div class="docs-sidebar-group-title">{html.escape(title)}</div>')
        for page in pages:
            out.append(
                f'<a class="docs-sidebar-item" href="{html.escape(page_href(doc, page), quote=True)}" '
                f'data-url="{content_url(page)}">{html.escape(page["title"])}</a>'
            )
        out.append("</div>")
    return "\n".join(out) + "\n"


def render_index(doc: dict) -> str:
    index = {
        "schema": "eshkol.site-published-pages.v1",
        "generated_by": "scripts/build_site_content.py",
        "views": {
            name: {"route": view["route"], "default": view["default"], "nav": f"/content/{NAV_PREFIX}{name}.html"}
            for name, view in doc["views"].items()
        },
        "pages": [
            {"slug": p["slug"], "title": p["title"], "view": p["view"], "section": p["section"],
             "source": p["file"], "url": content_url(p)}
            for p in doc["published"]
        ],
    }
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def expected_outputs(doc: dict) -> dict[str, str | None]:
    """Generated file name -> exact expected text (None: a rendered page)."""
    outputs: dict[str, str | None] = {f"{p['slug']}.html": None for p in doc["published"]}
    for view in doc["views"]:
        outputs[f"{NAV_PREFIX}{view}.html"] = render_nav(doc, view)
    outputs[INDEX_NAME] = render_index(doc)
    return outputs


def check(doc: dict, content_dir: Path = CONTENT_DIR) -> list[str]:
    """Problems that make the committed content disagree with the page list."""
    problems: list[str] = []
    outputs = expected_outputs(doc)
    on_disk = {p.name for p in content_dir.iterdir() if p.is_file()} if content_dir.is_dir() else set()
    for name in sorted(set(outputs) - on_disk):
        problems.append(f"site/static/content/{name} is missing; run scripts/build-site-content.sh")
    for name in sorted(on_disk - set(outputs)):
        problems.append(
            f"site/static/content/{name} is not produced by site/pages.json; "
            "list its source there or run scripts/build-site-content.sh to remove it"
        )
    for name, text in outputs.items():
        if text is not None and name in on_disk:
            if (content_dir / name).read_text(encoding="utf-8") != text:
                problems.append(
                    f"site/static/content/{name} is stale against site/pages.json; "
                    "run scripts/build-site-content.sh"
                )
    # Every published page must be reachable: listed in the sidebar its view loads.
    for page in doc["published"]:
        nav = outputs[f"{NAV_PREFIX}{page['view']}.html"] or ""
        if f'data-url="{content_url(page)}"' not in nav:
            problems.append(f"{page['slug']} is published but not in the {page['view']} navigation")
    return problems


def write_if_changed(path: Path, text: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, path)
    return True


def build(doc: dict, content_dir: Path = CONTENT_DIR, root: Path = ROOT) -> int:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("build_site_content: pandoc is required to render the pages", file=sys.stderr)
        return 1
    content_dir.mkdir(parents=True, exist_ok=True)
    for page in doc["absent"]:
        print(f"  WARNING: {page['file']} is listed as pending and does not exist yet; "
              f"'{page['title']}' is not published", file=sys.stderr)

    unresolved_total = 0
    for page in doc["published"]:
        result = subprocess.run(
            [pandoc, str(root / page["file"]), "-f", PANDOC_READER, "-t", "html", "--no-highlight", "--wrap=none"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode != 0:
            print(f"build_site_content: pandoc failed on {page['file']}: {result.stderr}", file=sys.stderr)
            return 1
        fragment, unresolved = rewrite_links(result.stdout, doc, page, root)
        unresolved_total += len(unresolved)
        for url in unresolved:
            print(f"  note: {page['file']}: link target not found, left as written: {url}", file=sys.stderr)
        changed = write_if_changed(content_dir / f"{page['slug']}.html", fragment)
        print(f"  {page['file']} -> {page['slug']}.html{'' if changed else ' (unchanged)'}")

    outputs = expected_outputs(doc)
    for name, text in outputs.items():
        if text is not None:
            write_if_changed(content_dir / name, text)
            print(f"  site/pages.json -> {name}")
    for stale in sorted(p for p in content_dir.iterdir() if p.is_file() and p.name not in outputs):
        stale.unlink()
        print(f"  removed {stale.name}: no longer produced by site/pages.json")

    print(f"Published {len(doc['published'])} page(s) in {len(doc['views'])} view(s); "
          f"{len(doc['absent'])} pending; {unresolved_total} link(s) left as written.")
    return 0


def self_test() -> bool:
    """Red/green fixtures for the page-list validator, the drift check and link rewriting."""
    import tempfile

    cases: list[tuple[str, bool]] = []

    def expect(name: str, ok: bool) -> None:
        cases.append((name, ok))

    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".selftest-site-pages-") as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "A.md").write_text("# A\n", encoding="utf-8")
        (root / "docs" / "B.md").write_text("# B\n", encoding="utf-8")
        (root / "docs" / "notes.txt").write_text("x\n", encoding="utf-8")

        def page_list(pages: list[dict], default: str = "a") -> Path:
            path = root / "pages.json"
            path.write_text(json.dumps({
                "schema": SCHEMA, "repository": "https://github.com/o/r", "branch": "master",
                "views": {"docs": {"route": "/docs", "default": default}}, "pages": pages,
            }), encoding="utf-8")
            return path

        a = {"file": "docs/A.md", "slug": "a", "view": "docs", "section": "S", "title": "A"}
        b = {"file": "docs/B.md", "slug": "b", "view": "docs", "section": "S", "title": "B"}
        later = {"file": "docs/LATER.md", "slug": "later", "view": "docs", "section": "S",
                 "title": "Later", "pending": True}

        def loads(pages: list[dict], default: str = "a") -> dict | None:
            try:
                return load_page_list(page_list(pages, default), root)
            except PageListError:
                return None

        doc = loads([a, b, later])
        expect("green_list_loads", doc is not None and [p["slug"] for p in doc["published"]] == ["a", "b"])
        expect("pending_absent_file_is_tolerated", doc is not None and doc["absent"][0]["slug"] == "later")
        expect("absent_file_not_pending_is_rejected",
               loads([a, {k: v for k, v in later.items() if k != "pending"}]) is None)
        expect("duplicate_slug_is_rejected", loads([a, dict(b, slug="a")]) is None)
        expect("duplicate_file_is_rejected", loads([a, dict(a, slug="a2")]) is None)
        expect("unknown_view_is_rejected", loads([a, dict(b, view="blog")]) is None)
        expect("unpublished_default_is_rejected", loads([a, later], default="later") is None)
        expect("reserved_slug_is_rejected", loads([a, dict(b, slug="nav_docs")]) is None)
        expect("path_escape_is_rejected", loads([a, dict(b, file="../B.md")]) is None)

        content = root / "content"
        content.mkdir()
        if doc is not None:
            for name, text in expected_outputs(doc).items():
                (content / name).write_text(text if text is not None else "<p>x</p>", encoding="utf-8")
            expect("check_green", check(doc, content) == [])
            (content / "stray.html").write_text("x", encoding="utf-8")
            expect("check_stray_fragment_fails", check(doc, content) != [])
            (content / "stray.html").unlink()
            (content / "b.html").unlink()
            expect("check_missing_fragment_fails", check(doc, content) != [])
            (content / "b.html").write_text("<p>x</p>", encoding="utf-8")
            (content / "nav-docs.html").write_text("<!-- stale -->", encoding="utf-8")
            expect("check_stale_navigation_fails", check(doc, content) != [])
            nav = render_nav(doc, "docs")
            expect("pending_page_not_in_navigation", "later" not in nav and 'data-url="/content/b.html"' in nav)
            out, unresolved = rewrite_links(
                '<a href="B.md#x">b</a><a href="notes.txt">n</a><a href="gone.md">g</a>', doc, doc["published"][0], root)
            expect("link_to_published_page_stays_on_site",
                   'href="/docs#page=b&amp;x" data-url="/content/b.html"' in out)
            expect("link_to_unpublished_file_goes_upstream",
                   'href="https://github.com/o/r/blob/master/docs/notes.txt"' in out)
            expect("missing_link_target_is_reported", unresolved == ["gone.md"])

    ok = all(passed for _, passed in cases)
    print("build_site_content.py self-test:")
    for name, passed in cases:
        print(f"  [{'OK' if passed else 'GATE IS BROKEN'}] {name}")
    print(f"self-test {'PASS' if ok else 'FAIL'}: {sum(p for _, p in cases)}/{len(cases)} fixtures")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed content against the page list without rendering")
    parser.add_argument("--self-test", action="store_true", help="run the red/green fixture suite")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    try:
        doc = load_page_list()
    except PageListError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.check:
        problems = check(doc)
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        if not problems:
            print(f"PASS: {len(doc['published'])} published page(s), {len(doc['absent'])} pending, "
                  "fragments and navigation agree with site/pages.json")
        return 1 if problems else 0
    return build(doc)


if __name__ == "__main__":
    raise SystemExit(main())
