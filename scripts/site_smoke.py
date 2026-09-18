#!/usr/bin/env python3
"""Browser smoke test of the committed website bundle (site/static).

Serves site/static on a loopback port with the same pathname fallback GitHub
Pages provides, drives it in a real browser engine through Playwright, and
fails on any console error or uncaught page error. Requests to other hosts
(fonts, the GitHub releases API) are answered locally with empty responses, so
the test needs no network and cannot be made green or red by a third party.

It checks that:
  - the homepage renders the "what's new" heading for the recorded release;
  - the downloads page shows the recorded tag and an install URL for it;
  - /docs and /tutorials load their generated navigation, with one sidebar
    item per page site/pages.json publishes in that view;
  - documentation pages open from the sidebar, from a #page= address, and
    from a link inside another published page.

Usage:
    python3 scripts/site_smoke.py [--channel chrome] [--screenshots DIR]

Requires the Python `playwright` package and, for --channel chrome, an
installed Google Chrome (use --channel chromium for Playwright's own build).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "site" / "static"
# Pages opened from the sidebar; each must be published by site/pages.json.
SIDEBAR_PAGES = ("known_issues", "changelog", "release_notes", "vm_parity", "faq",
                 "guide_automatic_differentiation", "reference_language")
DIRECT_PAGE = ("/tutorials", "tutorials_10_macros")


class SpaHandler(http.server.SimpleHTTPRequestHandler):
    """Static files, with index.html for extension-less paths (as Pages' 404 redirect does)."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm"}

    def send_head(self):  # noqa: D401 - http.server API
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if not (STATIC / path.lstrip("/")).exists() and "." not in path.rsplit("/", 1)[-1]:
            self.path = "/index.html"
        return super().send_head()

    def log_message(self, *args) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", default="chrome", help="browser channel (chrome, chromium, msedge)")
    parser.add_argument("--screenshots", type=Path, help="directory for full-page screenshots")
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("site_smoke: the Python 'playwright' package is required", file=sys.stderr)
        return 2

    tag = json.loads((ROOT / "tests/coverage/release_record.json").read_text(encoding="utf-8"))["tag"]
    pages = json.loads((STATIC / "content" / "pages.json").read_text(encoding="utf-8"))
    per_view = {view: sum(1 for p in pages["pages"] if p["view"] == view) for view in pages["views"]}
    published = {p["slug"] for p in pages["pages"]}

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SpaHandler, directory=str(STATIC)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    failures: list[str] = []
    errors: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            failures.append(message)

    def stub_external(route) -> None:
        url = route.request.url
        if url.startswith(base):
            route.continue_()
        elif "api.github.com" in url:
            route.fulfill(status=200, content_type="application/json", body="[]")
        elif route.request.resource_type == "stylesheet":
            route.fulfill(status=200, content_type="text/css", body="")
        else:
            route.fulfill(status=204, body="")

    def shot(page, name: str) -> None:
        if args.screenshots:
            args.screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=True)

    def doc_area_ok(page, area: str) -> bool:
        page.wait_for_function(
            "id => { const a = document.getElementById(id);"
            " return a && a.querySelector('h1, h2') && !a.textContent.includes('Loading...'); }",
            arg=area, timeout=15000)
        text = page.inner_text(f"#{area}")
        return "Failed to load" not in text and len(text) > 200

    print(f"site_smoke: {base} ({args.channel}), release {tag}")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel=args.channel, headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.route("**/*", stub_external)
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

            page.goto(base + "/", wait_until="networkidle")
            page.wait_for_selector("#content h2", timeout=15000)
            heading = f"What's new in {tag}"
            check(page.get_by_text(heading, exact=True).count() == 1, f"homepage renders '{heading}'")
            check(page.get_by_text("A parser with no recursion budget").count() == 1, "homepage what's-new cards render")
            shot(page, "home")
            page.get_by_text(heading, exact=True).scroll_into_view_if_needed()
            if args.screenshots:
                page.screenshot(path=str(args.screenshots / "whats-new.png"))

            page.goto(base + "/downloads", wait_until="networkidle")
            page.wait_for_selector("#release-status", timeout=15000)
            body = page.inner_text("#content")
            check(f"{tag} Publication" in body, "downloads page names the recorded release")
            check(f"releases/download/{tag}/eshkol-{tag}-linux-x64-lite.tar.gz" in body,
                  "Linux install snippet downloads the recorded release")
            shot(page, "downloads")

            for view, area in (("docs", "doc-content-area"), ("tutorials", "tutorial-content-area")):
                page.goto(base + pages["views"][view]["route"], wait_until="networkidle")
                page.wait_for_function(
                    "n => document.querySelectorAll('.docs-sidebar-item[data-url]').length === n",
                    arg=per_view[view], timeout=15000)
                check(True, f"/{view} sidebar lists all {per_view[view]} published pages")
                check(doc_area_ok(page, area), f"/{view} default page renders")
                shot(page, f"{view}-default")

            page.goto(base + "/docs", wait_until="networkidle")
            for slug in SIDEBAR_PAGES:
                if slug not in published:
                    check(False, f"{slug} is not published by site/pages.json")
                    continue
                page.click(f'.docs-sidebar-item[data-url="/content/{slug}.html"]')
                page.wait_for_function(
                    "u => document.querySelector('.docs-sidebar-item.active')?.getAttribute('data-url') === u",
                    arg=f"/content/{slug}.html", timeout=15000)
                check(doc_area_ok(page, "doc-content-area") and page.url.endswith(f"#page={slug}"),
                      f"sidebar opens {slug}")
                shot(page, f"docs-{slug}")

            link = page.locator('#doc-content-area a[data-url^="/content/"]').first
            if link.count():
                target = link.get_attribute("data-url")
                link.click()
                page.wait_for_function(
                    "u => document.querySelector('.docs-sidebar-item.active')?.getAttribute('data-url') === u",
                    arg=target, timeout=15000)
                check(doc_area_ok(page, "doc-content-area"), f"in-page link opens {target} on the site")
            else:
                check(False, "no in-page link between published pages to follow")

            route, slug = DIRECT_PAGE
            page.goto(f"{base}{route}#page={slug}", wait_until="networkidle")
            page.wait_for_function(
                "u => document.querySelector('.docs-sidebar-item.active')?.getAttribute('data-url') === u",
                arg=f"/content/{slug}.html", timeout=15000)
            check(doc_area_ok(page, "tutorial-content-area"), f"{route}#page={slug} opens that page directly")
            shot(page, f"tutorials-{slug}")

            browser.close()
    except Exception as exc:  # a timeout or a launch failure is a failed smoke test
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  [FAIL] {type(exc).__name__}: {str(exc).splitlines()[0]}")
    finally:
        server.shutdown()

    for error in errors:
        print(f"  [FAIL] {error}")
    ok = not failures and not errors
    print(f"site_smoke: {'PASS' if ok else 'FAIL'} ({len(failures)} failed check(s), {len(errors)} console error(s))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
