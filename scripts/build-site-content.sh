#!/usr/bin/env bash
# Build site content: render the documentation pages the website publishes.
#
# site/pages.json is the one declared list of pages (file, slug, view, sidebar
# section). scripts/build_site_content.py renders each listed Markdown file to
# site/static/content/<slug>.html with pandoc, and generates each view's
# sidebar (nav-<view>.html) and the published index (pages.json) from the same
# list. The WASM app loads those fragments at runtime via web-load-content.
#
# To publish another page, add one entry to site/pages.json and rerun this
# script. A listed file that does not exist yet is skipped with a warning when
# its entry is marked "pending": true.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v pandoc >/dev/null 2>&1; then
    echo "build-site-content.sh: pandoc is required" >&2
    exit 1
fi

echo "Building site content from site/pages.json..."
exec python3 scripts/build_site_content.py "$@"
