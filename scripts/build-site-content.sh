#!/bin/bash
# Build site content: Convert markdown docs to HTML fragments for the website.
# These HTML fragments are fetched by the WASM app at runtime via web-load-content.

set -e
cd "$(dirname "$0")/.."

CONTENT_DIR="site/static/content"
mkdir -p "$CONTENT_DIR"

echo "Building site content from markdown docs..."

# Root docs (LANGUAGE_GUIDE/QUICK_REFERENCE moved under docs/ in the v1.3 root
# reorg and are handled by the docs/ loop below; ANNOUNCEMENT added in v1.3).
for doc in ROADMAP CONTRIBUTING ANNOUNCEMENT; do
    if [ -f "${doc}.md" ]; then
        out="$CONTENT_DIR/$(echo "$doc" | tr '[:upper:]' '[:lower:]').html"
        pandoc "${doc}.md" -f markdown -t html --no-highlight -o "$out"
        echo "  ${doc}.md -> $(basename $out)"
    fi
done

# docs/ subdirectory
for doc in docs/ESHKOL_LANGUAGE_GUIDE docs/ESHKOL_QUICK_REFERENCE docs/COMPLETE_LANGUAGE_SPECIFICATION docs/API_REFERENCE docs/FEATURE_MATRIX docs/QUICKSTART; do
    if [ -f "${doc}.md" ]; then
        base=$(basename "$doc")
        out="$CONTENT_DIR/$(echo "$base" | tr '[:upper:]' '[:lower:]').html"
        pandoc "${doc}.md" -f markdown -t html --no-highlight -o "$out"
        echo "  ${doc}.md -> $(basename $out)"
    fi
done

echo "Done. Content files:"
ls -la "$CONTENT_DIR"/*.html 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}'
