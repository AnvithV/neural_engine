#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to build the PDF. Install it first." >&2
  exit 1
fi

pandoc report.md -o report.pdf --from markdown --pdf-engine=xelatex
echo "Wrote $SCRIPT_DIR/report.pdf"

