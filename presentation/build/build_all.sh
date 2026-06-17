#!/bin/bash
# Build every deliverable: dark on-screen deck (PDF), light/print deck (PDF), editable DOCX.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VENV="$DIR/../raven-api/.venv/bin/python"
cd "$DIR"

echo "1) charts (dark) + light assets + PNGs"
python3 build/make_charts.py
python3 build/make_assets.py

echo "2) dark slides deck -> Raven-Presentation.pdf"
python3 build/assemble.py dark slides raven-presentation.html ""
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$DIR/Raven-Presentation.pdf" "$DIR/raven-presentation.html" 2>/dev/null

echo "3) light + flow print deck -> Raven-Presentation-Print.pdf"
python3 build/assemble.py light flow raven-presentation-print.html light
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$DIR/Raven-Presentation-Print.pdf" "$DIR/raven-presentation-print.html" 2>/dev/null

echo "4) editable Word -> Raven-Presentation.docx"
"$VENV" build/make_docx.py

echo ""
ls -la "$DIR"/*.pdf "$DIR"/*.docx
