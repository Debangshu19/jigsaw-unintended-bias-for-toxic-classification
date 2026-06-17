#!/bin/bash
# Assemble the HTML then render it to PDF with headless Google Chrome.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../presentation
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

python3 "$DIR/build/assemble.py"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$DIR/Raven-Presentation.pdf" \
  "$DIR/raven-presentation.html" 2>/dev/null

echo "PDF: $DIR/Raven-Presentation.pdf"
ls -la "$DIR/Raven-Presentation.pdf"
