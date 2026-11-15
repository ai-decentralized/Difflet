"""insert_rows.py <doc> <rows file>: each '<tag>\t<row>' line of task6.py dataset's output goes
right before the doc's '<!-- <tag> rows -->' marker ('15b-phase' -> '<!-- 15b phase rows -->');
a row already in the doc is not added twice."""
import sys
from pathlib import Path

doc, rows = Path(sys.argv[1]), Path(sys.argv[2])
text = doc.read_text()
for line in rows.read_text().splitlines():
    tag, _, row = line.partition("\t")
    if not row or row in text:
        continue
    marker = f"<!-- {tag.replace('-', ' ')} rows -->"
    assert marker in text, marker
    text = text.replace(marker, f"{row}\n{marker}")
doc.write_text(text)
print("ok")
