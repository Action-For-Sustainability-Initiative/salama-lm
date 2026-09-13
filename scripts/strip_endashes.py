"""Replace en dashes with a plain hyphen across tracked files.

Unlike em dashes (see strip_emdashes.py), every en-dash usage found in this
repo is either a numeric/step range ("41.8-70.9%", "steps 0-580") or a
hyphenated compound modifier ("English-Kiswahili"), both of which read as
completely normal English with a plain hyphen. No contextual parsing needed,
so this is a straight substitution.

Usage: python scripts/strip_endashes.py [--check]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EN = "–"
EXTS = {".md", ".py", ".yaml", ".yml", ".html", ".ts", ".tsx", ".json", ".txt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           check=True).stdout.split()
    changed = remaining = 0
    for rel in files:
        p = Path(rel)
        if p.suffix not in EXTS or not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if EN not in text:
            continue
        if args.check:
            remaining += text.count(EN)
            print(f"{text.count(EN):4}  {rel}")
            continue
        p.write_text(text.replace(EN, "-"), encoding="utf-8")
        changed += 1
    if args.check:
        print(f"remaining en dashes: {remaining}")
        sys.exit(1 if remaining else 0)
    print(f"rewrote {changed} files")


if __name__ == "__main__":
    main()
