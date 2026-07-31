"""Replace em dashes with conventional punctuation across tracked files.

Rules, applied in order to each occurrence of an em dash:
  1. Paired dashes around an aside          -> parentheses
  2. Dash before a trailing clause          -> comma, or semicolon if the
                                               clause is independent
  3. Dash used as a list/table bullet gloss -> colon
  4. Unspaced dash between numbers          -> en-dash-free hyphen range

Markdown files are handled with the prose rules; source files get the same
treatment inside comments and docstrings (they contain no em dashes in code).
Run with --check to report remaining occurrences without writing.

Usage: python scripts/strip_emdashes.py [--check]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EM = "—"
EXTS = {".md", ".py", ".yaml", ".yml", ".html", ".ts", ".tsx", ".json", ".txt"}

# Words that start an independent clause: prefer a semicolon over a comma so
# the result is not a comma splice.
INDEPENDENT = re.compile(
    r"^(the|this|that|these|those|it|they|we|our|its|there|he|she|i|"
    r"a|an|no|not|nothing|every|each|both|neither|either|"
    r"[A-Z][a-z]+ (?:et al|shows|found|reports))\b", re.I)


def fix_line(line: str) -> str:
    if EM not in line:
        return line

    # numeric ranges: 5-157M, 41.8-70.9%
    line = re.sub(rf"(\d)\s*{EM}\s*(\d)", r"\1-\2", line)

    # paired asides: ", -- something -- rest" -> ", (something) rest"
    line = re.sub(rf",?\s*{EM}\s*([^{EM}]+?)\s*{EM}\s*", r" (\1) ", line)

    def single(m: re.Match) -> str:
        rest = m.group(1)
        if INDEPENDENT.match(rest.strip()):
            return "; " + rest
        return ", " + rest

    line = re.sub(rf"\s*{EM}\s*(.*)$", lambda m: single(m), line)
    # tidy artefacts
    line = re.sub(r"\s+([,;.])", r"\1", line)
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    line = re.sub(r"[ \t]+$", "", line)
    return line


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
        if EM not in text:
            continue
        if args.check:
            remaining += text.count(EM)
            print(f"{text.count(EM):4}  {rel}")
            continue
        new = "\n".join(fix_line(l) for l in text.split("\n"))
        if new != text:
            p.write_text(new, encoding="utf-8")
            changed += 1
    if args.check:
        print(f"remaining em dashes: {remaining}")
        sys.exit(1 if remaining else 0)
    print(f"rewrote {changed} files")


if __name__ == "__main__":
    main()
