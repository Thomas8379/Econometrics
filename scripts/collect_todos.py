"""Collect all TODO(econtools) comments and print grouped by category.

Usage:
    python scripts/collect_todos.py
    python scripts/collect_todos.py --root path/to/econtools
    python scripts/collect_todos.py --category adapter
    python scripts/collect_todos.py --format json > todos.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PATTERN = re.compile(
    r"#\s*TODO\(econtools\):\s*(\w+)\s*[-—]\s*(.+)"
)

_CATEGORIES = [
    "kb-entry",
    "test",
    "adapter",
    "render",
    "validate",
    "cli-cmd",
    "profile",
]


def find_todos(root: Path) -> list[dict[str, str]]:
    """Walk *root* recursively and collect all TODO(econtools) comments."""
    results: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        # Skip __pycache__ and .git
        parts = path.parts
        if "__pycache__" in parts or ".git" in parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = _PATTERN.search(line)
            if m:
                results.append(
                    {
                        "category": m.group(1),
                        "description": m.group(2).strip(),
                        "file": str(path.relative_to(root.parent)),
                        "line": str(lineno),
                    }
                )
    # Also scan YAML files
    for path in sorted(root.rglob("*.yaml")):
        parts = path.parts
        if "__pycache__" in parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = _PATTERN.search(line)
            if m:
                results.append(
                    {
                        "category": m.group(1),
                        "description": m.group(2).strip(),
                        "file": str(path.relative_to(root.parent)),
                        "line": str(lineno),
                    }
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "econtools"),
        help="Package root to scan (default: econtools/)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by category (e.g. adapter, test, kb-entry)",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path not found: {root}", file=sys.stderr)
        return 1

    todos = find_todos(root)
    if args.category:
        todos = [t for t in todos if t["category"] == args.category]

    if args.format == "json":
        print(json.dumps(todos, indent=2))
        return 0

    # Group by category
    grouped: dict[str, list[dict[str, str]]] = {}
    for todo in todos:
        grouped.setdefault(todo["category"], []).append(todo)

    if not grouped:
        print("No TODO(econtools) comments found.")
        return 0

    total = len(todos)
    print(f"Found {total} TODO(econtools) item(s):\n")
    for cat in sorted(grouped.keys()):
        items = grouped[cat]
        print(f"[{cat}] ({len(items)})")
        for item in items:
            loc = f"{item['file']}:{item['line']}"
            print(f"  {loc}")
            print(f"    {item['description']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
