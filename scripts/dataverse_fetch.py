"""Harvard Dataverse acquisition pipeline.

Fetch published replication datasets from any Dataverse instance into
data_lake/raw/, with SHA-256 integrity checking and manifest tracking.

Commands
--------
search
    Full-text search for datasets.  Supports client-side author filtering.
    Harvard Dataverse's search API does not support server-side filter queries
    (fq is not implemented), so author/keyword post-filtering is done locally.

        python scripts/dataverse_fetch.py search --query "Angrist" --max-results 50
        python scripts/dataverse_fetch.py search --query "Angrist" --author "Angrist"
        python scripts/dataverse_fetch.py search --query "returns to education" --max-results 100

inventory
    List every file in one dataset.

        python scripts/dataverse_fetch.py inventory --doi doi:10.7910/DVN/XXXXXXX

fetch
    Download files from a single dataset DOI.

        python scripts/dataverse_fetch.py fetch \\
            --doi doi:10.7910/DVN/XXXXXXX \\
            --dest data_lake/raw/angrist_replication \\
            --types dta csv

batch-fetch
    Search for datasets matching a query/author filter and download all of them.
    Useful for pulling everything by a given author or on a topic.

        # All datasets with "Angrist" as an author:
        python scripts/dataverse_fetch.py batch-fetch \\
            --query "Angrist" --author "Angrist" \\
            --dest data_lake/raw/angrist_replication

        # All Angrist + Krueger papers, tabular data only:
        python scripts/dataverse_fetch.py batch-fetch \\
            --query "Angrist Krueger" --author "Angrist" \\
            --dest data_lake/raw/angrist_replication \\
            --types tab dta csv

        # Dry run first to see what would be downloaded:
        python scripts/dataverse_fetch.py batch-fetch \\
            --query "Angrist" --author "Angrist" \\
            --dest data_lake/raw/angrist_replication --dry-run

Notes
-----
- The Dataverse search API paginates via the `start` parameter (max 1000 per page).
- `--author` is a case-insensitive substring match against the `authors` field returned
  by the search API.  It is purely client-side post-filtering.
- Re-running any fetch command is safe: files already on disk with matching SHA-256
  hashes are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

_DEFAULT_SERVER = "https://dataverse.harvard.edu"
_TIMEOUT = 60  # seconds per request
_RETRY_DELAYS = (1, 2, 4)  # exponential backoff delays in seconds


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(
    url: str,
    params: dict[str, Any] | None = None,
    stream: bool = False,
) -> requests.Response:
    """GET with exponential backoff for transient network errors.

    HTTP 4xx/5xx errors are raised immediately (no retry).
    Network-level errors (timeout, connection reset) are retried up to
    len(_RETRY_DELAYS) times.
    """
    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT, stream=stream)
            if resp.status_code >= 400:
                body = "" if stream else resp.text[:500]
                print(f"HTTP {resp.status_code}: {url}", file=sys.stderr)
                if body:
                    print(body, file=sys.stderr)
                resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError:
            raise  # do not retry HTTP errors
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                print(
                    f"  Attempt {attempt + 1} failed, retrying in {delay}s: {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataverse API calls
# ---------------------------------------------------------------------------


def search_datasets(
    query: str,
    max_results: int,
    server: str,
    author_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search Dataverse for datasets matching *query*, with optional author post-filter.

    Paginates automatically until *max_results* matching records are collected
    or the API is exhausted.  Harvard Dataverse's search API does not support
    server-side filter queries (fq), so author filtering is done client-side by
    checking whether *author_filter* (case-insensitive substring) appears in any
    element of the ``authors`` list returned per result.
    """
    url = f"{server}/api/search"
    collected: list[dict[str, Any]] = []
    page_size = min(max_results, 1000)
    start = 0

    while len(collected) < max_results:
        params: dict[str, Any] = {
            "q": query,
            "type": "dataset",
            "per_page": page_size,
            "start": start,
        }
        resp = _get(url, params=params)
        payload = resp.json()
        data = payload.get("data", {})
        page_items: list[dict[str, Any]] = data.get("items", [])
        total_count: int = data.get("total_count", 0)

        if not page_items:
            break

        for item in page_items:
            if author_filter:
                authors = item.get("authors") or []
                needle = author_filter.lower()
                if not any(needle in (a or "").lower() for a in authors):
                    continue
            collected.append(item)
            if len(collected) >= max_results:
                break

        start += len(page_items)
        if start >= total_count:
            break  # exhausted all server results

    return collected


def list_files(doi: str, server: str) -> list[dict[str, Any]]:
    """Return normalised file metadata for every file in the latest dataset version."""
    url = f"{server}/api/datasets/:persistentId/versions/:latest/files"
    params = {"persistentId": doi}
    resp = _get(url, params=params)
    payload = resp.json()

    # Dataverse returns {"status":"OK","data":[...]} but guard against bare lists
    if isinstance(payload, list):
        raw_files: list[dict[str, Any]] = payload
    else:
        raw_files = payload.get("data", [])

    files: list[dict[str, Any]] = []
    for item in raw_files:
        df = item.get("dataFile", {})
        files.append(
            {
                "file_id": df.get("id"),
                "filename": df.get("filename") or item.get("label", ""),
                "size_bytes": df.get("filesize", 0),
                "content_type": df.get("contentType", "application/octet-stream"),
            }
        )
    return files


def stream_file(file_id: int, dest: pathlib.Path, server: str) -> None:
    """Stream *file_id* from *server* into *dest* (write in chunks)."""
    url = f"{server}/api/access/datafile/{file_id}"
    resp = _get(url, stream=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: pathlib.Path) -> list[dict[str, Any]]:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return []


def _save_manifest(manifest_path: pathlib.Path, entries: list[dict[str, Any]]) -> None:
    manifest_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _find_by_file_id(
    entries: list[dict[str, Any]], file_id: int | None
) -> dict[str, Any] | None:
    if file_id is None:
        return None
    return next((e for e in entries if e.get("dataverse_file_id") == file_id), None)


def _find_by_filename(
    entries: list[dict[str, Any]], filename: str
) -> dict[str, Any] | None:
    return next((e for e in entries if e.get("filename") == filename), None)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace) -> int:
    author_filter: str | None = getattr(args, "author", None)
    suffix = f", author filter: {author_filter!r}" if author_filter else ""
    print(f"Searching {args.server} for: {args.query!r}{suffix} ...\n")

    items = search_datasets(args.query, args.max_results, args.server, author_filter)
    if not items:
        print("No results found.")
        return 0

    doi_w = max((len(i.get("global_id", "")) for i in items), default=3)
    doi_w = max(doi_w, 3)
    author_w = 30
    sep = "-" * (doi_w + 6 + 6 + author_w + 4 + 40)
    print(f"{'DOI':<{doi_w}}  {'Year':<6}  {'Files':>5}  {'Authors':<{author_w}}  Title")
    print(sep)
    for item in items:
        doi = item.get("global_id", "")
        title = (item.get("name") or "")[:38]
        pub = item.get("published_at") or ""
        year = pub[:4] if pub else ""
        file_count = item.get("file_count") or item.get("fileCount") or "?"
        authors = "; ".join(item.get("authors") or [])[:author_w]
        print(f"{doi:<{doi_w}}  {year:<6}  {str(file_count):>5}  {authors:<{author_w}}  {title}")

    print(f"\n{len(items)} result(s)  (--max-results={args.max_results})")
    return 0


def cmd_batch_fetch(args: argparse.Namespace) -> int:
    """Search then download every matching dataset into a single destination directory."""
    author_filter: str | None = getattr(args, "author", None)
    dest = pathlib.Path(args.dest)
    dry_run: bool = args.dry_run
    types: list[str] | None = (
        [t.lstrip(".").lower() for t in args.types] if args.types else None
    )

    suffix = f", author filter: {author_filter!r}" if author_filter else ""
    print(f"Searching {args.server} for: {args.query!r}{suffix} ...\n")
    items = search_datasets(args.query, args.max_results, args.server, author_filter)

    if not items:
        print("No datasets matched — nothing to fetch.")
        return 0

    print(f"Found {len(items)} dataset(s).\n")
    for i, item in enumerate(items, 1):
        doi = item.get("global_id", "")
        title = (item.get("name") or "")[:70]
        authors = "; ".join(item.get("authors") or [])
        file_count = item.get("file_count") or item.get("fileCount") or "?"
        print(f"[{i}/{len(items)}] {doi}")
        print(f"        {title}")
        print(f"        Authors: {authors}  |  Files: {file_count}")

        if dry_run:
            print("        [DRY-RUN] skipping download\n")
            continue

        if not doi:
            print("        WARN: no DOI, skipping\n", file=sys.stderr)
            continue

        # Delegate to the core fetch logic by constructing a mock namespace
        fetch_args = argparse.Namespace(
            doi=doi,
            dest=str(dest),
            types=args.types,
            dry_run=False,
            server=args.server,
        )
        rc = cmd_fetch(fetch_args)
        if rc != 0:
            print(f"        WARN: fetch returned exit code {rc}\n", file=sys.stderr)
        else:
            print()

    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    print(f"Fetching file list for {args.doi} ...", file=sys.stderr)
    files = list_files(args.doi, args.server)
    if not files:
        print("No files found in this dataset.", file=sys.stderr)
        return 0

    # Human-readable table → stderr so JSON on stdout stays pipe-friendly
    print(f"\nFiles in {args.doi} ({len(files)} total):", file=sys.stderr)
    print(f"{'ID':<12}  {'Size (B)':>12}  {'Type':<35}  Filename", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    for f in files:
        size_str = f"{f['size_bytes']:,}" if f["size_bytes"] else "?"
        print(
            f"{f['file_id']:<12}  {size_str:>12}  {f['content_type']:<35}  {f['filename']}",
            file=sys.stderr,
        )

    # JSON → stdout
    print(json.dumps(files, indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    dest = pathlib.Path(args.dest)
    doi: str = args.doi
    server: str = args.server
    dry_run: bool = args.dry_run
    types: list[str] | None = (
        [t.lstrip(".").lower() for t in args.types] if args.types else None
    )
    source_tag = dest.name

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    manifest_path = dest / "manifest.json"
    manifest: list[dict[str, Any]] = [] if dry_run else _load_manifest(manifest_path)

    print(f"Fetching file list for {doi} ...")
    files = list_files(doi, server)
    if not files:
        print("No files found in dataset.")
        return 0

    if types:
        files = [
            f
            for f in files
            if pathlib.Path(f["filename"]).suffix.lstrip(".").lower() in types
        ]
    if not files:
        print("No matching files after --types filter.")
        return 0

    print(f"{len(files)} file(s) to process.\n")
    downloaded = skipped = failed = 0

    for f in files:
        filename: str = f["filename"]
        file_id: int | None = f["file_id"]
        size_bytes: int = f["size_bytes"]
        content_type: str = f["content_type"]

        file_path = dest / filename
        tmp_path = dest / (filename + ".tmp")
        partial_path = dest / (filename + ".partial")
        size_str = f"{size_bytes:,}" if size_bytes else "unknown"

        if dry_run:
            print(f"  [DRY-RUN] {filename}  ({size_str} bytes, file_id={file_id})")
            continue

        # Check manifest by file_id (fastest dedup path)
        if file_id is not None and _find_by_file_id(manifest, file_id) is not None:
            if file_path.exists():
                print(f"  SKIP  {filename}  (already in manifest)")
                skipped += 1
                continue
            # Entry in manifest but file missing — re-download
            print(
                f"  WARN  {filename}  (manifest entry present but file missing — re-downloading)"
            )

        # File exists on disk — verify hash if we have a manifest entry by name
        if file_path.exists():
            by_name = _find_by_filename(manifest, filename)
            if by_name and _sha256_file(file_path) == by_name.get("sha256", ""):
                print(f"  SKIP  {filename}  (file exists, hash verified)")
                skipped += 1
                continue

        print(
            f"  GET   {filename}  ({size_str} bytes, file_id={file_id}) ... ",
            end="",
            flush=True,
        )
        try:
            stream_file(file_id, tmp_path, server)
            sha = _sha256_file(tmp_path)

            # Handle filename collisions (e.g. case-insensitive filesystems on Windows/macOS)
            if file_path.exists():
                existing_sha = _sha256_file(file_path)
                if existing_sha == sha:
                    # Identical content — de-duplicate silently
                    tmp_path.unlink()
                    print(f"sha256={sha[:12]}... (de-dup: matches existing {filename})")
                else:
                    # Different content — disambiguate with file_id suffix
                    alt_name = f"{file_path.stem}__{file_id}{file_path.suffix}"
                    alt_path = dest / alt_name
                    tmp_path.rename(alt_path)
                    filename = alt_name
                    file_path = alt_path
                    sha = _sha256_file(file_path)
                    print(f"sha256={sha[:12]}... (stored as {alt_name} due to name collision)")
            else:
                tmp_path.rename(file_path)
                print(f"sha256={sha[:12]}...")

            downloaded += 1

            entry: dict[str, Any] = {
                "filename": filename,
                "source_tag": source_tag,
                "doi": doi,
                "dataverse_file_id": file_id,
                "dataverse_server": server,
                "sha256": sha,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            manifest.append(entry)
            _save_manifest(manifest_path, manifest)

        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            if tmp_path.exists():
                tmp_path.rename(partial_path)
                print(
                    f"  Partial download preserved at: {partial_path}",
                    file=sys.stderr,
                )
            failed += 1

    if not dry_run:
        print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        print(f"Manifest: {manifest_path}  ({len(manifest)} total entries)")

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataverse_fetch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default=_DEFAULT_SERVER,
        metavar="URL",
        help=f"Dataverse server URL (default: {_DEFAULT_SERVER})",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- search ---
    p_search = sub.add_parser("search", help="Search for datasets by keyword")
    p_search.add_argument("--query", "-q", required=True, help="Search query")
    p_search.add_argument(
        "--max-results",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of results to show (default: 20)",
    )
    p_search.add_argument(
        "--author",
        metavar="NAME",
        help=(
            "Case-insensitive substring to match against author names. "
            "Applied client-side after the API search (the Dataverse API "
            "does not support server-side author filtering)."
        ),
    )

    # --- inventory ---
    p_inv = sub.add_parser("inventory", help="List all files in a dataset")
    p_inv.add_argument(
        "--doi",
        required=True,
        metavar="DOI",
        help="Persistent DOI (e.g. doi:10.7910/DVN/XXXXX)",
    )

    # --- fetch ---
    p_fetch = sub.add_parser("fetch", help="Download files from a Dataverse dataset")
    p_fetch.add_argument(
        "--doi",
        required=True,
        metavar="DOI",
        help="Persistent DOI (e.g. doi:10.7910/DVN/XXXXX)",
    )
    p_fetch.add_argument(
        "--dest",
        required=True,
        metavar="DIR",
        help="Destination directory (e.g. data_lake/raw/angrist_replication)",
    )
    p_fetch.add_argument(
        "--types",
        nargs="+",
        metavar="EXT",
        help="Filter to these file extensions (e.g. --types dta csv)",
    )
    p_fetch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing any files",
    )

    # --- batch-fetch ---
    p_batch = sub.add_parser(
        "batch-fetch",
        help="Search for datasets and download all matching ones",
    )
    p_batch.add_argument(
        "--query", "-q", required=True,
        help="Search query (free text, e.g. author surname or topic keywords)",
    )
    p_batch.add_argument(
        "--author",
        metavar="NAME",
        help=(
            "Case-insensitive substring filter on the authors field. "
            "Use this to pull all datasets by a specific person "
            "(e.g. --author Angrist). Applied client-side."
        ),
    )
    p_batch.add_argument(
        "--dest", required=True, metavar="DIR",
        help="Destination directory for all downloaded files",
    )
    p_batch.add_argument(
        "--types", nargs="+", metavar="EXT",
        help="Filter to these file extensions (e.g. --types tab dta csv)",
    )
    p_batch.add_argument(
        "--max-results",
        type=int,
        default=200,
        metavar="N",
        help=(
            "Maximum number of datasets to consider (default: 200). "
            "Set higher if the author has more deposits."
        ),
    )
    p_batch.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching datasets without downloading anything",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    dispatch = {
        "search": cmd_search,
        "inventory": cmd_inventory,
        "fetch": cmd_fetch,
        "batch-fetch": cmd_batch_fetch,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
