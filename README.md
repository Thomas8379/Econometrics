# Econometrics

Personal econometrics workspace for coursework, supervisions, and independent research.

---

## Folder Structure

```
Econometrics/
├── README.md
├── .gitignore
│
├── data_lake/
│   ├── raw/
│   │   └── wooldridge_and_oleg/   # 92 Wooldridge textbook datasets + 3 local = 95 .dta files
│   │       ├── manifest.json      # tracked: names, sources, SHA-256 hashes
│   │       └── *.dta              # gitignored binary files
│   └── curated/                   # cleaned / merged datasets (Parquet, versioned)
│
├── projects/                      # self-contained analysis projects
│   └── <project-name>/
│       ├── README.md
│       ├── data/                  # symlinks or refs into data_lake/
│       └── notebooks/
│
└── Supervisions/                  # supervision work (existing)
    └── <supervision-N>/
```

---

## Data Sources

### `data_lake/raw/wooldridge_and_oleg/`

| Source | Files | Description |
|--------|-------|-------------|
| Wooldridge | 92 | All datasets from *Introductory Econometrics: A Modern Approach* (Wooldridge). Sourced from [yiyuezhuo/Introductory-Econometrics](https://github.com/yiyuezhuo/Introductory-Econometrics). |
| Local | 3 | `fatkids_db692.dta`, `boat_race.dta`, `wheat_india.dta` — original project data. |

The `manifest.json` in that folder records every file's standardised name, original name, source tag, and SHA-256 hash.

---

## Data Conventions

### Raw is immutable

Files in `data_lake/raw/` are **never modified**. They are the ground-truth source of record. New data is added by appending new files or new source subdirectories — never by overwriting.

### Curated is versioned

`data_lake/curated/` holds cleaned, merged, or feature-engineered datasets derived from raw sources. Files here are versioned by date or semantic version in the filename, e.g. `wages_panel_v1.parquet`.

### Preferred intermediate format: Parquet

Parquet is the standard format for curated datasets and intermediate outputs:
- Efficient columnar storage with built-in compression
- Preserves column types (avoids CSV round-trip issues)
- Compatible with pandas, polars, R (`arrow`), Stata (via Python bridge)

`.dta` files live only in `raw/`. Once loaded into an analysis pipeline, save derived data as Parquet.

### Referencing shared data with hashes

To pin a dataset version in a project, record the filename and SHA-256 hash from `manifest.json`. Example:

```python
import hashlib, pathlib

def verify(path, expected_sha256):
    h = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    assert h == expected_sha256, f"Hash mismatch: {path}"
```

---

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Raw data files | `snake_case.dta` | `cps78_85.dta`, `wheat_india.dta` |
| Curated datasets | `snake_case_vN.parquet` | `wages_panel_v1.parquet` |
| Notebooks | `NN_description.ipynb` | `01_eda_wages.ipynb` |
| Projects | `kebab-case/` | `returns-to-education/` |
| Scripts | `snake_case.py` | `clean_wages.py` |

---

## Git Strategy

- `data_lake/raw/` is **gitignored** — binary `.dta` files are not tracked
- `data_lake/raw/wooldridge_and_oleg/manifest.json` **is tracked** — preserves the file inventory and hashes
- `data_lake/curated/` Parquet files are gitignored by default unless small enough to commit
- Notebooks: strip outputs before committing (`nbstripout` recommended)

---

## Reproducing the Raw Data Lake

```bash
# Re-download Wooldridge datasets
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/yiyuezhuo/Introductory-Econometrics.git /tmp/woo
cd /tmp/woo && git sparse-checkout set "stata data"

# Copy and lowercase
for f in "/tmp/woo/stata data"/*.{dta,DTA}; do
  dest=$(basename "$f" | tr '[:upper:]' '[:lower:]')
  cp "$f" "data_lake/raw/wooldridge_and_oleg/$dest"
done
rm -rf /tmp/woo
```

Local files (`fatkids_db692.dta`, `boat_race.dta`, `wheat_india.dta`) must be sourced separately.
