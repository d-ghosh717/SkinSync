# Skin1 – Dataset Guide

## Where to Put Your Kaggle CSV Files

Drop ALL your raw Kaggle CSVs into this folder:

```
d:\ANTI\Skin1\data\raw\
```

Any filename is fine. Nested subfolders are also scanned.

## Then Run the Filter

```bash
cd d:\ANTI\Skin1
python tools/csv_filter.py
```

This produces a clean, standardised file at:

```
d:\ANTI\Skin1\data\skin_dataset.csv
```

## What the Filter Does

1. **Auto-detects columns** — works with any Kaggle skin CSV regardless of column names
2. **Recognises tone labels** automatically: `fitzpatrick`, `shade`, `type I-VI`, `fair/medium/tan/deep`, etc.
3. **Handles hex colours** — converts `#rrggbb` → R, G, B automatically
4. **Generates L-value** from RGB if not present in CSV
5. **Deduplicates** by RGB value
6. **Reports** a breakdown of tone classes after filtering

## Output Columns

| Column | Type | Description |
|---|---|---|
| `tone_label` | str | Fair / Medium / Tan / Deep |
| `L_value` | float | LAB lightness 0–255 |
| `R` | int | Red channel 0–255 |
| `G` | int | Green channel 0–255 |
| `B` | int | Blue channel 0–255 |
| `source_file` | str | Original CSV filename |

## Recommended Kaggle Datasets

These are known to work with the filter:

| Dataset | What it has |
|---|---|
| Skin Tone Classification | fitzpatrick scale + RGB |
| Diverse Faces Skin | hex color codes |
| Foundation Shade Finder | tone labels + L values |
| Skin Color Dataset | R, G, B columns |

## Custom Options

```bash
# Point to a different input folder
python tools/csv_filter.py --input path/to/csvs --output path/to/output.csv
```
