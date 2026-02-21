"""
tools/csv_filter.py
--------------------
Universal Kaggle skin dataset normalizer.

DROP any skin-tone CSV from Kaggle into  d:/ANTI/Skin1/data/raw/
Then run:
    python tools/csv_filter.py

It will auto-detect columns, remap them, and output:
    d:/ANTI/Skin1/data/skin_dataset.csv

Required output columns
-----------------------
  tone_label    : str   – Fair / Medium / Tan / Deep
  L_value       : float – LAB lightness (0-255)
  R, G, B       : int   – skin RGB values (0-255)
  source_file   : str   – original filename tag

The script handles all of these common Kaggle column variants:
  - skin_tone, Skin Tone, tone, label, class, category
  - L, l_value, lightness, lab_l, L_channel
  - R / Red / r, G / Green / g, B / Blue / b
  - hex, hex_color, color_hex  (auto-converts #rrggbb → R G B)
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "skin_dataset.csv")


# ── Column synonym maps ───────────────────────────────────────────────────
TONE_SYNONYMS = [
    "skin_tone", "skintone", "skin tone", "tone", "label", "class",
    "category", "fitzpatrick", "skin_type", "skin type", "type",
    "shade", "complexion",
]

L_SYNONYMS = [
    "l", "l_value", "lightness", "lab_l", "l_channel",
    "luminance", "luma", "brightness",
]

R_SYNONYMS = ["r", "red", "r_value", "rgb_r"]
G_SYNONYMS = ["g", "green", "g_value", "rgb_g"]
B_SYNONYMS = ["b", "blue", "b_value", "rgb_b"]
HEX_SYNONYMS = ["hex", "hex_color", "color_hex", "hexcode", "hex_value", "#hex"]


# ── Tone normalisation map ────────────────────────────────────────────────
# Raw labels from various datasets → our 4 canonical classes
TONE_MAP = {
    # Fair variants
    "fair": "Fair", "very fair": "Fair", "light": "Fair",
    "very light": "Fair", "pale": "Fair", "white": "Fair",
    "i": "Fair", "1": "Fair", "type i": "Fair", "type_i": "Fair",
    # Medium variants
    "medium": "Medium", "beige": "Medium", "nude": "Medium",
    "natural": "Medium", "normal": "Medium", "olive": "Medium",
    "ii": "Medium", "iii": "Medium", "2": "Medium", "3": "Medium",
    "type ii": "Medium", "type iii": "Medium",
    "type_ii": "Medium", "type_iii": "Medium",
    # Tan variants
    "tan": "Tan", "caramel": "Tan", "brown": "Tan",
    "golden": "Tan", "sand": "Tan", "warm": "Tan",
    "iv": "Tan", "4": "Tan", "type iv": "Tan", "type_iv": "Tan",
    # Deep variants
    "deep": "Deep", "dark": "Deep", "mahogany": "Deep",
    "espresso": "Deep", "rich": "Deep", "ebony": "Deep",
    "v": "Deep", "vi": "Deep", "5": "Deep", "6": "Deep",
    "type v": "Deep", "type vi": "Deep",
    "type_v": "Deep", "type_vi": "Deep",
}


# ── Helpers ────────────────────────────────────────────────────────────────
def normalise_col(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def find_col(df: pd.DataFrame, synonyms: list) -> str | None:
    normed = {normalise_col(c): c for c in df.columns}
    for s in synonyms:
        key = normalise_col(s)
        if key in normed:
            return normed[key]
    return None


def hex_to_rgb(hex_str: str) -> tuple[int, int, int] | None:
    """Convert #rrggbb or rrggbb string to (R, G, B) int tuple."""
    hex_str = str(hex_str).strip().lstrip("#")
    m = re.fullmatch(r"([0-9a-fA-F]{6})", hex_str)
    if m:
        val = int(m.group(1), 16)
        return (val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF
    return None


def rgb_to_lab_l(r: float, g: float, b: float) -> float:
    """
    Approximate LAB L channel from RGB (0-255).
    Uses sRGB → linear → XYZ → L* conversion.
    Returns L* scaled to 0-255 to match our threshold system.
    """
    rgb = np.array([r, g, b], dtype=float) / 255.0
    # linearise
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # XYZ (D65)
    Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    # L* (CIE)
    fy = Y ** (1/3) if Y > 0.008856 else 7.787 * Y + 16/116
    l_star = max(0.0, 116 * fy - 16)
    # Scale to 0-255
    return round(l_star * 255 / 100, 2)


def map_tone_label(raw: str) -> str:
    """Normalise raw tone string to our 4-class system."""
    key = str(raw).strip().lower()
    return TONE_MAP.get(key, None)


# ── L-scale detector (samples whole column, not per-row) ─────────────────
def detect_l_scale(df: pd.DataFrame, l_col: str) -> str:
    """
    Determine the scale of the L column by inspecting its value range:
      '0-1'   → values are 0.0 – 1.0  (CSS-style lightness)
      '0-100' → values are 0 – 100    (CIE L*)
      '0-255' → values are 0 – 255    (our internal scale)
    """
    try:
        vals = pd.to_numeric(df[l_col], errors="coerce").dropna()
        if vals.empty:
            return '0-100'
        col_max = vals.max()
        if col_max <= 1.01:    # float 0-1
            return '0-1'
        elif col_max <= 101:   # L* 0-100
            return '0-100'
        else:                  # already 0-255
            return '0-255'
    except Exception:
        return '0-100'


def l_to_255(raw_l: float, scale: str) -> float:
    """Convert any L-scale value to our internal 0-255 system."""
    if scale == '0-1':
        return raw_l * 255.0
    elif scale == '0-100':
        return raw_l * 255.0 / 100.0
    else:
        return raw_l   # already 0-255


# ── Per-file processor ────────────────────────────────────────────────────
def process_file(path: str) -> pd.DataFrame | None:
    fname = os.path.basename(path)
    print(f"\n  ▶ Processing: {fname}")

    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    print(f"    Columns found: {list(df.columns)}")
    print(f"    Rows: {len(df)}")

    rows = []

    # Locate columns
    tone_col = find_col(df, TONE_SYNONYMS)
    l_col    = find_col(df, L_SYNONYMS)
    r_col    = find_col(df, R_SYNONYMS)
    g_col    = find_col(df, G_SYNONYMS)
    b_col    = find_col(df, B_SYNONYMS)
    hex_col  = find_col(df, HEX_SYNONYMS)

    print(f"    Detected → tone:{tone_col}  L:{l_col}  "
          f"R:{r_col} G:{g_col} B:{b_col}  hex:{hex_col}")

    if tone_col is None and l_col is None and hex_col is None \
       and not (r_col and g_col and b_col):
        print(f"    [SKIP] Cannot find usable columns in {fname}")
        return None

    # ── Detect L scale ONCE before the row loop ────────────────────────────
    l_scale = detect_l_scale(df, l_col) if l_col else '0-100'
    if l_col:
        print(f"    L-scale detected: {l_scale}")

    for _, row in df.iterrows():
        # ── Tone label ────────────────────────────────────────────────────
        tone = None
        if tone_col:
            tone = map_tone_label(str(row[tone_col]))

        # ── RGB  (always try hex if present, even when R/G/B cols exist) ──
        r = g = b = None
        hex_raw = None
        if r_col and g_col and b_col:
            try:
                r, g, b = int(row[r_col]), int(row[g_col]), int(row[b_col])
            except (ValueError, TypeError):
                pass
        if hex_col:   # always read hex string for dedup key
            hex_raw = str(row[hex_col]).strip()
            if r is None:   # only compute RGB from hex if not already set
                rgb = hex_to_rgb(hex_raw)
                if rgb:
                    r, g, b = rgb

        # ── L value ───────────────────────────────────────────────────────
        l_val = None
        if l_col:
            try:
                raw_l = float(row[l_col])
                l_val = l_to_255(raw_l, l_scale)   # ← scale-aware conversion
            except (ValueError, TypeError):
                pass
        if l_val is None and r is not None:
            l_val = rgb_to_lab_l(r, g, b)          # compute from RGB fallback

        # Auto-classify tone from L if no label
        if tone is None and l_val is not None:
            if l_val > 170:   tone = "Fair"
            elif l_val > 140: tone = "Medium"
            elif l_val > 110: tone = "Tan"
            else:             tone = "Deep"

        if tone is None:
            continue

        rows.append({
            "tone_label":  tone,
            "L_value":     round(l_val, 2) if l_val is not None else None,
            "R":           r,
            "G":           g,
            "B":           b,
            "hex":         hex_raw,    # keep for dedup
            "source_file": fname,
        })

    result = pd.DataFrame(rows)
    print(f"    → Kept {len(result)} valid rows")
    return result if len(result) > 0 else None


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Normalise Kaggle skin-tone CSVs to Skin1 standard format."
    )
    parser.add_argument("--input",  default=RAW_DIR,     help="Folder with raw CSVs")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(args.input,  exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    csv_files = glob.glob(os.path.join(args.input, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"\n[!] No CSV files found in: {args.input}")
        print(    "    Drop your Kaggle CSVs into that folder and re-run.")
        sys.exit(0)

    print(f"\n{'━'*60}")
    print(f"  Skin1 CSV Filter  |  Found {len(csv_files)} file(s)")
    print(f"{'━'*60}")

    frames = []
    for path in csv_files:
        result = process_file(path)
        if result is not None:
            frames.append(result)

    if not frames:
        print("\n[ERROR] No usable data extracted from any file.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # ── Smart deduplication ───────────────────────────────────────────────
    # Priority: deduplicate by (R,G,B) when available, by hex string otherwise.
    # Rows where ALL of R,G,B are NaN would collapse to one row with naive
    # drop_duplicates — so we handle them separately.
    has_rgb = combined[["R", "G", "B"]].notna().all(axis=1)
    rgb_rows = combined[has_rgb].drop_duplicates(
        subset=["R", "G", "B"], keep="first"
    )
    hex_only_rows = combined[~has_rgb]
    if "hex" in hex_only_rows.columns:
        hex_only_rows = hex_only_rows.drop_duplicates(subset=["hex"], keep="first")
    combined = pd.concat([rgb_rows, hex_only_rows], ignore_index=True)

    # Drop the hex helper column from output
    if "hex" in combined.columns:
        combined.drop(columns=["hex"], inplace=True)

    combined.sort_values("L_value", ascending=False, na_position="last", inplace=True)
    combined.to_csv(args.output, index=False)

    print(f"\n{'━'*60}")
    print(f"  ✅ Output: {args.output}")
    print(f"  Total rows: {len(combined)}")
    print(f"\n  Breakdown by tone class:")
    for tone, cnt in combined["tone_label"].value_counts().items():
        bar = "█" * min(40, int(cnt / max(len(combined), 1) * 40))
        print(f"    {tone:<8} {cnt:>5}  {bar}")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
