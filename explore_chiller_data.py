"""
ChillSense — Phase 1 Data Understanding Script
YUKTHI 2026 — Intelligent Energy & Equipment Monitoring

Purpose
-------
This script is NOT the preprocessing pipeline (see preprocessing_pipeline_spec.md
for that). It exists purely as evidence for the Phase 1 evaluation: it re-derives,
directly from the raw CSV, every dataset fact and plot used in the write-up
(dataset shape, missingness, temporal gaps, energy correlations, efficiency
proxy) so nothing is taken on faith from the Data Specification.

Everything here is written generically against the 11-field data contract —
no equipment count, date range, or interval is hard-coded — so it would run
the same way against a held-out judge dataset with a different shape.

Usage
-----
    python explore_chiller_data.py path/to/development_dataset.csv

Outputs (written to ./outputs/):
    01_missingness.png
    02_gaps_load_over_time.png
    03_load_vs_energy.png
    04_efficiency_vs_outside_temp.png
    dataset_facts.txt
    missingness_table.csv
    gap_summary.csv
    correlation_with_energy.csv
    efficiency_proxy_stats.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config (named, overridable — nothing here is a "magic number" tied to a
# specific dataset; these are exactly the thresholds from the preprocessing
# spec, kept here only for reproducing the diagnostic plots/tables)
# ---------------------------------------------------------------------------
MINOR_GAP_MULTIPLIER = 4     # delta beyond this many x nominal_interval = "major" gap
OUTPUT_DIR = "outputs"

EXPECTED_COLUMNS = [
    "timestamp", "equipment_id",
    "Chilled Water Rate (L/sec)", "Cooling Water Temperature (C)", "Building Load (RT)",
    "Chiller Energy Consumption (kWh)",
    "Outside Temperature (F)", "Dew Point (F)", "Humidity (%)", "Wind Speed (mph)", "Pressure (in)",
]

MEASUREMENT_FIELDS = [c for c in EXPECTED_COLUMNS if c not in ("timestamp", "equipment_id")]
ENERGY_COL = "Chiller Energy Consumption (kWh)"
LOAD_COL = "Building Load (RT)"
OUTSIDE_TEMP_COL = "Outside Temperature (F)"


def log(msg):
    print(msg)


# ---------------------------------------------------------------------------
# Stage 1 — Schema validation (gatekeeper)
# ---------------------------------------------------------------------------
def load_and_validate(path):
    df = pd.read_csv(path)

    # Case/whitespace-tolerant column match, per spec section 1
    normalized = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in EXPECTED_COLUMNS if c.strip().lower() not in normalized]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Re-map to canonical names in case of casing/whitespace drift
    rename_map = {normalized[c.strip().lower()]: c for c in EXPECTED_COLUMNS}
    df = df.rename(columns=rename_map)[EXPECTED_COLUMNS]

    # Parse timestamp; log unparseable values instead of silently dropping
    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_timestamps = df.loc[parsed.isna(), "timestamp"]
    if len(bad_timestamps):
        log(f"  WARNING: {len(bad_timestamps)} rows had unparseable timestamps (see indices below)")
        log(f"  {list(bad_timestamps.index)}")
    df["timestamp"] = parsed

    # Coerce measurement fields to numeric; count coercion failures separately
    # from genuinely-blank cells
    coercion_failures = {}
    for col in MEASUREMENT_FIELDS:
        raw_non_null = df[col].notna()
        coerced = pd.to_numeric(df[col], errors="coerce")
        failed = raw_non_null & coerced.isna()
        if failed.sum():
            coercion_failures[col] = int(failed.sum())
        df[col] = coerced

    if coercion_failures:
        log(f"  WARNING: coercion failures (non-numeric text, not blank): {coercion_failures}")
    else:
        log("  No coercion failures — all measurement fields are clean numeric or genuinely blank.")

    return df


# ---------------------------------------------------------------------------
# Stage 2 — Dataset facts (discovered, not assumed)
# ---------------------------------------------------------------------------
def dataset_facts(df):
    facts = {}
    facts["rows"] = len(df)
    facts["fields"] = df.shape[1]

    equip_counts = df["equipment_id"].value_counts().sort_index()
    facts["equipment_units"] = list(equip_counts.index)
    facts["rows_per_equipment"] = equip_counts.to_dict()

    facts["date_range_start"] = df["timestamp"].min()
    facts["date_range_end"] = df["timestamp"].max()

    dup_mask = df.duplicated(subset=["equipment_id", "timestamp"], keep=False)
    facts["duplicate_equip_timestamp_pairs"] = int(dup_mask.sum())

    # Discover nominal interval per equipment independently (never assume 30 min)
    nominal_intervals = {}
    for eq, grp in df.groupby("equipment_id"):
        grp_sorted = grp.sort_values("timestamp")
        deltas = grp_sorted["timestamp"].diff().dropna()
        if len(deltas):
            nominal_intervals[eq] = deltas.mode().iloc[0]
    facts["nominal_interval_per_equipment"] = nominal_intervals

    return facts


def print_facts(facts):
    log("\n=== DATASET FACTS (discovered from file) ===")
    log(f"Rows: {facts['rows']}")
    log(f"Fields: {facts['fields']}")
    log(f"Equipment units ({len(facts['equipment_units'])}): {facts['equipment_units']}")
    for eq, n in facts["rows_per_equipment"].items():
        log(f"  {eq}: {n} rows")
    log(f"Date range: {facts['date_range_start']} to {facts['date_range_end']}")
    log(f"Duplicate (equipment_id, timestamp) pairs: {facts['duplicate_equip_timestamp_pairs']}")
    log("Nominal interval per equipment (modal delta, NOT assumed):")
    for eq, interval in facts["nominal_interval_per_equipment"].items():
        log(f"  {eq}: {interval}")


# ---------------------------------------------------------------------------
# Stage 3 — Missingness
# ---------------------------------------------------------------------------
def missingness_table(df):
    miss = df[MEASUREMENT_FIELDS].isna().sum()
    pct = (miss / len(df) * 100).round(3)
    table = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    table = table.sort_values("missing_pct", ascending=False)
    return table


def plot_missingness(table, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    nonzero = table[table["missing_pct"] > 0].sort_values("missing_pct")
    ax.barh(nonzero.index, nonzero["missing_pct"], color="#3b6fa0")
    ax.set_xlabel("% missing")
    ax.set_title("Missing Values by Field")
    for i, v in enumerate(nonzero["missing_pct"]):
        ax.text(v, i, f" {v:.3f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 4 — Temporal gap detection (per equipment, independently sorted)
# ---------------------------------------------------------------------------
def detect_gaps(df, minor_gap_multiplier=MINOR_GAP_MULTIPLIER):
    """Returns (gap_registry_df, summary_df). Gap registry has one row per
    detected discontinuity (delta > nominal_interval)."""
    registry_rows = []
    for eq, grp in df.groupby("equipment_id"):
        grp_sorted = grp.sort_values("timestamp").reset_index(drop=True)
        deltas = grp_sorted["timestamp"].diff()
        nominal = deltas.mode().iloc[0]

        for i in range(1, len(grp_sorted)):
            delta = deltas.iloc[i]
            if pd.isna(delta):
                continue
            if delta <= nominal * 1.2:  # within tolerance of nominal -> not a gap
                continue
            classification = "major" if delta > nominal * minor_gap_multiplier else "minor"
            registry_rows.append({
                "equipment_id": eq,
                "gap_start_timestamp": grp_sorted.loc[i - 1, "timestamp"],
                "gap_end_timestamp": grp_sorted.loc[i, "timestamp"],
                "gap_duration": delta,
                "classification": classification,
            })

    registry = pd.DataFrame(registry_rows)

    summary_rows = []
    for eq in df["equipment_id"].unique():
        eq_gaps = registry[registry["equipment_id"] == eq] if len(registry) else pd.DataFrame()
        n_total = len(eq_gaps)
        n_major = (eq_gaps["classification"] == "major").sum() if n_total else 0
        largest = eq_gaps["gap_duration"].max() if n_total else pd.Timedelta(0)
        summary_rows.append({
            "equipment_id": eq,
            "gaps_beyond_nominal": n_total,
            "major_gaps": n_major,
            "largest_gap": largest,
        })
    summary = pd.DataFrame(summary_rows).sort_values("equipment_id")

    return registry, summary


def plot_gaps(df, registry, out_path):
    equipment_ids = sorted(df["equipment_id"].unique())
    fig, axes = plt.subplots(len(equipment_ids), 1, figsize=(11, 3 * len(equipment_ids)), sharex=True)
    if len(equipment_ids) == 1:
        axes = [axes]

    for ax, eq in zip(axes, equipment_ids):
        grp = df[df["equipment_id"] == eq].sort_values("timestamp")
        ax.plot(grp["timestamp"], grp[LOAD_COL], color="seagreen", linewidth=0.6)
        ax.set_title(f"{eq}: Building Load over time (red = major gap)")
        ax.set_ylabel("RT")

        eq_gaps = registry[(registry["equipment_id"] == eq) & (registry["classification"] == "major")]
        for _, row in eq_gaps.iterrows():
            ax.axvspan(row["gap_start_timestamp"], row["gap_end_timestamp"], color="red", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 5 — Correlation with energy
# ---------------------------------------------------------------------------
def correlation_with_energy(df):
    other_fields = [c for c in MEASUREMENT_FIELDS if c != ENERGY_COL]
    corrs = df[other_fields + [ENERGY_COL]].corr()[ENERGY_COL].drop(ENERGY_COL)
    corrs = corrs.sort_values(ascending=False)
    return corrs.to_frame(name="correlation_with_energy_kwh")


def plot_load_vs_energy(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for eq, grp in df.groupby("equipment_id"):
        ax.scatter(grp[LOAD_COL], grp[ENERGY_COL], s=4, alpha=0.3, label=eq)
    ax.set_xlabel("Building Load (RT)")
    ax.set_ylabel("Chiller Energy Consumption (kWh)")
    ax.set_title("Load vs Energy Consumption by Equipment")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 6 — Efficiency proxy (guarded against near-zero load)
# ---------------------------------------------------------------------------
def efficiency_proxy(df, near_zero_load_threshold=1e-3):
    df = df.copy()
    guard = df[LOAD_COL].abs() < near_zero_load_threshold
    df["efficiency_proxy"] = np.where(guard, np.nan, df[ENERGY_COL] / df[LOAD_COL])
    df["efficiency_undefined_low_load"] = guard

    n_guarded = int(guard.sum())
    log(f"\nEfficiency proxy near-zero-load guard: {n_guarded} row(s) hit the guard "
        f"(threshold |load| < {near_zero_load_threshold}).")

    stats = df.groupby("equipment_id")["efficiency_proxy"].agg(
        mean="mean", std="std", median="median", max="max"
    ).round(3)
    return df, stats


def plot_efficiency_vs_temp(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for eq, grp in df.groupby("equipment_id"):
        ax.scatter(grp[OUTSIDE_TEMP_COL], grp["efficiency_proxy"], s=4, alpha=0.3, label=eq)
    ax.set_xlabel("Outside Temperature (F)")
    ax.set_ylabel("Efficiency proxy (kWh/RT)")
    ax.set_title("Efficiency Proxy vs Outside Temperature")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(csv_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"Loading and validating: {csv_path}")
    df = load_and_validate(csv_path)

    facts = dataset_facts(df)
    print_facts(facts)
    with open(os.path.join(OUTPUT_DIR, "dataset_facts.txt"), "w") as f:
        for k, v in facts.items():
            f.write(f"{k}: {v}\n")

    log("\n=== MISSINGNESS ===")
    miss_table = missingness_table(df)
    log(miss_table.to_string())
    miss_table.to_csv(os.path.join(OUTPUT_DIR, "missingness_table.csv"))
    plot_missingness(miss_table, os.path.join(OUTPUT_DIR, "01_missingness.png"))

    log("\n=== TEMPORAL GAPS (major gap = delta > 4x nominal interval) ===")
    registry, gap_summary = detect_gaps(df)
    log(gap_summary.to_string(index=False))
    gap_summary.to_csv(os.path.join(OUTPUT_DIR, "gap_summary.csv"), index=False)
    plot_gaps(df, registry, os.path.join(OUTPUT_DIR, "02_gaps_load_over_time.png"))

    log("\n=== CORRELATION WITH ENERGY ===")
    corr_table = correlation_with_energy(df)
    log(corr_table.to_string())
    corr_table.to_csv(os.path.join(OUTPUT_DIR, "correlation_with_energy.csv"))
    plot_load_vs_energy(df, os.path.join(OUTPUT_DIR, "03_load_vs_energy.png"))

    log("\n=== EFFICIENCY PROXY (kWh / RT) ===")
    df_eff, eff_stats = efficiency_proxy(df)
    log(eff_stats.to_string())
    eff_stats.to_csv(os.path.join(OUTPUT_DIR, "efficiency_proxy_stats.csv"))
    plot_efficiency_vs_temp(df_eff, os.path.join(OUTPUT_DIR, "04_efficiency_vs_outside_temp.png"))

    log(f"\nDone. All tables and plots written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_chiller_data.py path/to/development_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
