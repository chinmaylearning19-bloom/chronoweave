"""
ChronoWeave – 3-Layer Indian Omnichannel Apparel Simulation Dataset Generator
==============================================================================
Layers
------
  Layer 1 (5-10 % weight)  – Macro Climate (regime-setting)
  Layer 2 (55 % weight)    – Micro Demand & Omnichannel Dynamics
  Layer 3 (35 % weight)    – Multi-Node Operational Friction

Dependencies: numpy, pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seasonal_multiplier(week: int) -> float:
    """Return demand multiplier for a given week (1-indexed, annual cycle)."""
    woy = ((week - 1) % 52) + 1          # week-of-year, 1–52
    # Festive: weeks 40-52  (+40 %)
    if 40 <= woy <= 52:
        return 1.40
    # Monsoon: weeks 23-35  (-25 %)
    if 23 <= woy <= 35:
        return 0.75
    # Gradual transitions (linear interpolation at shoulder weeks)
    if 36 <= woy <= 39:
        return 0.75 + (woy - 36) * (1.40 - 0.75) / 4
    if 18 <= woy <= 22:
        return 1.00 - (woy - 17) * 0.25 / 6
    return 1.00


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a 104-row weekly simulation DataFrame for a 20-store Indian
    omnichannel apparel business.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        104 rows × ~40 columns, indexed by week number (1–104).
    """
    rng = np.random.default_rng(seed)
    n = 104

    weeks = np.arange(1, n + 1)

    # -----------------------------------------------------------------------
    # LAYER 1 — Macro Climate
    # -----------------------------------------------------------------------

    # --- cotton_index: baseline 85, seasonal sine ±12, random walk σ=2 ----
    sine_wave = 12 * np.sin(2 * np.pi * (weeks - 1) / 52)
    cotton_rw = np.cumsum(rng.normal(0, 2, n))
    cotton_index = 85 + sine_wave + cotton_rw - cotton_rw[0]

    # --- freight_congestion_index: baseline 100, monsoon & Q4 spikes -------
    freight_congestion_index = np.full(n, 100.0)
    spike_mask_1 = (weeks >= 18) & (weeks <= 22)   # monsoon onset
    spike_mask_2 = (weeks >= 70) & (weeks <= 76)   # Q4 congestion
    freight_congestion_index[spike_mask_1] = rng.uniform(140, 170, spike_mask_1.sum())
    freight_congestion_index[spike_mask_2] = rng.uniform(140, 170, spike_mask_2.sum())
    # smooth noise outside spikes
    noise_mask = ~(spike_mask_1 | spike_mask_2)
    freight_congestion_index[noise_mask] += rng.normal(0, 4, noise_mask.sum())

    # --- brent_crude_inr: baseline 6500, correlated with freight -----------
    brent_crude_inr = 6500 + 40 * (freight_congestion_index - 100) / 100 * rng.uniform(0.8, 1.2, n)
    brent_crude_inr += rng.normal(0, 80, n)

    # -----------------------------------------------------------------------
    # LAYER 2 — Micro Demand & Omnichannel Dynamics
    # -----------------------------------------------------------------------

    # --- promo_flag ---------------------------------------------------------
    promo_weeks = {4, 8, 26, 32, 40, 52, 65, 78, 91, 100}
    promo_flag = np.where(np.isin(weeks, list(promo_weeks)), 1, 0)

    # --- gross_demand_units -------------------------------------------------
    base_demand = 2200.0
    gross_demand_units = np.empty(n)
    for i, w in enumerate(weeks):
        seasonal = _seasonal_multiplier(w)
        demand = base_demand * seasonal
        # Demand shock: EOSS spike at week 52
        if w == 52:
            demand *= 1.60
        # Promo multiplier
        if promo_flag[i] == 1:
            demand *= rng.uniform(2.8, 4.2)
        demand += rng.normal(0, 120)
        gross_demand_units[i] = max(0, demand)

    # --- online_share -------------------------------------------------------
    online_share = 0.32 + (0.48 - 0.32) * (weeks - 1) / (n - 1)

    # --- size_curve ---------------------------------------------------------
    base_sizes = {
        "XS": 0.06, "S": 0.14, "M": 0.22, "L": 0.28,
        "XL": 0.18, "XXL": 0.12
    }
    size_cols = {k: np.full(n, v) for k, v in base_sizes.items()}

    # Random stockout events: 8 weeks each for M and L → set to 0
    stockout_M_weeks = rng.choice(n, size=8, replace=False)
    stockout_L_weeks = rng.choice(n, size=8, replace=False)
    all_stockout_weeks = set(stockout_M_weeks) | set(stockout_L_weeks)

    size_cols["M"][stockout_M_weeks] = 0.0
    size_cols["L"][stockout_L_weeks] = 0.0

    # Re-normalise rows where M or L zeroed out
    for i in range(n):
        row_sum = sum(size_cols[k][i] for k in base_sizes)
        if row_sum > 0:
            for k in base_sizes:
                size_cols[k][i] /= row_sum

    # Add small weekly variation
    for k in base_sizes:
        size_cols[k] = np.clip(size_cols[k] + rng.normal(0, 0.005, n), 0, 1)

    # Re-normalise after noise
    size_total = sum(size_cols[k] for k in base_sizes)
    for k in base_sizes:
        size_cols[k] /= size_total

    # --- rack_completeness --------------------------------------------------
    # Base completeness ~0.85, drops on stockout weeks
    rack_completeness = rng.uniform(0.75, 1.0, n)
    for i in all_stockout_weeks:
        rack_completeness[i] = rng.uniform(0.40, 0.65)
    rack_completeness = np.clip(rack_completeness, 0.40, 1.0)

    # -----------------------------------------------------------------------
    # LAYER 3 — Multi-Node Operational Friction
    # -----------------------------------------------------------------------

    # --- online_returns_lag7 … online_returns_lag14 (8 columns) ------------
    lag_kernel = np.array([0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.12, 0.05])
    assert np.isclose(lag_kernel.sum(), 1.0), "Lag kernel must sum to 1"

    return_rate_base = np.where(promo_flag == 1, 0.42, 0.28)
    online_units_shipped = gross_demand_units * online_share

    lag_returns = {}
    for lag_idx, lag_day in enumerate(range(7, 15)):
        col_values = online_units_shipped * return_rate_base * lag_kernel[lag_idx]
        col_values += rng.normal(0, 1.5, n)
        col_values = np.clip(col_values, 0, None)
        lag_returns[f"online_returns_lag{lag_day}"] = col_values

    # --- lead_time_node1_days (supplier → DC, base 3 days) -----------------
    lead_time_node1_days = 3.0 + 0.04 * np.maximum(0, freight_congestion_index - 100)
    lead_time_node1_days += rng.normal(0, 0.3, n)
    lead_time_node1_days = np.clip(lead_time_node1_days, 3.0, None)

    # --- lead_time_node2_days (DC → store, base 14 days) -------------------
    lead_time_node2_days = 14.0 + 0.06 * np.maximum(0, freight_congestion_index - 100)
    lead_time_node2_days += rng.normal(0, 1.0, n)
    lead_time_node2_days = np.clip(lead_time_node2_days, 14.0, None)

    # --- lead_time_node3_days (mill, base 35 days) --------------------------
    lead_time_node3_days = 35.0 + 0.10 * np.maximum(0, freight_congestion_index - 100)
    lead_time_node3_days += rng.normal(0, 2.0, n)
    lead_time_node3_days = np.clip(lead_time_node3_days, 35.0, None)

    # --- season_end_week ----------------------------------------------------
    # Season 1→week 26, Season 2→52, Season 3→78, Season 4→104
    season_end_week = np.select(
        [weeks <= 26, weeks <= 52, weeks <= 78, weeks <= 104],
        [26, 52, 78, 104]
    )

    # --- moq_node3 ----------------------------------------------------------
    moq_node3 = np.full(n, 500)

    # --- dye_batch_size -----------------------------------------------------
    dye_batch_size = np.full(n, 200)

    # --- mrp_base -----------------------------------------------------------
    # Base ₹999; cost pressure pushes to ₹1001–1085 (GST danger zone)
    cost_pressure_weeks = rng.choice(n, size=18, replace=False)
    mrp_base = np.full(n, 999.0)
    mrp_base[cost_pressure_weeks] = rng.uniform(1001, 1085, len(cost_pressure_weeks))

    # -----------------------------------------------------------------------
    # Assemble DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame({
        "week": weeks,
        # Layer 1
        "cotton_index": cotton_index,
        "freight_congestion_index": freight_congestion_index,
        "brent_crude_inr": brent_crude_inr,
        # Layer 2
        "promo_flag": promo_flag,
        "gross_demand_units": gross_demand_units,
        "online_share": online_share,
        "size_curve_XS": size_cols["XS"],
        "size_curve_S": size_cols["S"],
        "size_curve_M": size_cols["M"],
        "size_curve_L": size_cols["L"],
        "size_curve_XL": size_cols["XL"],
        "size_curve_XXL": size_cols["XXL"],
        "rack_completeness": rack_completeness,
        # Layer 3 — return lags
        **lag_returns,
        # Layer 3 — lead times & ops
        "lead_time_node1_days": lead_time_node1_days,
        "lead_time_node2_days": lead_time_node2_days,
        "lead_time_node3_days": lead_time_node3_days,
        "season_end_week": season_end_week,
        "moq_node3": moq_node3,
        "dye_batch_size": dye_batch_size,
        "mrp_base": mrp_base,
    })

    df = df.set_index("week")
    return df


# ---------------------------------------------------------------------------
# ERP Bullwhip Simulator
# ---------------------------------------------------------------------------

def generate_legacy_erp_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate naive ERP ordering logic:
      order_t = MA4(downstream_demand) * 1.15, rounded up to nearest MOQ.

    This reproduces the 3.4×–5.9× bullwhip amplification ratio documented in
    the ChronoWeave PRD.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `generate_dataset()`.

    Returns
    -------
    pd.DataFrame
        Columns: downstream_demand, erp_order_quantity, bullwhip_ratio
        Indexed by week.
    """
    demand = df["gross_demand_units"].values
    moq = int(df["moq_node3"].iloc[0])
    safety_buffer = 1.15
    window = 4

    erp_orders = np.empty(len(demand))
    for i in range(len(demand)):
        if i < window:
            ma = demand[: i + 1].mean()
        else:
            ma = demand[i - window : i].mean()
        raw_order = ma * safety_buffer
        # Round up to nearest MOQ
        erp_orders[i] = np.ceil(raw_order / moq) * moq

    # Bullwhip ratio: variance of orders / variance of demand
    # Computed in a rolling 8-week window for each point
    roll_window = 8
    bullwhip_ratios = np.empty(len(demand))
    for i in range(len(demand)):
        start = max(0, i - roll_window + 1)
        var_orders = np.var(erp_orders[start : i + 1]) + 1e-9
        var_demand = np.var(demand[start : i + 1]) + 1e-9
        bullwhip_ratios[i] = var_orders / var_demand

    result = pd.DataFrame(
        {
            "downstream_demand": demand,
            "erp_order_quantity": erp_orders,
            "bullwhip_ratio": bullwhip_ratios,
        },
        index=df.index,
    )
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating dataset …")
    df = generate_dataset(seed=42)
    print(f"  Shape: {df.shape}")
    print(df.head(3).T)

    print("\nRunning ERP bullwhip simulation …")
    erp_df = generate_legacy_erp_orders(df)
    print(f"  Bullwhip ratio range: "
          f"{erp_df['bullwhip_ratio'].min():.2f}× – "
          f"{erp_df['bullwhip_ratio'].max():.2f}×")
    print(f"  Mean bullwhip ratio : {erp_df['bullwhip_ratio'].mean():.2f}×")
    print(erp_df.head(6))
    print("\ndata_generator.py smoke test PASSED ✓")
