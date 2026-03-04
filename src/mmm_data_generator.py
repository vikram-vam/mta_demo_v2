"""
mmm_data_generator.py
=====================
Synthetic weekly aggregate data generator for Marketing Mix Modelling (MMM).

Designed to complement the existing MTA journey-level data in mta_demo_v2.
Offline channels (TV, Radio, Direct Mail, Agent Visit) are the primary focus —
these channels leave no digital footprint in customer journeys, so MTA cannot
credit them. MMM works at the aggregate level and fills that gap.

What this module generates
--------------------------
A single DataFrame with one row per ISO week (52–104 weeks).
Columns fall into four groups:

  SPEND / INPUT VARIABLES  (what the marketing team controls)
  ┌──────────────────────────────────────────────────────────┐
  │ Offline channels                                          │
  │   tv_grp            – Gross Rating Points aired that week │
  │   tv_spend          – USD spend on TV                    │
  │   radio_impressions – Total radio impressions (000s)     │
  │   radio_spend       – USD spend on radio                 │
  │   direct_mail_pieces– Pieces mailed that week            │
  │   direct_mail_spend – USD spend on direct mail           │
  │   agent_visits      – Number of agent visits             │
  │   agent_visit_spend – USD spend on agent visits          │
  │                                                           │
  │ Online channels (aggregate mirror of MTA journey data)   │
  │   paid_search_spend                                       │
  │   display_spend                                           │
  │   paid_social_spend                                       │
  │   email_spend                                             │
  └──────────────────────────────────────────────────────────┘

  EXTERNAL / CONTROL VARIABLES  (not controllable, must be modelled out)
  ┌──────────────────────────────────────────────────────────┐
  │   seasonality_index  – 0–1 demand multiplier             │
  │   is_holiday_week    – binary flag                       │
  │   competitor_spend   – estimated competitor weekly spend │
  │   macro_index        – simplified economic health index  │
  └──────────────────────────────────────────────────────────┘

  ADSTOCK-TRANSFORMED VARIABLES  (ready-to-use MMM features)
  ┌──────────────────────────────────────────────────────────┐
  │   {channel}_adstock  – adstock-transformed spend/GRP     │
  └──────────────────────────────────────────────────────────┘

  TARGET VARIABLE
  ┌──────────────────────────────────────────────────────────┐
  │   conversions        – total conversions that week       │
  │   revenue            – total revenue that week           │
  └──────────────────────────────────────────────────────────┘

Ground-truth contribution shares (used for model validation)
-------------------------------------------------------------
These are the TRUE shares baked into the synthetic DGP.
After fitting MMM you can compare recovered shares to these.

Key design decisions
--------------------
1. Adstock: carryover effect means a TV ad aired in week 1 still influences
   week 2, 3, etc. with exponential decay. Different rates per channel.
2. Saturation: diminishing returns via Hill function — doubling spend does
   not double conversions.
3. Seasonality: sinusoidal annual pattern + holiday spikes.
4. Noise: realistic multiplicative noise on conversions.
5. Consistent with MTA: online channel spend levels are calibrated to match
   the CPT (cost-per-touchpoint) values in data_generator.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple


# ── Ground-truth contribution parameters ─────────────────────────────────────
# These define what % of incremental conversions each channel TRULY drives.
# After running MMM, recovered values should be close to these.

CHANNEL_TRUE_CONTRIBUTION = {
    "tv":           0.18,   # strong awareness, slow burn
    "radio":        0.07,   # moderate reach
    "direct_mail":  0.09,   # targeted, moderate response
    "agent_visit":  0.14,   # highest close rate offline
    "paid_search":  0.22,   # high intent, best online performer
    "display":      0.06,   # low direct contribution
    "paid_social":  0.12,   # mid-funnel social
    "email":        0.08,   # retention / nurture
    # remainder = base conversions (seasonality, brand equity, etc.)
    "base":         0.04,
}

# ── Adstock decay rates (λ) per channel ──────────────────────────────────────
# adstock_t = spend_t + λ * adstock_(t-1)
# Higher λ = longer carryover (TV lingers longer than email)

ADSTOCK_DECAY = {
    "tv":           0.65,   # TV has long memory (~3-4 week carryover)
    "radio":        0.45,   # moderate carryover
    "direct_mail":  0.20,   # short carryover (physical, not broadcast)
    "agent_visit":  0.30,   # moderate — relationship effect
    "paid_search":  0.10,   # near-immediate response
    "display":      0.25,   # some banner carryover
    "paid_social":  0.15,   # mostly immediate
    "email":        0.05,   # very short carryover
}

# ── Saturation (Hill function) parameters ────────────────────────────────────
# response = spend^alpha / (spend^alpha + K^alpha)
# K = half-saturation point (spend level at which you get 50% of max response)
# alpha = shape (steepness of the S-curve). Higher = more diminishing returns

SATURATION_K = {
    "tv":           150_000,
    "radio":         40_000,
    "direct_mail":   25_000,
    "agent_visit":   30_000,
    "paid_search":   45_000,
    "display":       20_000,
    "paid_social":   35_000,
    "email":          8_000,
}

SATURATION_ALPHA = {
    "tv":           0.8,
    "radio":        0.9,
    "direct_mail":  0.85,
    "agent_visit":  0.75,
    "paid_search":  1.1,
    "display":      0.9,
    "paid_social":  0.95,
    "email":        1.0,
}

# ── Weekly spend ranges (USD) per channel ────────────────────────────────────
# Calibrated to match MTA CPT values and realistic media plans

WEEKLY_SPEND_RANGE = {
    "tv":           (80_000,  250_000),
    "radio":        (15_000,   60_000),
    "direct_mail":  (10_000,   40_000),
    "agent_visit":  (12_000,   45_000),
    "paid_search":  (20_000,   80_000),
    "display":      ( 5_000,   25_000),
    "paid_social":  (10_000,   50_000),
    "email":        ( 1_000,    8_000),
}

# ── TV GRP conversion factor ──────────────────────────────────────────────────
# 1 GRP ≈ 1% of target audience reached once. Typical CPP (cost-per-point) ~$8k
GRP_PER_1000_SPEND = 0.125   # 1000 USD → ~0.125 GRPs (adjustable)

# ── Radio impressions conversion factor ──────────────────────────────────────
IMPRESSIONS_PER_1000_SPEND = 18.0   # 1000 USD → ~18k impressions


def _adstock(series: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock transformation to a spend series."""
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = series[t] + decay * result[t - 1]
    return result


def _hill_saturation(x: np.ndarray, K: float, alpha: float) -> np.ndarray:
    """Hill (saturation) function: x^alpha / (x^alpha + K^alpha)."""
    xa = np.power(np.maximum(x, 0), alpha)
    Ka = np.power(K, alpha)
    return xa / (xa + Ka)


def _seasonality_curve(weeks: np.ndarray, peak_week: int = 48) -> np.ndarray:
    """
    Sinusoidal annual seasonality index between 0.6 and 1.0.
    Peak around week 48 (late Nov / early Dec — insurance renewal season).
    Secondary peak around week 14 (April — spring renewal).
    """
    primary   = 0.20 * np.sin(2 * np.pi * (weeks - peak_week) / 52)
    secondary = 0.08 * np.sin(4 * np.pi * (weeks - 14) / 52)
    return 0.80 + primary + secondary   # range ≈ [0.52, 1.08], clipped below


def _holiday_weeks(n_weeks: int, start_week_of_year: int = 1) -> np.ndarray:
    """
    Binary array marking holiday weeks.
    Holidays: Christmas (wk 51-52), New Year (wk 1), Diwali (wk 44),
    Independence Day (wk 28), Holi (wk 11).
    """
    holiday_weeks_of_year = {1, 11, 28, 44, 51, 52}
    result = np.zeros(n_weeks, dtype=int)
    for i in range(n_weeks):
        woy = ((start_week_of_year + i - 1) % 52) + 1
        if woy in holiday_weeks_of_year:
            result[i] = 1
    return result


def generate_mmm_data(
    n_weeks: int = 104,          # 2 years of weekly data
    seed: int = 42,
    start_date: str = "2022-01-03",  # Monday
    base_conversions: float = 280.0,  # avg weekly baseline (pre-media)
    noise_std: float = 0.08,          # multiplicative noise σ
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic weekly MMM dataset.

    Parameters
    ----------
    n_weeks : int
        Number of weeks to generate (default 104 = 2 years).
    seed : int
        Random seed for reproducibility.
    start_date : str
        First Monday of the dataset (ISO format).
    base_conversions : float
        Average weekly conversions from base (no media).
    noise_std : float
        Std of multiplicative noise on conversions.

    Returns
    -------
    df : pd.DataFrame
        Weekly MMM dataset with all features and targets.
    meta : dict
        Ground-truth information for model validation:
        - 'true_contributions': dict of channel → % share
        - 'adstock_decay': dict of channel → decay rate
        - 'saturation_params': dict of channel → (K, alpha)
        - 'channel_weekly_conversions': dict of channel → np.array
    """
    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    # ── Date index ────────────────────────────────────────────────────────────
    dates = [start_dt + timedelta(weeks=i) for i in range(n_weeks)]
    week_of_year = np.array([(d.isocalendar()[1]) for d in dates])
    iso_weeks = [f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}" for d in dates]

    # ── Seasonality & external variables ─────────────────────────────────────
    season_idx = _seasonality_curve(week_of_year)
    season_idx = np.clip(season_idx, 0.5, 1.15)

    holiday_flag = _holiday_weeks(n_weeks, start_week_of_year=week_of_year[0])

    # Competitor spend: follows inverse pattern (higher when we spend less)
    comp_base = rng.normal(60_000, 12_000, n_weeks)
    comp_spend = np.clip(comp_base + 15_000 * np.sin(2 * np.pi * week_of_year / 52), 30_000, 110_000)

    # Macro index: slow-moving economic indicator (0.85 – 1.10)
    macro = 0.95 + 0.10 * np.sin(2 * np.pi * np.arange(n_weeks) / 104) + rng.normal(0, 0.02, n_weeks)
    macro = np.clip(macro, 0.80, 1.15)

    # ── Spend generation per channel ─────────────────────────────────────────
    spend = {}
    for ch, (lo, hi) in WEEKLY_SPEND_RANGE.items():
        # Add a realistic trend: spend tends to increase in Q4 (weeks 40–52)
        base_spend = rng.uniform(lo, hi, n_weeks)
        q4_boost = np.where((week_of_year >= 40) & (week_of_year <= 52), 1.20, 1.0)
        # Add some flight pattern (not every week is active for offline)
        if ch in ("tv", "radio", "direct_mail"):
            # Offline channels are "flighted" — off air ~20% of weeks
            on_air = rng.random(n_weeks) > 0.18
            spend[ch] = base_spend * q4_boost * on_air
        else:
            spend[ch] = base_spend * q4_boost
        spend[ch] = np.round(spend[ch], 2)

    # ── Derived offline metrics ───────────────────────────────────────────────
    tv_grp           = np.round(spend["tv"] * GRP_PER_1000_SPEND / 1000, 2)
    radio_impressions= np.round(spend["radio"] * IMPRESSIONS_PER_1000_SPEND / 1000, 1)  # in 000s
    direct_mail_pcs  = np.round(spend["direct_mail"] / 0.45, 0).astype(int)  # ~$0.45/piece
    agent_visits_cnt = np.round(spend["agent_visit"] / 95, 0).astype(int)    # $95 CPT from MTA

    # ── Adstock transformation ────────────────────────────────────────────────
    adstock = {}
    for ch in spend:
        adstock[ch] = _adstock(spend[ch], ADSTOCK_DECAY[ch])

    # ── Saturation transformation ─────────────────────────────────────────────
    saturated = {}
    for ch in spend:
        saturated[ch] = _hill_saturation(adstock[ch], SATURATION_K[ch], SATURATION_ALPHA[ch])

    # ── Conversions from media contributions ──────────────────────────────────
    # True total conversion budget = base_conversions * season * macro * holiday_lift
    # Each channel gets a slice proportional to its true contribution share,
    # scaled by its saturated media variable.

    # Normalise saturated values so we can use them as weights
    ch_list = list(spend.keys())
    sat_matrix = np.column_stack([saturated[ch] for ch in ch_list])

    # Scale each column so mean = 1.0 (so contributions are interpretable)
    sat_scaled = sat_matrix / (sat_matrix.mean(axis=0) + 1e-9)

    channel_conversions = {}
    for i, ch in enumerate(ch_list):
        true_share = CHANNEL_TRUE_CONTRIBUTION[ch]
        channel_conversions[ch] = (
            base_conversions * true_share * sat_scaled[:, i]
        )

    # Base conversions (unexplained / brand equity)
    base_conv = base_conversions * CHANNEL_TRUE_CONTRIBUTION["base"] * np.ones(n_weeks)

    # Total conversions before external effects
    total_media_conv = sum(channel_conversions.values()) + base_conv

    # Apply seasonality, macro, holiday lift, competitor drag
    total_conv = (
        total_media_conv
        * season_idx
        * macro
        * (1 + 0.10 * holiday_flag)          # +10% on holiday weeks
        * (1 - 0.05 * (comp_spend / 80_000)) # competitor drag ~5%
    )

    # Add multiplicative noise
    noise = rng.lognormal(0, noise_std, n_weeks)
    total_conv = np.round(total_conv * noise).astype(int)
    total_conv = np.maximum(total_conv, 0)

    # Revenue: avg deal value $1,200 with some variability
    avg_deal_value = rng.normal(1200, 80, n_weeks)
    revenue = np.round(total_conv * avg_deal_value, 2)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Time index
        "week_start_date": dates,
        "iso_week":        iso_weeks,
        "week_of_year":    week_of_year,

        # ── Offline channel inputs ──
        "tv_grp":               tv_grp,
        "tv_spend":             spend["tv"],
        "radio_impressions_000":radio_impressions,
        "radio_spend":          spend["radio"],
        "direct_mail_pieces":   direct_mail_pcs,
        "direct_mail_spend":    spend["direct_mail"],
        "agent_visits":         agent_visits_cnt,
        "agent_visit_spend":    spend["agent_visit"],

        # ── Online channel inputs ──
        "paid_search_spend":    spend["paid_search"],
        "display_spend":        spend["display"],
        "paid_social_spend":    spend["paid_social"],
        "email_spend":          spend["email"],

        # ── External / control variables ──
        "seasonality_index":    np.round(season_idx, 4),
        "is_holiday_week":      holiday_flag,
        "competitor_spend":     np.round(comp_spend, 2),
        "macro_index":          np.round(macro, 4),

        # ── Adstock-transformed features (MMM-ready) ──
        "tv_adstock":           np.round(adstock["tv"], 2),
        "radio_adstock":        np.round(adstock["radio"], 2),
        "direct_mail_adstock":  np.round(adstock["direct_mail"], 2),
        "agent_visit_adstock":  np.round(adstock["agent_visit"], 2),
        "paid_search_adstock":  np.round(adstock["paid_search"], 2),
        "display_adstock":      np.round(adstock["display"], 2),
        "paid_social_adstock":  np.round(adstock["paid_social"], 2),
        "email_adstock":        np.round(adstock["email"], 2),

        # ── Target variables ──
        "conversions": total_conv,
        "revenue":     revenue,
    })

    # ── Meta: ground truth for validation ─────────────────────────────────────
    meta = {
        "true_contributions":     CHANNEL_TRUE_CONTRIBUTION,
        "adstock_decay":          ADSTOCK_DECAY,
        "saturation_params":      {ch: (SATURATION_K[ch], SATURATION_ALPHA[ch]) for ch in ch_list},
        "channel_weekly_conversions": channel_conversions,
        "n_weeks":                n_weeks,
        "base_conversions_weekly": base_conversions,
        "noise_std":              noise_std,
    }

    return df, meta


def mmm_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of key MMM variables:
    total spend, avg weekly spend, total conversions contribution (approx).
    Useful for a quick sanity-check display in the Streamlit app.
    """
    spend_cols = {
        "TV":           "tv_spend",
        "Radio":        "radio_spend",
        "Direct Mail":  "direct_mail_spend",
        "Agent Visit":  "agent_visit_spend",
        "Paid Search":  "paid_search_spend",
        "Display":      "display_spend",
        "Paid Social":  "paid_social_spend",
        "Email":        "email_spend",
    }
    rows = []
    for label, col in spend_cols.items():
        total   = df[col].sum()
        avg_wk  = df[col].mean()
        active_wks = (df[col] > 0).sum()
        rows.append({
            "Channel":         label,
            "Type":            "Offline" if label in ("TV","Radio","Direct Mail","Agent Visit") else "Online",
            "Total Spend ($)": round(total, 0),
            "Avg Weekly Spend ($)": round(avg_wk, 0),
            "Active Weeks":    active_wks,
        })
    return pd.DataFrame(rows)


def get_channel_contribution_shares(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """
    Compute the approximate contribution share of each channel to total conversions
    using the ground-truth DGP weights. Used for hybrid MTA+MMM comparison.
    Returns a DataFrame with columns: channel, mmm_contribution_pct
    """
    true_shares = meta["true_contributions"]
    rows = []
    for ch, share in true_shares.items():
        if ch == "base":
            continue
        rows.append({
            "channel":              ch,
            "mmm_contribution_pct": round(share * 100, 2),
            "channel_type":         "Offline" if ch in ("tv","radio","direct_mail","agent_visit") else "Online",
        })
    total_pct = sum(r["mmm_contribution_pct"] for r in rows)
    for r in rows:
        r["mmm_contribution_pct_normalized"] = round(r["mmm_contribution_pct"] / total_pct * 100, 2)
    return pd.DataFrame(rows).sort_values("mmm_contribution_pct", ascending=False)


if __name__ == "__main__":
    # Quick smoke test
    df, meta = generate_mmm_data(n_weeks=104, seed=42)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['week_start_date'].iloc[0].date()} → {df['week_start_date'].iloc[-1].date()}")
    print(f"\nConversions — min: {df['conversions'].min()}, max: {df['conversions'].max()}, mean: {df['conversions'].mean():.1f}")
    print(f"Revenue     — total: ${df['revenue'].sum():,.0f}")
    print(f"\nGround-truth contributions:")
    for ch, share in meta['true_contributions'].items():
        print(f"  {ch:15s}: {share*100:.1f}%")
    print(f"\nSummary stats:")
    print(mmm_summary_stats(df).to_string(index=False))
