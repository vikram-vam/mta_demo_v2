"""
hybrid_attribution.py
=====================
Bridges Marketing Mix Modelling (MMM) and Multi-Touch Attribution (MTA)
into a single unified framework.

The Problem
-----------
MTA (Shapley / Markov) can only credit channels that appear in digital
journey logs. Offline channels (TV, Radio, Direct Mail, Agent Visits)
are invisible to MTA — they get near-zero credit even though they drive
real conversions.

MMM works at aggregate level and captures offline impact, but it cannot
tell us which specific customer or journey was influenced.

The Hybrid Solution
-------------------
1. MTA handles online channel credit (per-journey granularity).
2. MMM handles offline channel sizing (weekly aggregate).
3. Hybrid model:
   a. Takes MTA Shapley weights for online channels.
   b. Takes MMM contribution shares for offline channels.
   c. Blends them with a mixing parameter α into a UNIFIED WEIGHT per channel.
   d. Converts everything to a common metric: Cost Per Incremental Conversion (CPIC)
      and Marginal ROI.

Common Metric: CPIC and Marginal ROI
--------------------------------------
CPIC  = Total Spend / Unified Conversions attributed to channel
MROI  = Revenue per Conversion / CPIC  (higher = better)

This allows TV to sit on the same chart as Paid Search and be compared fairly.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .data_generator import CHANNELS, CHANNEL_LABELS, CHANNEL_CPT, CHANNEL_TYPE


# ── Default blending weight ──────────────────────────────────────────────────
# α = weight given to MTA for online channels
# (1-α) = weight given to MMM for all channels
# For offline-heavy businesses, use lower α (e.g., 0.4).
# For digital-native businesses, use higher α (e.g., 0.75).
DEFAULT_ALPHA = 0.60


def blend_mta_mmm(
    mta_weights: Dict[str, float],
    mmm_contributions: Dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    online_channels: Optional[list] = None,
) -> Dict[str, float]:
    """
    Blend MTA Shapley weights with MMM contribution shares.

    Formula
    -------
    For ONLINE channels:
        hybrid_weight = α * mta_weight + (1 - α) * mmm_contribution
    For OFFLINE channels:
        hybrid_weight = mmm_contribution
        (MTA gives near-zero credit to offline, so we trust MMM fully)

    Parameters
    ----------
    mta_weights : dict
        Shapley/Markov attribution weights, channel → fractional credit (sum ~1).
    mmm_contributions : dict
        MMM-derived contribution shares, channel → fractional share (sum ~1).
    alpha : float
        Blending weight for MTA component (0 = pure MMM, 1 = pure MTA).
    online_channels : list, optional
        List of channels considered "online". Defaults to standard set.

    Returns
    -------
    hybrid_weights : dict
        Blended fractional weights, sum = 1.0.
    """
    if online_channels is None:
        online_channels = ["paid_search", "display", "paid_social",
                           "email", "organic_search", "direct"]

    # Normalise inputs to sum to 1.0 (defensive)
    mta_total  = sum(mta_weights.values()) or 1.0
    mmm_total  = sum(v for k, v in mmm_contributions.items() if k != "base") or 1.0
    mta_norm   = {k: v / mta_total  for k, v in mta_weights.items()}
    mmm_norm   = {k: v / mmm_total  for k, v in mmm_contributions.items() if k != "base"}

    # Make sure all channels are present in both dicts (fill missing with 0)
    all_channels = list(set(list(mta_norm.keys()) + list(mmm_norm.keys())))

    hybrid = {}
    for ch in all_channels:
        mta_w = mta_norm.get(ch, 0.0)
        mmm_w = mmm_norm.get(ch, 0.0)

        if ch in online_channels:
            hybrid[ch] = alpha * mta_w + (1 - alpha) * mmm_w
        else:
            # Offline: MTA gives ~0, trust MMM entirely
            hybrid[ch] = mmm_w

    # Re-normalise to sum to 1
    total = sum(hybrid.values()) or 1.0
    hybrid = {k: v / total for k, v in hybrid.items()}
    return hybrid


def compute_unified_metrics(
    hybrid_weights: Dict[str, float],
    total_conversions: float,
    total_revenue: float,
    channel_spend: Optional[Dict[str, float]] = None,
    avg_deal_value: float = 1200.0,
) -> pd.DataFrame:
    """
    Compute the common metric table: Unified Conversions, CPIC, and Marginal ROI
    for every channel.

    Parameters
    ----------
    hybrid_weights : dict
        Blended weights from blend_mta_mmm().
    total_conversions : float
        Total conversions in the period (from actual data or MMM estimate).
    total_revenue : float
        Total revenue in the period.
    channel_spend : dict, optional
        Actual spend per channel. Falls back to CPT * estimated touchpoints.
    avg_deal_value : float
        Average revenue per conversion (used if total_revenue not available).

    Returns
    -------
    pd.DataFrame with columns:
        channel, channel_label, channel_type, hybrid_weight,
        attributed_conversions, spend, cpic, marginal_roi
    """
    rows = []
    for ch, weight in hybrid_weights.items():
        if ch not in CHANNEL_LABELS:
            continue

        attributed_conv = total_conversions * weight
        attributed_rev  = total_revenue * weight

        # Get spend: use provided spend or estimate from CPT
        if channel_spend and ch in channel_spend:
            spend = channel_spend[ch]
        else:
            # Estimate: attributed_conv * CPT (rough proxy)
            cpt = CHANNEL_CPT.get(ch, 10)
            spend = attributed_conv * cpt

        cpic = spend / attributed_conv if attributed_conv > 0 else np.inf
        mroi = attributed_rev / spend   if spend > 0         else 0.0

        rows.append({
            "channel":               ch,
            "channel_label":         CHANNEL_LABELS.get(ch, ch),
            "channel_type":          CHANNEL_TYPE.get(ch, "Unknown"),
            "hybrid_weight":         round(weight * 100, 2),        # as %
            "attributed_conversions":round(attributed_conv, 1),
            "attributed_revenue":    round(attributed_rev, 2),
            "spend":                 round(spend, 2),
            "cpic":                  round(cpic, 2),
            "marginal_roi":          round(mroi, 3),
        })

    df = pd.DataFrame(rows).sort_values("hybrid_weight", ascending=False)
    return df


def compare_mta_vs_mmm_vs_hybrid(
    mta_weights: Dict[str, float],
    mmm_contributions: Dict[str, float],
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """
    Side-by-side comparison of MTA credits, MMM contributions,
    and hybrid blended weights for every channel.

    Returns a DataFrame perfect for a bar chart or table in Streamlit.
    """
    hybrid_weights = blend_mta_mmm(mta_weights, mmm_contributions, alpha)

    # Normalise all to % for apples-to-apples comparison
    mta_total = sum(mta_weights.values()) or 1.0
    mmm_total = sum(v for k, v in mmm_contributions.items() if k != "base") or 1.0

    rows = []
    for ch in CHANNELS:
        rows.append({
            "channel":       ch,
            "channel_label": CHANNEL_LABELS.get(ch, ch),
            "channel_type":  CHANNEL_TYPE.get(ch, "Unknown"),
            "mta_pct":       round(mta_weights.get(ch, 0) / mta_total * 100, 2),
            "mmm_pct":       round(mmm_contributions.get(ch, 0) / mmm_total * 100, 2),
            "hybrid_pct":    round(hybrid_weights.get(ch, 0) * 100, 2),
        })

    df = pd.DataFrame(rows)
    df["credit_gap"] = df["mmm_pct"] - df["mta_pct"]  # + = MTA undercredits, - = overcredits
    return df.sort_values("hybrid_pct", ascending=False)


def offline_credit_recovery(
    mta_weights: Dict[str, float],
    mmm_contributions: Dict[str, float],
) -> Dict:
    """
    Quantify how much offline channels are undercredited by MTA alone.
    Returns a summary dict useful for annotations in the Streamlit UI.
    """
    offline_chs = ["tv", "radio", "direct_mail", "agent_visit"]
    mta_total   = sum(mta_weights.values()) or 1.0
    mmm_total   = sum(v for k, v in mmm_contributions.items() if k != "base") or 1.0

    mta_offline_pct = sum(mta_weights.get(ch, 0) for ch in offline_chs) / mta_total * 100
    mmm_offline_pct = sum(mmm_contributions.get(ch, 0) for ch in offline_chs) / mmm_total * 100

    return {
        "mta_offline_credit_pct":  round(mta_offline_pct, 1),
        "mmm_offline_credit_pct":  round(mmm_offline_pct, 1),
        "undercredit_gap_pct":     round(mmm_offline_pct - mta_offline_pct, 1),
        "mta_online_credit_pct":   round(100 - mta_offline_pct, 1),
        "mmm_online_credit_pct":   round(100 - mmm_offline_pct, 1),
    }
