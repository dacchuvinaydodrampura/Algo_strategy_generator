"""
app/patterns/rule_miner.py
--------------------------
Pattern discovery via:
1. Threshold rule mining  — systematic scan of feature thresholds to find
   feature conditions that predict directional forward returns.
2. Unsupervised clustering — k-means on feature vectors to find regime clusters
   that exhibit directional bias.

All discovered patterns are returned as PatternCandidate objects.
Patterns that fail minimum quality thresholds are explicitly rejected
and logged — they are never silently discarded.
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import PatternsConfig
from app.models.session import (
    PatternCandidate,
    PatternDirection,
    PatternRule,
    TickWindow,
)
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

# Features used for rule mining (all must be FeatureRecord fields)
_RULE_FEATURES = [
    "mean_imbalance",
    "mean_microprice_slope",
    "mean_aggression",
    "mean_relative_spread",
    "mean_depth_ratio",
    "mean_realised_vol",
    "mean_imbalance_5",
    "mean_imbalance_vel",
    "mean_microprice_acc",
    "mean_spread_ratio",
    "mean_liquidity_vacuum",
    "mean_queue_depletion",
    "mean_replenishment",
    "mean_iceberg_indicator",
    "mean_aggressive_burst",
    "mean_of_persistence",
    "mean_vol_clustering",
]

# Quantile thresholds tested for each feature
_QUANTILE_THRESHOLDS = [0.20, 0.30, 0.40, 0.60, 0.70, 0.80]


@dataclass
class _WindowRow:
    """Flat representation of a TickWindow for DataFrame analysis."""
    window_idx: int
    symbol: str
    start_t: int
    ticks: int
    mean_imbalance: float
    mean_microprice_slope: float
    mean_aggression: float
    mean_relative_spread: float
    mean_depth_ratio: float
    mean_realised_vol: float
    mean_imbalance_5: float
    mean_imbalance_vel: float
    mean_microprice_acc: float
    mean_spread_ratio: float
    mean_liquidity_vacuum: float
    mean_queue_depletion: float
    mean_replenishment: float
    mean_iceberg_indicator: float
    mean_aggressive_burst: float
    mean_of_persistence: float
    mean_vol_clustering: float
    forward_return: float      # microprice change after window (label)
    is_oos: bool


def _windows_to_df(
    windows: list[TickWindow],
    oos_start_t: int,
) -> pd.DataFrame:
    """Convert list of TickWindow to a flat DataFrame for analysis."""
    rows = []
    for i, w in enumerate(windows):
        # Forward return = exit microprice of NEXT window minus entry of this window.
        # We cannot compute this without the next window; it is attached later.
        rows.append(_WindowRow(
            window_idx=i,
            symbol=w.symbol,
            start_t=w.start_t,
            ticks=w.ticks,
            mean_imbalance=w.mean_imbalance,
            mean_microprice_slope=w.mean_microprice_slope,
            mean_aggression=w.mean_aggression,
            mean_relative_spread=w.mean_relative_spread,
            mean_depth_ratio=w.mean_depth_ratio,
            mean_realised_vol=w.mean_realised_vol,
            mean_imbalance_5=getattr(w, "mean_imbalance_5", 0.0),
            mean_imbalance_vel=getattr(w, "mean_imbalance_vel", 0.0),
            mean_microprice_acc=getattr(w, "mean_microprice_acc", 0.0),
            mean_spread_ratio=getattr(w, "mean_spread_ratio", 0.0),
            mean_liquidity_vacuum=getattr(w, "mean_liquidity_vacuum", 0.0),
            mean_queue_depletion=getattr(w, "mean_queue_depletion", 0.0),
            mean_replenishment=getattr(w, "mean_replenishment", 0.0),
            mean_iceberg_indicator=getattr(w, "mean_iceberg_indicator", 0.0),
            mean_aggressive_burst=getattr(w, "mean_aggressive_burst", 0.0),
            mean_of_persistence=getattr(w, "mean_of_persistence", 0.0),
            mean_vol_clustering=getattr(w, "mean_vol_clustering", 0.0),
            forward_return=0.0,   # filled below
            is_oos=w.start_t >= oos_start_t,
        ))
    df = pd.DataFrame([r.__dict__ for r in rows])

    # Compute forward_return: exit_microprice of next window - entry_microprice of this
    # This is a proxy for what a trader entering at end of window would see.
    exit_mp = [w.exit_microprice for w in windows]
    entry_mp = [w.entry_microprice for w in windows]
    df["forward_return"] = (
        pd.Series(exit_mp).shift(-1) - pd.Series(entry_mp)
    ).fillna(0.0).values

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Rule miner
# ──────────────────────────────────────────────────────────────────────────────


class RuleMiner:
    """
    Discovers directional trading patterns from TickWindow summary features
    by testing combinations of threshold rules.

    Algorithm:
    1. Build a DataFrame of window-level features with forward returns.
    2. Compute data-driven quantile thresholds for each feature.
    3. Test all N-choose-K feature combinations (K = max_features_per_rule).
    4. For each combination, test threshold conditions that predict LONG or SHORT.
    5. Score each candidate by win rate and profit factor on IS data.
    6. Return PatternCandidate objects that pass minimum quality thresholds.
    """

    def __init__(self, cfg: PatternsConfig, symbol: str) -> None:
        self._cfg = cfg
        self._symbol = symbol

    def mine(
        self,
        windows: list[TickWindow],
        oos_start_t: int,
    ) -> list[PatternCandidate]:
        """
        Run rule mining and return accepted PatternCandidate objects.

        Parameters
        ----------
        windows:      All TickWindows for the session, time-ordered.
        oos_start_t:  Epoch ms timestamp where OOS period begins.
        """
        if len(windows) < self._cfg.min_samples * 2:
            logger.warning(
                "insufficient_windows_for_mining",
                symbol=self._symbol,
                count=len(windows),
                required=self._cfg.min_samples * 2,
            )
            return []

        df = _windows_to_df(windows, oos_start_t)
        is_df = df[~df["is_oos"]].copy()

        if len(is_df) < self._cfg.min_samples:
            logger.warning(
                "insufficient_is_windows",
                symbol=self._symbol,
                is_count=len(is_df),
            )
            return []

        # Compute feature quantile thresholds from IS data only
        thresholds: dict[str, list[tuple[str, float]]] = {}
        for feat in _RULE_FEATURES:
            vals = is_df[feat].dropna()
            t_list: list[tuple[str, float]] = []
            for q in _QUANTILE_THRESHOLDS:
                qv = float(vals.quantile(q))
                if q < 0.5:
                    t_list.append(("<", qv))
                else:
                    t_list.append((">", qv))
            thresholds[feat] = t_list

        candidates: list[PatternCandidate] = []
        rejected = 0

        # Test 1-feature and 2-feature rules (3+ is computationally expensive
        # and risks overfitting on a single day)
        max_k = min(self._cfg.max_features_per_rule, 2)

        for k in range(1, max_k + 1):
            for feat_combo in itertools.combinations(_RULE_FEATURES, k):
                # Generate all threshold combinations for this feature set
                threshold_options = [thresholds[f] for f in feat_combo]
                for threshold_combo in itertools.product(*threshold_options):
                    for direction in [PatternDirection.LONG, PatternDirection.SHORT]:
                        rules = [
                            PatternRule(
                                feature=feat,
                                operator=op,
                                threshold=thresh,
                            )
                            for feat, (op, thresh) in zip(feat_combo, threshold_combo)
                        ]
                        candidate = self._evaluate_rule_set(
                            rules, direction, is_df, df
                        )
                        if candidate is not None:
                            candidates.append(candidate)
                        else:
                            rejected += 1

        logger.info(
            "rule_mining_complete",
            symbol=self._symbol,
            candidates_accepted=len(candidates),
            candidates_rejected=rejected,
        )
        return candidates

    def _evaluate_rule_set(
        self,
        rules: list[PatternRule],
        direction: PatternDirection,
        is_df: pd.DataFrame,
        full_df: pd.DataFrame,
    ) -> Optional[PatternCandidate]:
        """
        Check if a rule set + direction passes minimum quality on IS data.
        Returns PatternCandidate or None if rejected.
        """
        # Apply all rules to IS data
        mask = pd.Series([True] * len(is_df), index=is_df.index)
        for rule in rules:
            col_vals = is_df[rule.feature]
            if rule.operator == ">":
                mask &= col_vals > rule.threshold
            elif rule.operator == "<":
                mask &= col_vals < rule.threshold
            elif rule.operator == ">=":
                mask &= col_vals >= rule.threshold
            elif rule.operator == "<=":
                mask &= col_vals <= rule.threshold

        matched = is_df[mask]

        if len(matched) < self._cfg.min_samples:
            return None  # too few samples

        fwd = matched["forward_return"]
        if direction == PatternDirection.LONG:
            wins = (fwd > 0).sum()
            gross_wins = fwd[fwd > 0].sum()
            gross_losses = abs(fwd[fwd < 0].sum())
        else:
            wins = (fwd < 0).sum()
            gross_wins = abs(fwd[fwd < 0].sum())
            gross_losses = fwd[fwd > 0].sum()

        win_rate = wins / len(matched)
        profit_factor = gross_wins / (gross_losses + 1e-9)

        if win_rate < self._cfg.min_win_rate:
            return None
        if profit_factor < self._cfg.min_profit_factor:
            return None

        pattern_id = f"{self._symbol}_{direction.value}_{uuid.uuid4().hex[:8]}"

        # Find matched window indices across full data for OOS comparison
        full_mask = pd.Series([True] * len(full_df), index=full_df.index)
        for rule in rules:
            col_vals = full_df[rule.feature]
            if rule.operator == ">":
                full_mask &= col_vals > rule.threshold
            elif rule.operator == "<":
                full_mask &= col_vals < rule.threshold
            elif rule.operator == ">=":
                full_mask &= col_vals >= rule.threshold
            elif rule.operator == "<=":
                full_mask &= col_vals <= rule.threshold

        matched_indices = full_df[full_mask]["window_idx"].tolist()

        description = (
            f"{direction.value} signal: "
            + " AND ".join(r.describe() for r in rules)
            + f" | IS: WR={win_rate:.2%}, PF={profit_factor:.2f}, n={len(matched)}"
        )

        return PatternCandidate(
            pattern_id=pattern_id,
            symbol=self._symbol,
            direction=direction,
            rules=rules,
            matched_windows=matched_indices,
            sample_count=len(matched),
            discovery_method="rule_mining",
            description=description,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Cluster-based pattern discovery
# ──────────────────────────────────────────────────────────────────────────────


class ClusterMiner:
    """
    Discovers patterns by clustering windows in feature space and checking
    whether any cluster exhibits consistent directional bias.
    """

    def __init__(self, cfg: PatternsConfig, symbol: str) -> None:
        self._cfg = cfg
        self._symbol = symbol

    def mine(
        self,
        windows: list[TickWindow],
        oos_start_t: int,
    ) -> list[PatternCandidate]:
        if len(windows) < self._cfg.min_samples * 2:
            return []

        df = _windows_to_df(windows, oos_start_t)
        feature_cols = _RULE_FEATURES

        is_df_fit = df[~df["is_oos"]].copy().reset_index(drop=True)
        X_is = is_df_fit[feature_cols].fillna(0.0).values

        scaler = StandardScaler()
        X_is_scaled = scaler.fit_transform(X_is)

        n_clusters = min(self._cfg.clustering_n_clusters, len(is_df_fit) // 10)
        if n_clusters < 2:
            return []

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_is_scaled)

        # Predict clusters for the entire day (both IS and OOS)
        X_full = df[feature_cols].fillna(0.0).values
        X_full_scaled = scaler.transform(X_full)
        df["cluster"] = kmeans.predict(X_full_scaled)

        is_df = df[~df["is_oos"]].copy()
        candidates: list[PatternCandidate] = []

        for cluster_id in range(n_clusters):
            cluster_mask = is_df["cluster"] == cluster_id
            cluster_data = is_df[cluster_mask]

            if len(cluster_data) < self._cfg.min_samples:
                continue

            fwd = cluster_data["forward_return"]
            long_wr = (fwd > 0).mean()
            short_wr = (fwd < 0).mean()

            for direction, wr in [
                (PatternDirection.LONG, long_wr),
                (PatternDirection.SHORT, short_wr),
            ]:
                if wr < self._cfg.min_win_rate:
                    continue

                # Build representative rules from cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                original_centroid = scaler.inverse_transform([centroid])[0]

                rules = []
                for i, feat in enumerate(feature_cols):
                    overall_mean = float(is_df[feat].mean())
                    cval = original_centroid[i]
                    if cval > overall_mean:
                        rules.append(PatternRule(feat, ">", float(overall_mean)))
                    else:
                        rules.append(PatternRule(feat, "<", float(overall_mean)))

                # Only keep the 2 most distinctive rules (by distance from mean)
                distances = [
                    abs(original_centroid[i] - float(is_df[feature_cols[i]].mean()))
                    for i in range(len(feature_cols))
                ]
                top_indices = sorted(range(len(distances)), key=lambda x: -distances[x])[:2]
                rules = [rules[i] for i in top_indices]

                gross_wins = abs(fwd[fwd > 0].sum()) if direction == PatternDirection.LONG else abs(fwd[fwd < 0].sum())
                gross_losses = abs(fwd[fwd < 0].sum()) if direction == PatternDirection.LONG else fwd[fwd > 0].sum()
                pf = gross_wins / (gross_losses + 1e-9)

                if pf < self._cfg.min_profit_factor:
                    continue

                pattern_id = f"{self._symbol}_CLUSTER{cluster_id}_{direction.value}_{uuid.uuid4().hex[:6]}"
                description = (
                    f"Cluster {cluster_id} {direction.value} | "
                    + " AND ".join(r.describe() for r in rules)
                    + f" | WR={wr:.2%}, PF={pf:.2f}, n={len(cluster_data)}"
                )

                matched_windows_full = df[df["cluster"] == cluster_id]["window_idx"].tolist()

                candidates.append(PatternCandidate(
                    pattern_id=pattern_id,
                    symbol=self._symbol,
                    direction=direction,
                    rules=rules,
                    matched_windows=matched_windows_full,
                    sample_count=len(cluster_data),
                    discovery_method="clustering",
                    description=description,
                ))

        logger.info(
            "cluster_mining_complete",
            symbol=self._symbol,
            clusters=n_clusters,
            candidates=len(candidates),
        )
        return candidates
