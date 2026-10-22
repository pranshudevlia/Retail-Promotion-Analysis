"""Small, shared data and plotting helpers for the notebook workflow."""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42


def project_root(start=None):
    candidate = (Path.cwd() if start is None else Path(start)).resolve()
    for folder in (candidate, *candidate.parents):
        if (folder / "README.md").exists():
            return folder
    raise FileNotFoundError("Could not locate README.md.")


def add_promotion_fields(transactions):
    """Derive observed discount fields without changing raw discount columns."""
    frame = transactions.copy()
    discount_columns = ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]
    frame["discount_amount"] = -frame[discount_columns].clip(upper=0).sum(axis=1)
    frame["gross_value_proxy"] = frame["SALES_VALUE"] + frame["discount_amount"]
    frame["discount_rate"] = np.where(
        frame["gross_value_proxy"] > 0,
        frame["discount_amount"] / frame["gross_value_proxy"],
        np.nan,
    )
    retail = frame["RETAIL_DISC"] < 0
    coupon = (frame["COUPON_DISC"] < 0) | (frame["COUPON_MATCH_DISC"] < 0)
    frame["promotion_type"] = np.select(
        [retail & coupon, coupon, retail],
        ["Retail + coupon", "Coupon only", "Retail only"],
        default="No recorded discount",
    )
    frame["is_discounted"] = frame["discount_amount"] > 0
    return frame


def add_bar_labels(ax, percentage=False):
    """Label every finite bar."""
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            elif percentage:
                labels.append(f"{value:.2f}%")
            else:
                labels.append(f"{value:,.2f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)


def bootstrap_ci(values, statistic=np.median, confidence=0.95, iterations=2000, random_state=42):
    """Percentile bootstrap interval for a one-sample statistic."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(random_state)
    estimates = np.empty(iterations)
    for index in range(iterations):
        estimates[index] = statistic(rng.choice(array, size=len(array), replace=True))
    tail = (1 - confidence) / 2
    return tuple(np.quantile(estimates, [tail, 1 - tail]))


def summarize_frame(name, frame, numeric_column, key=None):
    """Print the five mandatory cleaning checks with explicit labels."""
    print(f"{name} row count: {frame.shape[0]:,}")
    print(f"{name} column count: {frame.shape[1]:,}")
    print(f"{name} null counts:")
    print(frame.isna().sum().to_string())
    print(f"{name} exact duplicate count: {frame.duplicated().sum():,}")
    if key is not None:
        keys = [key] if isinstance(key, str) else list(key)
        print(f"{name} duplicate count on verified key {keys}: {frame.duplicated(keys).sum():,}")
    print(f"{name} DataFrame info:")
    frame.info()
    print(f"{name} descriptive statistics for {numeric_column}:")
    print(frame[numeric_column].describe().to_string())
