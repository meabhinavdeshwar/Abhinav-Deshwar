"""Utility functions for demand forecasting app."""

from __future__ import annotations

import pandas as pd


def parse_rows(rows: list[dict]) -> pd.DataFrame:
    """Convert list of row dicts with year, month, demand into DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["date", "demand"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df.sort_values("date", inplace=True)
    df = df.drop(columns=[col for col in ["id"] if col in df])
    df = df[["date", "demand"]]
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["demand"]).reset_index(drop=True)
    return df


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate months count and interpolate missing interior months."""
    if df.empty:
        return df
    months = len(df)
    if not 1 <= months <= 60:
        raise ValueError("History must contain between 1 and 60 months")
    full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    df = df.set_index("date").reindex(full_index)
    df.index.name = "date"
    df["demand"] = df["demand"].interpolate(method="linear")
    return df.reset_index()


def reorder_point(forecast_first: float, safety_stock: float, avg_demand: float, lead_time: float) -> float:
    """Calculate reorder point."""
    return forecast_first + safety_stock + avg_demand * lead_time
