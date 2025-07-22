"""Forecasting models and metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import Series
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

try:
    from pmdarima import auto_arima  # type: ignore
    PMDARIMA_AVAILABLE = True
except Exception:  # pragma: no cover - pmdarima optional
    PMDARIMA_AVAILABLE = False


@dataclass
class ForecastResult:
    name: str
    forecast: Series
    mae: float
    mse: float
    mape: float
    fit: Series | None = None


def mae(y_true: Series, y_pred: Series) -> float:
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: Series, y_pred: Series) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mape(y_true: Series, y_pred: Series) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def sma(series: Series, window: int, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    fit = series.rolling(window=window).mean()
    last = fit.iloc[-1]
    fc = pd.Series([last] * horizon, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS"))
    return (fc, fit) if return_fit else (fc, None)


def wma(series: Series, weights: list[int], horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    w = np.array(weights)
    w = w / w.sum()
    def calc(i):
        if i < len(w):
            return np.nan
        window = series.iloc[i - len(w):i]
        return np.sum(window.values * w)
    fit = Series([calc(i) for i in range(len(series))], index=series.index)
    last = fit.dropna().iloc[-1]
    fc = pd.Series([last] * horizon, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS"))
    return (fc, fit) if return_fit else (fc, None)


def ses(series: Series, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    model = SimpleExpSmoothing(series).fit()
    fc = model.forecast(horizon)
    return (fc, model.fittedvalues) if return_fit else (fc, None)


def des(series: Series, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    model = Holt(series).fit()
    fc = model.forecast(horizon)
    return (fc, model.fittedvalues) if return_fit else (fc, None)


def trend_es(series: Series, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    fc = model.forecast(horizon)
    return (fc, model.fittedvalues) if return_fit else (fc, None)


def holt_winters(series: Series, seasonal: str, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    model = ExponentialSmoothing(series, trend="add", seasonal=seasonal, seasonal_periods=12).fit()
    fc = model.forecast(horizon)
    return (fc, model.fittedvalues) if return_fit else (fc, None)


def arima_model(series: Series, horizon: int, return_fit: bool = False) -> Tuple[Series, Series | None]:
    if not PMDARIMA_AVAILABLE:
        raise RuntimeError("pmdarima not available")
    model = auto_arima(series, seasonal=True, m=12)
    fc = pd.Series(model.predict(horizon), index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS"))
    fit = pd.Series(model.predict_in_sample(), index=series.index)
    return (fc, fit) if return_fit else (fc, None)


FORECAST_FUNCS: Dict[str, Callable[..., Tuple[Series, Series | None]]] = {
    "SMA": lambda s, h, return_fit=False: sma(s, window=3, horizon=h, return_fit=return_fit),
    "WMA": lambda s, h, return_fit=False: wma(s, weights=[1, 2, 3, 4, 5], horizon=h, return_fit=return_fit),
    "SES": ses,
    "DES": des,
    "Trend ES": trend_es,
    "HW Additive": lambda s, h, return_fit=False: holt_winters(s, "add", h, return_fit=return_fit),
    "HW Multiplicative": lambda s, h, return_fit=False: holt_winters(s, "mul", h, return_fit=return_fit),
}
if PMDARIMA_AVAILABLE:
    FORECAST_FUNCS["ARIMA"] = arima_model

