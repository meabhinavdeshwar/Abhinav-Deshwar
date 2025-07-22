"""Dash callbacks for interactive forecasting."""

from __future__ import annotations

import dash
from dash import Dash, Input, Output, State, dash_table, html, dcc
import plotly.graph_objs as go
import pandas as pd

from models import FORECAST_FUNCS, mae, mse, mape
from utils import parse_rows, validate_history, reorder_point


# helper to create toast messages (not implemented; placeholder)
def create_toast(message: str, color: str = "primary") -> html.Div:
    return html.Div(message, className=f"alert alert-{color}")


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("history-table", "data"),
        Output("history-bar", "figure"),
        Output("row-store", "data"),
        Input("add-row", "n_clicks"),
        Input("clear-rows", "n_clicks"),
        State("input-year", "value"),
        State("input-month", "value"),
        State("input-demand", "value"),
        State("row-store", "data"),
        prevent_initial_call=True,
    )
    def update_rows(add_clicks, clear_clicks, year, month, demand, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        action = ctx.triggered[0]["prop_id"].split(".")[0]
        rows = data or []
        if action == "add-row" and year and month and demand is not None:
            rows.append({"year": int(year), "month": int(month), "demand": float(demand)})
        elif action == "clear-rows":
            rows = []
        df = parse_rows(rows)
        fig = go.Figure(data=[go.Bar(x=df["date"], y=df["demand"])]).update_layout(margin=dict(t=0, b=0))
        return rows, fig, rows

    @app.callback(
        Output("model-options", "options"),
        Output("model-options", "disabled"),
        Input("model-options", "id"),
        prevent_initial_call=False,
    )
    def toggle_arima(_):
        options = [{"label": name, "value": name} for name in FORECAST_FUNCS.keys()]
        disabled = False
        return options, disabled

    @app.callback(
        Output("results-graph", "figure"),
        Output("results-table", "data"),
        Output("summary", "children"),
        Input("run-forecast", "n_clicks"),
        State("row-store", "data"),
        State("model-options", "value"),
        State("input-horizon", "value"),
        State("input-lead", "value"),
        State("input-safety", "value"),
        prevent_initial_call=True,
    )
    def run_forecast(_, rows, selected_models, horizon, lead_time, safety_stock):
        if not rows or not selected_models:
            raise dash.exceptions.PreventUpdate
        df = validate_history(parse_rows(rows))
        series = pd.Series(df["demand"].values, index=df["date"])
        horizon = int(horizon)
        results = []
        avg_demand = series.mean()
        for name in selected_models:
            fc, fit = FORECAST_FUNCS[name](series, horizon, return_fit=True)
            fit_index = series.index[-len(fit):]
            mae_v = mae(series.loc[fit_index], fit)
            mse_v = mse(series.loc[fit_index], fit)
            mape_v = mape(series.loc[fit_index], fit)
            results.append({
                "name": name,
                "forecast": fc,
                "mae": mae_v,
                "mse": mse_v,
                "mape": mape_v,
                "fit": fit,
            })
        best = min(results, key=lambda r: r["mape"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines+markers", name="Actual"))
        for r in results:
            line = dict(width=4 if r["name"] == best["name"] else 2)
            fig.add_trace(go.Scatter(x=r["forecast"].index, y=r["forecast"], mode="lines", name=r["name"], line=line))
        fig.update_layout(margin=dict(t=10, b=10))
        table_data = [{
            "Model": r["name"],
            "MAE": f"{r['mae']:.2f}",
            "MSE": f"{r['mse']:.2f}",
            "MAPE %": f"{r['mape']:.2f}",
            "Best": "Yes" if r["name"] == best["name"] else ""
        } for r in results]
        rop = reorder_point(best["forecast"].iloc[0], float(safety_stock), avg_demand, float(lead_time))
        summary = dcc.Markdown(f"**Best model:** {best['name']}  \n**Reorder point:** {rop:.2f}")
        return fig, table_data, summary

