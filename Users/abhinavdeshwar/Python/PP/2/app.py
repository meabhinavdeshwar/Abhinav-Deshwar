"""Main entry point for the Dash forecasting app."""

from __future__ import annotations

import dash
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc

from callbacks import register_callbacks
from models import FORECAST_FUNCS, PMDARIMA_AVAILABLE

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

header = html.H2("Supply Chain Demand Forecasting", className="text-center my-4")

card1 = dbc.Card(
    [
        dbc.CardHeader("1. Enter Historical Demand"),
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("Year"),
                        dbc.Input(id="input-year", type="number", min=2000, step=1),
                    ], className="mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Month"),
                        dbc.Input(id="input-month", type="number", min=1, max=12, step=1),
                    ], className="mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Demand"),
                        dbc.Input(id="input-demand", type="number", min=0, step=1),
                    ], className="mb-2"),
                    dbc.Button("Add Row", id="add-row", className="btn-add me-2"),
                    dbc.Button("Clear All", id="clear-rows", className="btn-clear"),
                ], md=4),
                dbc.Col([
                    dash_table.DataTable(
                        id="history-table",
                        columns=[{"name": c, "id": c} for c in ["year", "month", "demand"]],
                        data=[],
                        row_deletable=True,
                        style_table={"height": "300px", "overflowY": "auto"},
                        style_cell={"textAlign": "center"},
                        className="table table-sm table-striped",
                    ),
                    dcc.Graph(id="history-bar", style={"height": "200px"}),
                ], md=8),
            ])
        ),
        dcc.Store(id="row-store", data=[]),
    ],
    className="step-card card step-1",
)

models_options = [
    {"label": name, "value": name} for name in FORECAST_FUNCS.keys()
]
card2 = dbc.Card(
    [
        dbc.CardHeader("2. Select Forecasting Models"),
        dbc.CardBody(
            dcc.Dropdown(
                id="model-options",
                options=models_options,
                multi=True,
                placeholder="Select models",
                value=list(FORECAST_FUNCS.keys()) if False else [],
                disabled=False,
            )
        ),
    ],
    className="step-card card step-2",
)

card3 = dbc.Card(
    [
        dbc.CardHeader("3. Configure Parameters"),
        dbc.CardBody(
            dbc.Row([
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("Horizon"),
                    dbc.Input(id="input-horizon", type="number", min=1, value=3)
                ]), md=4),
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("Lead Time"),
                    dbc.Input(id="input-lead", type="number", min=0, value=1)
                ]), md=4),
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText("Safety Stock"),
                    dbc.Input(id="input-safety", type="number", min=0, value=0)
                ]), md=4),
            ], className="mb-2"),
            dbc.Button("Run Forecast", id="run-forecast", className="btn btn-primary")
        ),
    ],
    className="step-card card step-3",
)

card4 = dbc.Card(
    [
        dbc.CardHeader("4. Results"),
        dbc.CardBody(
            [
                dcc.Graph(id="results-graph"),
                dash_table.DataTable(
                    id="results-table",
                    columns=[{"name": c, "id": c} for c in ["Model", "MAE", "MSE", "MAPE %", "Best"]],
                    data=[],
                    style_cell={"textAlign": "center"},
                    className="table table-sm table-striped mb-2",
                ),
                html.Div(id="summary"),
            ]
        ),
    ],
    className="step-card card step-4",
)

app.layout = dbc.Container([header, card1, card2, card3, card4], fluid=True)

register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
