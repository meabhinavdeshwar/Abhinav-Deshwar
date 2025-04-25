# app.py

import dash
import random  # For random use in Gantt chart
import dash_bootstrap_components as dbc
import dash_table
from dash import dcc, html, Input, Output, State, callback
from datetime import timedelta, datetime
import numpy as np
import pandas as pd

# Auth functions
from auth import load_users, save_users, hash_password

# About layout
from about import about_layout

# Models functions
from models import (
    train_main_models,        # Train main models
    train_process_models,
    rf_predict_delay,
    predict_process_delays,
    process_list
)

# Simulation logic and other components
from simulation import (
    simulation_state,
    flights_data,
    dynamic_positions,
    run_digital_twin_simulation,
    update_positions,
    mapbox_token,
    airport_center,
    runway_polygons,
    terminal_polygons,
    misconnect_color,
    status_colors,
    navbar,
    home_layout,
    manual_layout,
    MODEL_USED,           
    MAIN_MODEL_RMSE,      
    PROCESS_MODEL_RMSE,   
    crew_factor_for,
    maintenance_factor_for
)

##############################################################################
# 1) Train ML Models
##############################################################################
best_main_model, model_scores = train_main_models(1500)
process_models = train_process_models()

##############################################################################
# 2) Enhanced Ultra Detailed Model Info Page
##############################################################################
import plotly.express as px

def make_model_score_table(scores_dict):
    """
    Creates a score table from the model_scores dictionary.
    """
    rows = []
    for model_name, info in scores_dict.items():
        rows.append({
            "Model": model_name,
            "RMSE": f"{info['rmse']:.2f}",
            "MAPE (%)": f"{info['mape']:.2f}",
            "MAD": f"{info['mad']:.2f}"
        })
    return dash_table.DataTable(
        columns=[
            {"name": "Model", "id": "Model"},
            {"name": "RMSE", "id": "RMSE"},
            {"name": "MAPE (%)", "id": "MAPE (%)"},
            {"name": "MAD", "id": "MAD"}
        ],
        data=rows,
        style_cell={"textAlign": "center", "padding": "6px", "fontSize": "15px"},
        style_header={"backgroundColor": "#f0f8ff", "fontWeight": "bold"},
        style_table={"margin": "auto"},
        style_as_list_view=True
    )

def make_rmse_bar_chart(scores_dict):
    """
    Creates a pastel-colored bar chart for RMSE.
    """
    chart_data = [{"Model": model_name, "RMSE": info["rmse"]} for model_name, info in scores_dict.items()]
    df_chart = pd.DataFrame(chart_data)
    fig = px.bar(
        df_chart,
        x="Model",
        y="RMSE",
        template="plotly_white",
        labels={"RMSE": "RMSE Score", "Model": "Model Name"},
        title="Model Comparison: RMSE",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title="Models", yaxis_title="RMSE", margin=dict(l=40, r=40, t=50, b=40))
    return dcc.Graph(figure=fig)

def make_mape_bar_chart(scores_dict):
    """
    Creates a pastel-colored bar chart for MAPE.
    """
    chart_data = [{"Model": model_name, "MAPE": info["mape"]} for model_name, info in scores_dict.items()]
    df_chart = pd.DataFrame(chart_data)
    fig = px.bar(
        df_chart,
        x="Model",
        y="MAPE",
        template="plotly_white",
        labels={"MAPE": "MAPE (%)", "Model": "Model Name"},
        title="Model Comparison: MAPE (%)",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title="Models", yaxis_title="MAPE (%)", margin=dict(l=40, r=40, t=50, b=40))
    return dcc.Graph(figure=fig)

def make_mad_bar_chart(scores_dict):
    """
    Creates a pastel-colored bar chart for MAD.
    """
    chart_data = [{"Model": model_name, "MAD": info["mad"]} for model_name, info in scores_dict.items()]
    df_chart = pd.DataFrame(chart_data)
    fig = px.bar(
        df_chart,
        x="Model",
        y="MAD",
        template="plotly_white",
        labels={"MAD": "MAD Score", "Model": "Model Name"},
        title="Model Comparison: MAD",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title="Models", yaxis_title="MAD", margin=dict(l=40, r=40, t=50, b=40))
    return dcc.Graph(figure=fig)

def metric_definitions_accordion():
    """
    Provides a collapsible accordion with metric definitions.
    """
    return dbc.Accordion([
        dbc.AccordionItem([
            html.P("RMSE is calculated as sqrt(mean((predicted - actual)^2)). It penalizes larger errors more, making it sensitive to outliers.")
        ], title="RMSE"),
        dbc.AccordionItem([
            html.P("MAPE is computed as mean(|(predicted - actual) / actual| * 100). It provides a relative error measure but can be unstable when actual values are small.")
        ], title="MAPE (%)"),
        dbc.AccordionItem([
            html.P("MAD is the mean of absolute differences between predictions and actual values. It is robust against outliers and easy to interpret.")
        ], title="MAD")
    ], start_collapsed=True)

def detailed_metric_explanations():
    """
    Provides an in-depth explanation of the error metrics.
    """
    explanation_text = [
        html.H5("Detailed Metric Analysis", className="text-primary"),
        html.P("• RMSE: Lower values indicate better model performance. Its squaring of errors means larger mistakes have a higher impact. "
               "If RMSE is high, consider robust regression or outlier removal."),
        html.P("• MAPE: Expresses errors as a percentage which is intuitive for comparison. However, it can be unreliable when actual values are near zero."),
        html.P("• MAD: Gives an average magnitude of errors, providing a straightforward measure of model performance without squaring errors."),
        html.P("These metrics help you understand both the absolute and relative performance of your models.")
    ]
    return dbc.Card(
        dbc.CardBody(explanation_text),
        className="mt-4 mb-4",
        style={"backgroundColor": "#f9f9f9"}
    )

def advanced_analysis_recommendations():
    """
    Provides actionable recommendations for model improvement.
    """
    recommendations = [
        html.H5("Advanced Analysis & Recommendations", className="text-success"),
        html.P("1. Data Quality: Clean your data and handle outliers effectively."),
        html.P("2. Feature Engineering: Consider new features or transform existing ones to better capture patterns."),
        html.P("3. Model Ensembling: Combine different models (e.g., RF, XGB, Linear) to reduce variance."),
        html.P("4. Hyperparameter Tuning: Utilize grid or random search to optimize model parameters."),
        html.P("5. Cross-Validation: Regularly perform cross-validation to ensure the model generalizes well."),
        html.P("6. Continuous Monitoring: Retrain models with new data periodically to maintain performance.")
    ]
    return dbc.Card(
        dbc.CardBody(recommendations),
        className="mt-4 mb-4",
        style={"backgroundColor": "#fefefe", "border": "1px solid #eaeaea"}
    )

def modelinfo_layout(scores_dict):
    """
    Combines all elements into a comprehensive Model Info page layout.
    """
    return dbc.Container([
        html.H2("Model Information & Detailed Analysis", className="mt-4 mb-4 text-center text-secondary"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Scoreboard", className="bg-info text-white"),
                    dbc.CardBody([
                        html.P("The table below shows key error metrics for each model:", className="text-dark"),
                        make_model_score_table(scores_dict),
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("RMSE Comparison", className="bg-light text-dark"),
                    dbc.CardBody([make_rmse_bar_chart(scores_dict)])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("MAPE Comparison", className="bg-light text-dark"),
                    dbc.CardBody([make_mape_bar_chart(scores_dict)])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("MAD Comparison", className="bg-light text-dark"),
                    dbc.CardBody([make_mad_bar_chart(scores_dict)])
                ], className="mb-4")
            ], width=12)
        ]),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H4("Metric Definitions", className="mb-3"),
                metric_definitions_accordion()
            ], width=12)
        ], className="mb-5"),
        
        detailed_metric_explanations(),
        advanced_analysis_recommendations()
        
    ], fluid=True)

##############################################################################
# 3) Create the Dash App and Navbar
##############################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Airport Digital Twin"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Simulation Dashboard", href="/")),
        dbc.NavItem(dbc.NavLink("Manual Input", href="/manual")),
        dbc.NavItem(dbc.NavLink("About Us", href="/about")),
        dbc.NavItem(dbc.NavLink("Model Info", href="/modelinfo")),
        dbc.NavItem(dbc.NavLink("Logout", href="/logout", id="logout-link", style={"display": "none"}))
    ],
    brand="YUL Digital Twin",
    brand_href="/",
    color="dark",
    dark=True
)

####################
# Combined Layout (Main and Login)
####################
app.layout = html.Div([
    dcc.Store(id="current-user", storage_type="session"),
    dcc.Location(id="url"),
    navbar,
    html.Div(id="page-content")
])

####################
# Routing / Page Switch
####################
@app.callback(
    Output("page-content", "children"),
    Output("logout-link", "style"),
    Output("current-user", "data", allow_duplicate=True),
    Input("url", "pathname"),
    State("current-user", "data"),
    prevent_initial_call=True,
)
def route_pages(pathname, current_user):
    if pathname == "/logout":
        return login_layout, {"display": "none"}, None
    if not current_user:
        return login_layout, {"display": "none"}, dash.no_update
    if pathname == "/manual":
        return manual_layout, {"display": "block"}, dash.no_update
    elif pathname == "/about":
        return about_layout, {"display": "block"}, dash.no_update
    elif pathname == "/modelinfo":
        return modelinfo_layout(model_scores), {"display": "block"}, dash.no_update
    else:
        return home_layout, {"display": "block"}, dash.no_update

####################
# Improved Login/Signup Page Layout
####################
login_layout = html.Div(
    style={"position": "relative"},
    children=[
        # --- VIDEO BACKGROUND ---
        html.Video(
            src="/assets/Welcome Video.mp4",  # ensure this file is in your assets folder
            autoPlay=True,
            muted=True,
            loop=True,
            controls=False,
            style={
                "position": "fixed",
                "right": "0",
                "bottom": "0",
                "minWidth": "100%",
                "minHeight": "100%",
                "objectFit": "cover",
                "zIndex": "-1"
            }
        ),
        # --- LOGIN CARD CONTAINER ---
        dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    html.Img(
                                        src="/assets/SKY AI.jpg",  # ensure this file is in your assets folder
                                        style={
                                            "width": "120px",
                                            "display": "block",
                                            "margin": "0 auto",
                                            "marginBottom": "10px"
                                        }
                                    ),
                                    html.H2(
                                        "Sky AI",
                                        className="text-center",
                                        style={"fontWeight": "bold", "marginBottom": "0px"}
                                    ),
                                    html.P(
                                        "Login / Sign Up",
                                        className="text-center text-muted"
                                    ),
                                    # Tagline or mission statement
                                    html.P(
                                        "Experience advanced AI analytics and real-time airport operations.",
                                        className="text-center text-info",
                                        style={"fontSize": "0.9rem", "marginTop": "5px"}
                                    )
                                ], className="mb-4"),
                                # Tabs: Login, Sign Up
                                dcc.Tabs(id="login-signup-tabs", value="login-tab", children=[
                                    dcc.Tab(label="Login", value="login-tab", children=[
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Username:", style={"fontWeight": "bold"}),
                                                dcc.Input(
                                                    id="login-username",
                                                    type="text",
                                                    placeholder="Enter username",
                                                    style={"width": "100%"}
                                                ),
                                                html.Br(), html.Br(),
                                                html.Label("Password:", style={"fontWeight": "bold"}),
                                                dcc.Input(
                                                    id="login-password",
                                                    type="password",
                                                    placeholder="Enter password",
                                                    style={"width": "100%"}
                                                ),
                                                html.Br(), html.Br(),
                                                dbc.Button("Login", id="login-button", color="primary", style={"width": "100%"}),
                                                html.Div(id="login-message", style={"marginTop": "10px"})
                                            ], width=12)
                                        ], justify="center")
                                    ]),
                                    dcc.Tab(label="Sign Up", value="signup-tab", children=[
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("New Username:", style={"fontWeight": "bold"}),
                                                dcc.Input(
                                                    id="signup-username",
                                                    type="text",
                                                    placeholder="Choose username",
                                                    style={"width": "100%"}
                                                ),
                                                html.Br(), html.Br(),
                                                html.Label("New Password:", style={"fontWeight": "bold"}),
                                                dcc.Input(
                                                    id="signup-password",
                                                    type="password",
                                                    placeholder="Choose password",
                                                    style={"width": "100%"}
                                                ),
                                                html.Br(), html.Br(),
                                                dbc.Button("Sign Up", id="signup-button", color="success", style={"width": "100%"}),
                                                html.Div(id="signup-message", style={"marginTop": "10px"})
                                            ], width=12)
                                        ], justify="center")
                                    ])
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("By logging in, you agree to our Terms & Conditions."),
                                        html.Br(),
                                        html.A("Contact Support", href="mailto:skiaiconcordia@gmail.com", className="text-muted"),
                                    ], className="text-center mt-3")
                                ])
                            ]),
                            className="shadow p-4",
                            style={
                                "borderRadius": "10px",
                                "backgroundColor": "rgba(255, 255, 255, 0.8)"
                            }
                        ),
                        width=4
                    ),
                    justify="center",
                    align="center",
                    style={"height": "100vh"}
                )
            ]
        )
    ]
)

##################
# Login Callback
##################
@app.callback(
    Output("login-message", "children"),
    Output("current-user", "data"),
    Output("url", "pathname"),
    Input("login-button", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def process_login(n_clicks, username, password):
    if not username or not password:
        return "Please enter both username and password.", dash.no_update, dash.no_update
    user_db = load_users()
    hashed_input = hash_password(password)
    if username not in user_db:
        return "User does not exist.", dash.no_update, dash.no_update
    if user_db[username] != hashed_input:
        return "Incorrect password.", dash.no_update, dash.no_update
    return "", username, "/"

##################
# Signup Callback
##################
@app.callback(
    Output("signup-message", "children"),
    Input("signup-button", "n_clicks"),
    State("signup-username", "value"),
    State("signup-password", "value"),
    prevent_initial_call=True,
)
def process_signup(n_clicks, username, password):
    if not username or not password:
        return "Please enter a username and password."
    user_db = load_users()
    if username in user_db:
        return "Username already exists. Choose a different name."
    user_db[username] = hash_password(password)
    save_users(user_db)
    return "Sign Up successful! You can now log in."

################################
# Simulation & Dashboard Callbacks
################################

# 1) Flight Details Callback (Updated: Grouped and Scrollable)
@app.callback(
    Output("flight-details", "children"),
    Input("flight-selector", "value")
)
def update_flight_details(selected_flight):
    print("Flight selected index:", selected_flight)
    if flights_data.empty:
        return html.Div("No flight data available.", style={"color": "red"})
    if selected_flight is None or selected_flight >= len(flights_data):
        return html.Div("Invalid flight selection.", style={"color": "red"})
    flight = flights_data.iloc[selected_flight].to_dict()
    
    # Define groups of details to display
    basic_info = [
        ("Flight Name", "flight_name"),
        ("Flight ID", "flight_id"),
        ("Airline", "airline"),
        ("Aircraft Type", "aircraft_type"),
        ("Origin", "origin"),
        ("Destination", "destination"),
    ]
    timing_info = [
        ("ETA", "eta"),
        ("ETD", "etd"),
        ("Turnaround Analytics", "turnaround_analytics"),
    ]
    status_info = [
        ("Status", "status"),
        ("Refueling Status", "refueling_status"),
        ("Baggage Status", "baggage_status"),
        ("Taxiing Status", "taxiing_status"),
        ("Gate Efficiency", "gate_efficiency"),
        ("Maintenance Status", "maintenance_status"),
        ("Crew Availability", "crew_availability"),
        ("Cleaning Status", "cleaning_status"),
        ("Security Status", "security_status"),
    ]
    delay_info = [
        ("Predicted Delay (mins)", "predicted_delay"),
        ("Previous Delay (mins)", "previous_delay"),
    ]
    extra_info = [
        ("Fuel Status", "fuel_status"),
        ("Emissions (kg CO2)", "emissions"),
        ("Maintenance Counter", "maintenance_counter"),
        ("Crew Pool", "crew_pool"),
        ("Seat Capacity", "seat_capacity"),
        ("Overbooked", "overbooked"),
        ("Staff Shortage Factor", "staff_shortage_factor"),
        ("Day Wind Speed", "day_wind_speed"),
        ("Day Precipitation", "day_precipitation"),
        ("Day Temperature", "day_temperature")
    ]
    
    def make_section(title, section):
        rows = []
        for label, key in section:
            value = flight.get(key, "N/A")
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            rows.append(html.Tr([
                html.Td(html.B(label), style={"padding": "4px 8px"}),
                html.Td(str(value), style={"padding": "4px 8px"})
            ]))
        return html.Div([
            html.H5(title, className="mt-3 mb-2"),
            dbc.Table(
                [html.Tbody(rows)],
                bordered=True,
                striped=True,
                hover=True,
                responsive=True,
                style={"fontSize": "14px", "marginBottom": "10px"}
            )
        ])
    
    content = html.Div([
        html.H4("Flight Details", className="text-center mb-3"),
        make_section("Basic Information", basic_info),
        make_section("Timing Information", timing_info),
        make_section("Status Information", status_info),
        make_section("Delay Information", delay_info),
        make_section("Additional Information", extra_info)
    ], style={
        "maxHeight": "500px", 
        "overflowY": "auto", 
        "padding": "10px", 
        "border": "1px solid #ccc", 
        "borderRadius": "5px"
    })
    
    return content

# 2) KPI Row (Ultra Hyper Realistic Values)
@app.callback(
    Output("avg-delay-kpi", "children"),
    Output("flights-on-time-kpi", "children"),
    Output("gate-efficiency-kpi", "children"),
    Output("runway-efficiency-kpi", "children"),
    Input("position-update-interval", "n_intervals")
)
def update_kpis(_):
    # For a very good airport, we simulate ultra-realistic KPIs:
    # - Average delay time: ~5 minutes
    # - Flight on time: 95%
    # - Gate efficiency: 98%
    # - Runway efficiency: 99%
    avg_delay = 5  # in minutes
    on_time_pct = 95
    gate_eff = 98
    runway_eff = 99
    return f"{avg_delay} mins", f"{on_time_pct}%", f"{gate_eff}%", f"{runway_eff}%"

# 3) Gantt Chart
@app.callback(
    Output("gantt-chart", "figure"),
    Input("flight-selector", "value")
)
def update_gantt_chart(selected_flight):
    import plotly.express as px
    from datetime import timedelta
    
    f = flights_data.iloc[selected_flight]
    
    processes = [
        {
            "Task": "Arrive (ETA)",
            "Start": f["eta"] - timedelta(minutes=15),
            "Finish": f["eta"],
            "Status": "Arrived"
        },
        {
            "Task": "Refuel",
            "Start": f["eta"],
            "Finish": f["eta"] + timedelta(minutes=random.randint(5, 10)),
            "Status": "In Progress"
        },
        {
            "Task": "Baggage & Cargo",
            "Start": f["eta"] + timedelta(minutes=5),
            "Finish": f["eta"] + timedelta(minutes=15),
            "Status": "Pending"
        },
        {
            "Task": "Cleaning",
            "Start": f["eta"] + timedelta(minutes=10),
            "Finish": f["eta"] + timedelta(minutes=20),
            "Status": "Scheduled"
        },
        {
            "Task": "Security Clearance",
            "Start": f["eta"] + timedelta(minutes=15),
            "Finish": f["etd"] - timedelta(minutes=10),
            "Status": "Pending"
        },
        {
            "Task": "Maintenance",
            "Start": f["eta"] + timedelta(minutes=20),
            "Finish": f["etd"] - timedelta(minutes=5),
            "Status": "Scheduled"
        },
        {
            "Task": "Departure (ETD)",
            "Start": f["etd"] - timedelta(minutes=5),
            "Finish": f["etd"],
            "Status": "Scheduled"
        }
    ]
    
    df = pd.DataFrame(processes)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Task",
        hover_data=["Status"]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white")
    return fig

# 4) Status Pie
@app.callback(
    Output("status-pie-chart", "figure"),
    Input("flight-selector", "value")
)
def update_status_pie_chart(_):
    import plotly.express as px
    st_counts = flights_data["status"].value_counts()
    fig = px.pie(names=st_counts.index, values=st_counts.values)
    return fig

# 5) Efficiency Line
@app.callback(
    Output("efficiency-line-chart", "figure"),
    Input("flight-selector", "value")
)
def update_efficiency_line_chart(_):
    import plotly.express as px
    df_eff = pd.DataFrame({
        "Time": ["T-" + str(i) for i in range(10, 0, -1)],
        "Efficiency": [random.randint(70, 100) for _ in range(10)]
    })
    fig = px.line(df_eff, x="Time", y="Efficiency", template="plotly_white")
    return fig

# 6) Prediction Scatter
@app.callback(
    Output("prediction-scatter", "figure"),
    Input("flight-selector", "value")
)
def update_prediction_scatter(_):
    import plotly.express as px
    flights_data["actual_delay"] = flights_data["predicted_delay"] + np.random.normal(0, 2, len(flights_data))
    flights_data["actual_delay"] = flights_data["actual_delay"].clip(lower=0)
    fig = px.scatter(
        x=flights_data["actual_delay"],
        y=flights_data["predicted_delay"],
        labels={"x": "Actual Delay (mins)", "y": "Predicted Delay (mins)"},
        template="plotly_white"
    )
    minv = 0
    maxv = max(flights_data["actual_delay"].max(), flights_data["predicted_delay"].max())
    fig.add_shape(
        type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="red", dash="dash")
    )
    return fig

# 7) Maintenance Prediction
@app.callback(
    Output("maintenance-prediction", "figure"),
    Input("flight-selector", "value")
)
def update_maintenance_prediction(_):
    import plotly.express as px
    def predict_maint_time(st):
        if st == "Overdue": return random.randint(0, 5)
        elif st == "Pending": return random.randint(5, 15)
        else: return random.randint(15, 30)
    flights_data["pred_maintenance_time"] = flights_data["maintenance_status"].apply(predict_maint_time)
    fig = px.bar(flights_data, x="flight_name", y="pred_maintenance_time", template="plotly_white")
    fig.update_layout(xaxis_tickangle=45)
    return fig

# 8) Emissions Bar
@app.callback(
    Output("emissions-bar-chart", "figure"),
    Input("flight-selector", "value")
)
def update_emissions_chart(_):
    import plotly.express as px
    fig = px.bar(
        flights_data, x="flight_name", y="emissions",
        labels={"flight_name": "Flight", "emissions": "Emissions (kg CO2)"},
        template="plotly_white"
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

# 9) Crew Status Chart
@app.callback(
    Output("crew-status-chart", "figure"),
    Input("position-update-interval", "n_intervals")
)
def update_crew_status_chart(_):
    import plotly.express as px
    if not simulation_state["crew_history"]:
        return {}
    times = [t for t, _ in simulation_state["crew_history"]]
    states = [s for _, s in simulation_state["crew_history"]]
    df = pd.DataFrame(states)
    df["time"] = times
    df_melt = df.melt(id_vars="time", var_name="pool", value_name="state")
    st_counts = df_melt.groupby(["time", "state"]).size().reset_index(name="count")
    fig = px.area(st_counts, x="time", y="count", color="state", template="plotly_white")
    return fig

# 10) Airport Map
@app.callback(
    Output("airport-map", "figure"),
    Input("flight-selector", "value"),
    Input("map-style-dropdown", "value"),
    Input("position-update-interval", "n_intervals"),
    Input("bearing-slider", "value"),
    Input("pitch-slider", "value"),
    Input("polygon-checklist", "value")
)
def update_airport_map(sel_flight, map_style, n_intervals, bearing, pitch, polygon_vals):
    import plotly.graph_objects as go
    update_positions()
    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_token,
            center=dict(lat=airport_center[0], lon=airport_center[1]),
            zoom=15,
            style=map_style,
            bearing=bearing,
            pitch=pitch
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        template="plotly_white"
    )
    show_poly = ("show" in polygon_vals)
    if show_poly:
        for rp in runway_polygons:
            fig.add_trace(go.Scattermapbox(
                fill="toself",
                lat=[p[0] for p in rp],
                lon=[p[1] for p in rp],
                fillcolor="rgba(200,0,0,0.2)",
                line=dict(color="red"),
                name="Runway Area"
            ))
        for tp in terminal_polygons:
            fig.add_trace(go.Scattermapbox(
                fill="toself",
                lat=[p[0] for p in tp],
                lon=[p[1] for p in tp],
                fillcolor="rgba(0,0,200,0.2)",
                line=dict(color="blue"),
                name="Terminal Area"
            ))
    for i in range(len(flights_data)):
        f = flights_data.iloc[i]
        lat = dynamic_positions.at[i, "lat"]
        lon = dynamic_positions.at[i, "lon"]
        if f["connection_misconnect"]:
            color = misconnect_color
        else:
            if f["resource_conflict"] or f["predicted_delay"] > 30:
                color = "#ff0000"
            else:
                color = status_colors.get(f["status"], "blue")
        hov_txt = (
            f"{f['flight_name']}<br>"
            f"ETA: {f['eta'].strftime('%H:%M')} / ETD: {f['etd'].strftime('%H:%M')}<br>"
            f"Passengers: {f['passengers']}"
        )
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(
                size=25,
                symbol="airport-15",
                color=color,
                allowoverlap=True
            ),
            name=f["flight_name"],
            hoverinfo="text", hovertext=hov_txt
        ))
    if flights_data.iloc[sel_flight]["status"] == "Taxiing":
        chosen = flights_data.iloc[sel_flight]
        if chosen.get("taxi_route"):
            fig.add_trace(go.Scattermapbox(
                lat=[r[0] for r in chosen["taxi_route"]],
                lon=[r[1] for r in chosen["taxi_route"]],
                mode="lines",
                line=dict(width=2, color="cyan"),
                name="Taxi Route"
            ))
    for v_id, v_info in simulation_state["vehicles"].items():
        vt = v_info["type"]
        fig.add_trace(go.Scattermapbox(
            lat=[v_info["lat"]], lon=[v_info["lon"]],
            mode="markers",
            marker=dict(size=8, symbol="car", color="purple"),
            name=vt,
            hoverinfo="text",
            hovertext=f"{vt} (ID {v_id})"
        ))
    return fig

# 11) Position Update Interval
@app.callback(
    Output("position-update-interval", "n_intervals"),
    Output("runway-count", "children"),
    Output("gate-count", "children"),
    Output("taxiing-count", "children"),
    Output("hanger-count", "children"),
    Output("avg-delay", "children"),
    Output("avg-emissions", "children"),
    Output("security-cleared-percent", "children"),
    Output("operational-cost", "children"),
    Output("departed-flights-count", "children"),
    Output("on-time-percent", "children"),
    Input("position-update-interval", "n_intervals"),
    State("advanced-scheduling-checklist", "value")
)
def trigger_position_update(n_intervals, adv_values):
    simulation_state["disable_advanced_scheduling"] = ("disable" in adv_values)
    run_digital_twin_simulation()
    r_count = (flights_data["status"] == "Runway").sum()
    g_count = (flights_data["status"] == "Gate").sum()
    t_count = (flights_data["status"] == "Taxiing").sum()
    h_count = (flights_data["status"] == "Hanger").sum()
    # The following KPIs are now simulated for a very good airport:
    avg_delay = 5  # Average delay in minutes
    on_time_pct = 95  # 95% of flights are on time
    gate_eff = 98     # 98% gate efficiency
    runway_eff = 99   # 99% runway efficiency
    a_em = round(flights_data["emissions"].mean(), 2)
    tot = len(flights_data)
    sec_str = "Security Cleared: 100%" if tot > 0 else "Security Cleared: N/A"
    cost_str = f"${round(simulation_state['operational_cost'], 2)}"
    departed_str = f"Departed: {simulation_state['total_departures']}"
    ot_str = f"On-Time Departures: {on_time_pct}%"
    return (
        n_intervals,
        f"Flights on Runway: {r_count}",
        f"Flights at Gates: {g_count}",
        f"Flights Taxiing: {t_count}",
        f"Flights in Hanger: {h_count}",
        f"{avg_delay} mins",
        f"Avg Emissions: {a_em} kg CO2",
        sec_str,
        cost_str,
        departed_str,
        ot_str
    )

# 12) Clock
@app.callback(
    Output("digital-clock", "children"),
    Input("position-update-interval", "n_intervals")
)
def update_digital_clock(_):
    return simulation_state["current_time"].strftime("%Y-%m-%d %H:%M:%S")

# 13) Event Log Modal
@app.callback(
    Output("event-log-modal", "is_open"),
    Output("event-log-body", "children"),
    Input("open-event-log", "n_clicks"),
    Input("close-event-log", "n_clicks"),
    State("event-log-modal", "is_open")
)
def toggle_event_log(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        if open_clicks and not close_clicks:
            logs = simulation_state["event_log"][-100:]
            text = "\n".join([f"{t.strftime('%Y-%m-%d %H:%M:%S')}: {msg}" for t, msg in logs])
            return True, html.Pre(text, style={"whiteSpace": "pre-wrap"})
        return False, None
    return is_open, None

# 14) Download Input CSV
@app.callback(
    Output("download-input-csv", "data"),
    Input("btn_download_input", "n_clicks"),
    prevent_initial_call=True
)
def download_input_data(n_clicks):
    from dash import dcc
    w = simulation_state["weather"]
    rows = []
    for i in range(len(flights_data)):
        f = flights_data.iloc[i]
        mf = maintenance_factor_for(f["maintenance_status"])
        cf = crew_factor_for(f["crew_availability"])
        rowd = {
            "flight_id": f["flight_id"],
            "flight_name": f["flight_name"],
            "origin": f["origin"],
            "destination": f["destination"],
            "wide_body": f["wide_body"],
            "seat_capacity": f["seat_capacity"],
            "passengers": f["passengers"],
            "wind_speed": w["wind_speed"],
            "precipitation": w["precipitation"],
            "maintenance_factor": mf,
            "crew_factor": cf,
            "previous_delay": f["previous_delay"],
            "is_international": f["is_international"],
            "overbooked": int(f["overbooked"]),
            "staff_shortage_factor": f["staff_shortage_factor"]
        }
        rows.append(rowd)
    df_in = pd.DataFrame(rows)
    return dcc.send_data_frame(df_in.to_csv, "airport_input_data.csv", index=False)

# 15) Download Predictions CSV
@app.callback(
    Output("download-predictions-csv", "data"),
    Input("btn_download_predictions", "n_clicks"),
    prevent_initial_call=True
)
def download_prediction_data(_):
    from dash import dcc
    import random
    rows = []
    for i in range(len(flights_data)):
        f = flights_data.iloc[i]
        mf = maintenance_factor_for(f["maintenance_status"])
        cf = crew_factor_for(f["crew_availability"])
        base_feats = {
            "wind_speed": simulation_state["weather"]["wind_speed"],
            "precipitation": simulation_state["weather"]["precipitation"],
            "maintenance_factor": mf,
            "crew_factor": cf,
            "previous_delay": f["previous_delay"],
            "is_international": f["is_international"],
            "overbooked": 1 if f["overbooked"] else 0,
            "staff_shortage_factor": f["staff_shortage_factor"]
        }
        
        flight_level_pred = rf_predict_delay(best_main_model, base_feats)
        flight_level_act = flight_level_pred + random.gauss(0, 2)
        flight_level_act = max(0, flight_level_act)
        flight_level_err = flight_level_act - flight_level_pred
        
        total_pred = 0
        total_act = 0
        subdelay_preds = {}
        subdelay_acts = {}
        subdelay_errors = {}
        for proc in process_list:
            staff_count = simulation_state["staff_scheduling"].get(proc, 1)
            pdly_pred = predict_process_delays(process_models, simulation_state["weather"], staff_count, proc)
            pdly_act = pdly_pred + random.gauss(0, 1.5)
            pdly_act = max(0, pdly_act)
            sub_err = pdly_act - pdly_pred
            subdelay_preds[proc] = pdly_pred
            subdelay_acts[proc] = pdly_act
            subdelay_errors[proc] = sub_err
            total_pred += pdly_pred
            total_act += pdly_act
        
        final_pred = flight_level_pred + total_pred
        final_act = flight_level_act + total_act
        final_err = final_act - final_pred
        
        recs = []
        if f["resource_conflict"]:
            recs.append("Add staff or reschedule tasks.")
        if f["maintenance_status"] == "Overdue":
            recs.append("Perform urgent maintenance.")
        if f["crew_availability"] == "Not Available":
            recs.append("Reassign crew pool.")
        if final_pred > 30:
            recs.append("Notify airline about potential delay.")
        if not recs:
            recs.append("No major issues.")
        action_str = " | ".join(recs)
        
        rowdict = {
            "flight_id": f["flight_id"],
            "flight_name": f["flight_name"],
            "maintenance_status": f["maintenance_status"],
            "crew_availability": f["crew_availability"],
            "resource_conflict": f["resource_conflict"],
            "connection_misconnect": f["connection_misconnect"],
            "emissions": round(f["emissions"], 2),
            "operational_cost_so_far": round(simulation_state["operational_cost"], 2),
            "flight_level_delay_pred": round(flight_level_pred, 2),
            "flight_level_delay_act": round(flight_level_act, 2),
            "flight_level_delay_error": round(flight_level_err, 2),
            "subdelay_baggage_pred": round(subdelay_preds["baggage"], 2),
            "subdelay_baggage_act": round(subdelay_acts["baggage"], 2),
            "subdelay_baggage_error": round(subdelay_errors["baggage"], 2),
            "subdelay_fueling_pred": round(subdelay_preds["fueling"], 2),
            "subdelay_fueling_act": round(subdelay_acts["fueling"], 2),
            "subdelay_fueling_error": round(subdelay_errors["fueling"], 2),
            "subdelay_pushback_pred": round(subdelay_preds["pushback"], 2),
            "subdelay_pushback_act": round(subdelay_acts["pushback"], 2),
            "subdelay_pushback_error": round(subdelay_errors["pushback"], 2),
            "subdelay_cleaning_pred": round(subdelay_preds["cleaning"], 2),
            "subdelay_cleaning_act": round(subdelay_acts["cleaning"], 2),
            "subdelay_cleaning_error": round(subdelay_errors["cleaning"], 2),
            "subdelay_security_pred": round(subdelay_preds["security"], 2),
            "subdelay_security_act": round(subdelay_acts["security"], 2),
            "subdelay_security_error": round(subdelay_errors["security"], 2),
            "final_predicted_delay": round(final_pred, 2),
            "final_actual_delay": round(final_act, 2),
            "final_error": round(final_err, 2),
            "model_used": MODEL_USED,
            "model_accuracy": "Multi-Model (RF, XGB, Linear), see scoreboard",
            "actionable_insights": action_str
        }
        rows.append(rowdict)
    df_pred = pd.DataFrame(rows)
    return dcc.send_data_frame(df_pred.to_csv, "airport_predicted_data.csv", index=False)

################################
# Manual Input Calculation
################################
@app.callback(
    Output("manual-prediction-result", "children"),
    Input("btn-manual-calc", "n_clicks"),
    State("manual-flight-selector", "value"),
    State("input-wind-speed", "value"),
    State("input-precipitation", "value"),
    State("input-maintenance", "value"),
    State("input-crew", "value"),
    State("input-prev-delay", "value"),
    State("input-international", "value"),
    State("input-overbooked", "value"),
    State("input-staff-shortage", "value"),
    State("manual-baggage-staff", "value"),
    State("manual-fueling-staff", "value"),
    State("manual-pushback-staff", "value"),
    State("manual-cleaning-staff", "value"),
    State("manual-security-staff", "value"),
    prevent_initial_call=True
)
def calculate_manual_prediction(n_clicks,
    selected_flight_idx,
    wind_sp, precip, maint_f, crew_f, prev_d, is_int, overb, staff_sh,
    bag_staff, fuel_staff, push_staff, clean_staff, sec_staff
):
    if not n_clicks:
        return ""
    
    f = flights_data.iloc[selected_flight_idx]
    flight_info = f"Selected Flight: {f['flight_name']} (ID {f['flight_id']})"

    base_feats = {
        "wind_speed": wind_sp or 0,
        "precipitation": precip or 0,
        "maintenance_factor": maint_f or 0,
        "crew_factor": crew_f or 0,
        "previous_delay": prev_d or 0,
        "is_international": is_int or 0,
        "overbooked": overb or 0,
        "staff_shortage_factor": staff_sh or 0
    }
    flight_delay = rf_predict_delay(best_main_model, base_feats)

    weather_dict = {
        "wind_speed": wind_sp or 0,
        "precipitation": precip or 0
    }
    staff_map = {
        "baggage": bag_staff or 0,
        "fueling": fuel_staff or 0,
        "pushback": push_staff or 0,
        "cleaning": clean_staff or 0,
        "security": sec_staff or 0
    }
    process_delays = {}
    for proc in process_list:
        dly = predict_process_delays(process_models, weather_dict, staff_map[proc], proc)
        process_delays[proc] = dly
    
    total_proc = sum(process_delays.values())
    total_pred = flight_delay + total_proc
    breakdown = "\n".join([f"  - {p.title()}: {round(v,2)} mins" for p, v in process_delays.items()])
    summary = f"""
{flight_info}

Flight-level (RF/XGB/Linear) Delay: {round(flight_delay,2)} mins
Sub-Process Delays:
{breakdown}

-------------------------------
TOTAL Predicted Delay = {round(total_pred,2)} mins
"""
    return html.Pre(summary, style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})

#######################
# Show Scoreboard
#######################
@app.callback(
    Output("model-scores-div", "children"),
    Input("url", "pathname")
)
def show_model_scores(_):
    """
    Display the RMSE, MAPE, MAD for each ML model we trained.
    This is read from the global `model_scores` dict.
    """
    lines = []
    for model_name, info in model_scores.items():
        rmse = info["rmse"]
        mape = info["mape"]
        mad = info["mad"]
        msg = f"{model_name}: RMSE={rmse:.2f}, MAPE={mape:.2f}%, MAD={mad:.2f}"
        lines.append(html.Div(msg))
    return lines

if __name__ == "__main__":
    app.run_server(debug=True)