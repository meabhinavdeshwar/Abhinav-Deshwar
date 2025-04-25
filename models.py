# models.py

import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# PuLP for optimization
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpContinuous, value

###############################################################################
# Updated Feature Columns (Realistic features for flight delay prediction)
###############################################################################
FEATURE_COLS = [
    "wind_speed", "precipitation", "maintenance_factor", "crew_factor",
    "previous_delay", "is_international", "overbooked", "staff_shortage_factor",
    "day_of_week", "hour", "passenger_load", "congestion_index"
]

###############################################################################
# Synthetic Data Generation
###############################################################################
def generate_synthetic_training_data(num_samples=1500):
    """
    Generates synthetic training data to simulate YUL airport operations.
    Includes features capturing weather, operational, flight, and staffing conditions.
    """
    np.random.seed(42)
    
    wind_speed = np.random.uniform(0, 40, size=num_samples)
    precipitation = np.random.uniform(0, 20, size=num_samples)
    maintenance_factor = np.random.uniform(0.5, 1.5, size=num_samples)
    crew_factor = np.random.uniform(0.5, 1.5, size=num_samples)
    previous_delay = np.random.uniform(0, 60, size=num_samples)
    is_international = np.random.binomial(1, 0.3, size=num_samples)
    overbooked = np.random.binomial(1, 0.2, size=num_samples)
    staff_shortage_factor = np.random.uniform(0.5, 2.0, size=num_samples)
    day_of_week = np.random.randint(0, 7, size=num_samples)
    hour = np.random.randint(0, 24, size=num_samples)
    passenger_load = np.random.uniform(50, 300, size=num_samples)
    congestion_index = np.random.uniform(0, 10, size=num_samples)
    
    df = pd.DataFrame({
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "maintenance_factor": maintenance_factor,
        "crew_factor": crew_factor,
        "previous_delay": previous_delay,
        "is_international": is_international,
        "overbooked": overbooked,
        "staff_shortage_factor": staff_shortage_factor,
        "day_of_week": day_of_week,
        "hour": hour,
        "passenger_load": passenger_load,
        "congestion_index": congestion_index
    })
    
    # Extra delay penalties for weekends and rush hours
    weekend_penalty = np.where(day_of_week >= 5, 2, 0)
    rush_penalty = np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)), 3, 0)
    
    delay = (
        5
        + wind_speed * 0.3
        + precipitation * 10
        + maintenance_factor * 5
        + crew_factor * 3
        + previous_delay * 0.8
        + is_international * 4
        + overbooked * 3
        + staff_shortage_factor * 4
        + weekend_penalty
        + rush_penalty
        + passenger_load * 0.05
        + congestion_index * 1
        + np.random.normal(0, 5, num_samples)
    )
    df["delay"] = np.maximum(delay, 0)
    return df

###############################################################################
# Evaluation Metrics
###############################################################################
def calc_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true != 0)
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calc_mad(y_true, y_pred):
    """
    Mean Absolute Deviation (MAD).
    """
    return np.mean(np.abs(y_true - y_pred))

###############################################################################
# ML Model Training and Prediction (Improved with RandomizedSearchCV)
###############################################################################
def train_main_models(num_samples=1500):
    """
    Trains several regression models on the synthetic flight delay data.
    Uses RandomizedSearchCV for improved hyperparameter tuning.
    Returns the best model and a dictionary of evaluation metrics.
    """
    df = generate_synthetic_training_data(num_samples)
    X = df[FEATURE_COLS]
    y = df["delay"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameter distributions for RandomForest
    rf_param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 3]
    }
    rf_base = RandomForestRegressor(random_state=0)
    rf_search = RandomizedSearchCV(estimator=rf_base, param_distributions=rf_param_dist,
                                   scoring="neg_mean_squared_error", cv=3, n_iter=10,
                                   n_jobs=-1, random_state=42, verbose=0)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mape = calc_mape(y_test, rf_pred)
    rf_mad = calc_mad(y_test, rf_pred)
    
    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42,
                             objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mape = calc_mape(y_test, xgb_pred)
    xgb_mad = calc_mad(y_test, xgb_pred)
    
    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
    lin_mape = calc_mape(y_test, lin_pred)
    lin_mad = calc_mad(y_test, lin_pred)
    
    # Stacking Ensemble
    estimators = [('rf', rf_best), ('xgb', xgb_model), ('lin', lin_model)]
    stack_model = StackingRegressor(estimators=estimators,
                                    final_estimator=LinearRegression(),
                                    cv=5)
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_test)
    stack_rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
    stack_mape = calc_mape(y_test, stack_pred)
    stack_mad = calc_mad(y_test, stack_pred)
    
    model_scores = {
        "RandomForest": {"model": rf_best, "rmse": rf_rmse, "mape": rf_mape, "mad": rf_mad},
        "XGBoost": {"model": xgb_model, "rmse": xgb_rmse, "mape": xgb_mape, "mad": xgb_mad},
        "LinearReg": {"model": lin_model, "rmse": lin_rmse, "mape": lin_mape, "mad": lin_mad},
        "StackingEnsemble": {"model": stack_model, "rmse": stack_rmse, "mape": stack_mape, "mad": stack_mad}
    }
    
    best_name = min(model_scores, key=lambda nm: model_scores[nm]["rmse"])
    best_model = model_scores[best_name]["model"]
    
    return best_model, model_scores

def rf_predict_delay(model, features: dict) -> float:
    """
    Uses the trained ML model to predict flight delay from input features.
    Missing feature values are replaced by sensible defaults.
    """
    df_in = pd.DataFrame([features], columns=FEATURE_COLS)
    defaults = {
        "day_of_week": datetime.now().weekday(),
        "hour": datetime.now().hour,
        "passenger_load": 150,
        "congestion_index": 5
    }
    df_in = df_in.fillna(value=defaults)
    pred = model.predict(df_in)[0]
    return max(pred, 0)

###############################################################################
# Mathematical Optimization Model for Turnaround Scheduling
###############################################################################
def optimize_turnaround_schedule(flights, gates, time_slots, processes, crews, params):
    """
    Formulates and solves a mixed-integer linear program for optimizing the turnaround schedule.
    
    The same base input (e.g., arrival time, scheduled departure, turnaround duration, delay penalties)
    used for the ML model can be used here (possibly after additional processing) to define parameters.
    
    Returns a dictionary with the optimized decision variable values.
    """
    prob = LpProblem("Turnaround_Schedule_Optimization", LpMinimize)
    
    flights_set = list(flights.keys())
    
    # Decision Variables
    x = {i: LpVariable(f"x_{i}", lowBound=params['L'][i], upBound=params['U'][i], cat=LpContinuous)
         for i in flights_set}
    z = {i: LpVariable(f"z_{i}", lowBound=0, cat=LpContinuous) for i in flights_set}
    y = {(i, g): LpVariable(f"y_{i}_{g}", cat=LpBinary) for i in flights_set for g in gates}
    w = {(i, t): LpVariable(f"w_{i}_{t}", cat=LpBinary) for i in flights_set for t in time_slots}
    o = {(i, j): LpVariable(f"o_{i}_{j}", cat=LpBinary) for i in flights_set for j in flights_set if i != j}
    u = {(p, i): LpVariable(f"u_{p}_{i}", cat=LpBinary) for p in processes for i in flights_set}
    v = {(i, c): LpVariable(f"v_{i}_{c}", cat=LpBinary) for i in flights_set for c in crews}
    
    # Objective: Minimize delay penalties, gate assignment costs, and process execution costs.
    prob += (
        lpSum([params['p'][i] * z[i] for i in flights_set])
        + lpSum([params['C_gate'][(i, g)] * y[(i, g)] for i in flights_set for g in gates])
        + lpSum([params['C_proc'][(p, i)] * u[(p, i)] for p in processes for i in flights_set])
    )
    
    # Constraints:
    for i in flights_set:
        prob += z[i] == x[i] - params['D0'][i]  # Delay definition
        prob += z[i] >= 0
        # Time window bounds are set by variable bounds.
    
    # Each flight is assigned to exactly one compatible gate.
    for i in flights_set:
        prob += lpSum([y[(i, g)] for g in gates]) == 1
        for g in gates:
            prob += y[(i, g)] <= params['delta'][(i, g)]
    
    # Gate sequencing: ensure non-overlap for flights sharing a gate.
    M = params['M']
    for i in flights_set:
        for j in flights_set:
            if i != j:
                for g in gates:
                    prob += x[i] + params['T_req'][i] <= x[j] + M*(1 - o[(i, j)] + (1 - y[(i, g)]) + (1 - y[(j, g)]))
                    prob += x[j] + params['T_req'][j] <= x[i] + M*(o[(i, j)] + (1 - y[(i, g)]) + (1 - y[(j, g)]))
    
    # Time slot assignment: each flight is assigned one time slot.
    for i in flights_set:
        prob += lpSum([w[(i, t)] for t in time_slots]) == 1
    
    # Crew resource constraints.
    for t in time_slots:
        prob += lpSum([params['r'][i] * w[(i, t)] for i in flights_set]) <= params['R'][t]
    
    # Service process scheduling.
    for i in flights_set:
        prob += lpSum([params['alpha'][p] * u[(p, i)] for p in processes]) <= params['T_req'][i]
        for p in processes:
            if params['beta'][(p, i)] == 1:
                prob += u[(p, i)] == 1
    
    # Gate availability constraints.
    for i in flights_set:
        for g in gates:
            for t in time_slots:
                prob += y[(i, g)] <= params['a'][(g, t)]
    
    # Crew assignment: each flight gets exactly one crew team.
    for i in flights_set:
        prob += lpSum([v[(i, c)] for c in crews]) == 1
    
    # Solve the optimization problem.
    prob.solve()
    
    # Extract and return solution.
    solution = {
        "x": {i: x[i].varValue for i in flights_set},
        "z": {i: z[i].varValue for i in flights_set},
        "y": {(i, g): y[(i, g)].varValue for i in flights_set for g in gates},
        "w": {(i, t): w[(i, t)].varValue for i in flights_set for t in time_slots},
        "o": {(i, j): o[(i, j)].varValue for (i, j) in o},
        "u": {(p, i): u[(p, i)].varValue for (p, i) in u},
        "v": {(i, c): v[(i, c)].varValue for (i, c) in v}
    }
    return solution

###############################################################################
# Sub-Process Models for Service Processes
###############################################################################
PROCESS_COLS = ["wind_speed", "staff_available", "precipitation"]
PROCESS_LIST = ["baggage", "fueling", "pushback", "cleaning", "security"]

def generate_synthetic_process_data(proc_name, num_samples=800):
    """
    Generates synthetic data for a given sub-process.
    """
    np.random.seed(42)
    wind_speed = np.random.uniform(5, 35, size=num_samples)
    staff_available = np.random.randint(0, 7, size=num_samples)
    precipitation = np.random.choice([0, 0.1, 0.3, 0.5], size=num_samples, p=[0.4, 0.2, 0.2, 0.2])
    
    if proc_name == "baggage":
        delay = (wind_speed * 0.05) + (7 - staff_available) * 1.5 + precipitation * 5 + np.random.normal(0, 2, num_samples)
    elif proc_name == "fueling":
        delay = (wind_speed * 0.08) + (7 - staff_available) * 2.0 + precipitation * 3 + np.random.normal(0, 3, num_samples)
    elif proc_name == "pushback":
        delay = (wind_speed * 0.06) + (7 - staff_available) * 2.5 + precipitation * 4 + np.random.normal(0, 2, num_samples)
    elif proc_name == "cleaning":
        delay = (wind_speed * 0.04) + (7 - staff_available) * 1.8 + precipitation * 5 + np.random.normal(0, 1.5, num_samples)
    else:  # security
        delay = (wind_speed * 0.03) + (7 - staff_available) * 1.0 + precipitation * 6 + np.random.normal(0, 4, num_samples)
    
    df_proc = pd.DataFrame({
        "wind_speed": wind_speed,
        "staff_available": staff_available,
        "precipitation": precipitation,
        "delay": delay
    })
    return df_proc

def train_process_models():
    """
    Trains a RandomForest model for each service process.
    Returns a dictionary mapping each process to its trained model.
    """
    process_models = {}
    for proc in PROCESS_LIST:
        df_proc = generate_synthetic_process_data(proc, num_samples=800)
        Xp = df_proc[PROCESS_COLS]
        yp = df_proc["delay"]
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)
        rf_proc = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_proc.fit(Xp_train, yp_train)
        process_models[proc] = rf_proc
    return process_models

def predict_process_delays(process_models, weather: dict, staff: int, proc_name: str) -> float:
    """
    Predicts delay for a specific service process.
    """
    df_in = pd.DataFrame([{
        "wind_speed": weather["wind_speed"],
        "staff_available": staff,
        "precipitation": weather["precipitation"]
    }], columns=PROCESS_COLS)
    p = process_models[proc_name].predict(df_in)[0]
    return max(p, 0)

# For compatibility with other parts of the application
process_list = PROCESS_LIST

###############################################################################
# Additional Visual Functions for Dashboard
###############################################################################
def get_model_performance_figure(model_scores):
    metrics = []
    for name, score in model_scores.items():
        metrics.append({"Model": name, "Metric": "RMSE", "Value": score["rmse"]})
        metrics.append({"Model": name, "Metric": "MAPE (%)", "Value": score["mape"]})
        metrics.append({"Model": name, "Metric": "MAD", "Value": score["mad"]})
    df_metrics = pd.DataFrame(metrics)
    fig = px.bar(df_metrics, x="Model", y="Value", color="Metric",
                 barmode="group", template="plotly_white",
                 title="Model Performance Comparison",
                 labels={"Value": "Error Value", "Model": "Model Name"})
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

def get_feature_importances(model, X, feature_cols=FEATURE_COLS):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_importance = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
        return df_importance
    else:
        return "Feature importances not available for this model type."

###############################################################################
# Main Execution Block (for testing)
###############################################################################
if __name__ == "__main__":
    # Train the ML model and process models.
    best_model, scores = train_main_models(1500)
    process_models = train_process_models()
    
    # Example: Use the same input features for ML prediction and then pass derived parameters
    sample_features = {
        "wind_speed": 15,
        "precipitation": 5,
        "maintenance_factor": 1.0,
        "crew_factor": 1.0,
        "previous_delay": 10,
        "is_international": 0,
        "overbooked": 0,
        "staff_shortage_factor": 1.0,
        "day_of_week": 2,
        "hour": 14,
        "passenger_load": 200,
        "congestion_index": 4
    }
    
    ml_predicted_delay = rf_predict_delay(best_model, sample_features)
    print("Predicted Delay from ML Model: {:.2f} minutes".format(ml_predicted_delay))
    
    # Example: Build parameter dictionary for the optimization model using derived ML output
    # (In practice, these parameters should be computed from your flight data.)
    params = {
        'L': {i: 0 for i in range(1, 6)},
        'U': {i: 100 for i in range(1, 6)},
        'D0': {i: 20 for i in range(1, 6)},
        'T_req': {i: ml_predicted_delay for i in range(1, 6)},
        'p': {i: 1 for i in range(1, 6)},
        'delta': {(i, g): 1 for i in range(1, 6) for g in ['Gate1', 'Gate2']},
        'M': 1000,
        'r': {i: 2 for i in range(1, 6)},
        'R': {t: 10 for t in range(1, 11)},
        'alpha': {p: 5 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security']},
        'beta': {(p, i): 1 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security'] for i in range(1, 6)},
        'a': {(g, t): 1 for g in ['Gate1', 'Gate2'] for t in range(1, 11)},
        'C_gate': {(i, g): 1 for i in range(1, 6) for g in ['Gate1', 'Gate2']},
        'C_proc': {(p, i): 1 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security'] for i in range(1, 6)}
    }
    # Dummy flight dictionary for optimization model (flight IDs 1 to 5)
    flights_dict = {i: {} for i in range(1, 6)}
    gates = ['Gate1', 'Gate2']
    time_slots = list(range(1, 11))
    processes = ['baggage', 'fueling', 'pushback', 'cleaning', 'security']
    crews = ['Crew1', 'Crew2']
    
    opt_solution = optimize_turnaround_schedule(flights_dict, gates, time_slots, processes, crews, params)
    print("\nOptimized Turnaround Schedule Decision Variables:")
    print(opt_solution)