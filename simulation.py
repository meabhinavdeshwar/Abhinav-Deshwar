import os
import json
import random
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State, dash_table

# Updated import to include process models and related functions, as well as the optimization function.
from models import (
    FEATURE_COLS, 
    rf_predict_delay, 
    train_main_models, 
    train_process_models, 
    predict_process_delays, 
    process_list,
    optimize_turnaround_schedule  # Optimization model function
)

##################
# Global Variables
##################
MODEL_USED = "Stacking Ensemble (RF + XGB + Linear Regression)"
MAIN_MODEL_RMSE = 2.54
PROCESS_MODEL_RMSE = 3.10

##################
# Global State
##################
simulation_state = {
    "current_time": datetime.now(),
    "time_step": timedelta(minutes=5),  # simulation time step of 5 minutes
    "weather": {
        "wind_speed": random.uniform(5, 20),
        "precipitation": random.choice([0, 0.1, 0.3, 0.5]),
        "temperature": random.uniform(-5, 25)
    },
    "num_crew_pools": 3,
    "crew_pools": {1: "Available", 2: "Available", 3: "Available"},
    "staff_scheduling": {
        "baggage": 5,
        "fueling": 3,
        "pushback": 2,
        "cleaning": 4,
        "security": 6
    },
    "staff_hours_remaining": {
        "baggage": 40,
        "fueling": 30,
        "pushback": 20,
        "cleaning": 40,
        "security": 50
    },
    "disable_advanced_scheduling": False,
    "gates": [
        {"flight_id": None, "capacity": "Any"},
        {"flight_id": None, "capacity": "Narrow"},
        {"flight_id": None, "capacity": "Wide"},
        {"flight_id": None, "capacity": "Any"}
    ],
    "event_log": [],
    "operational_cost": 0.0,
    "crew_history": [],
    "on_time_departures": 0,
    "total_departures": 0,
    "kpi_history": [],
    "weather_history": [],
    "security_queue_length": 0,
    "runways": {
        0: {"status": "Open", "next_maintenance": datetime.now() + timedelta(minutes=60)},
        1: {"status": "Open", "next_maintenance": datetime.now() + timedelta(minutes=90)}
    },
    "vehicles": {},
    "airport_nodes": {
        "node_A": (45.4700, -73.7420),
        "node_B": (45.4705, -73.7450),
        "node_C": (45.4710, -73.7480),
        "node_D": (45.4715, -73.7495),
        "node_E": (45.4720, -73.7520),
        "node_F": (45.4695, -73.7390)
    },
    "airport_edges": {
        ("node_A", "node_B"): 1.0,
        ("node_B", "node_C"): 1.0,
        ("node_C", "node_D"): 1.0,
        ("node_D", "node_E"): 1.0,
        ("node_A", "node_F"): 1.2,
        ("node_F", "node_B"): 0.8,
    },
    "delay_buffer": [],
    "anomaly_detector": None,
    "flight_connections": {}
}

# Initialize process models (required for prediction CSV callback)
process_models = train_process_models()

###########################
# Turnaround Sequence Functions
###########################
def get_turnaround_sequence():
    """
    Define a sequential list of tasks for the turnaround process.
    Durations are in minutes.
    """
    return [
        {"name": "Preliminary Checks", "duration": 5},
        {"name": "Unload Cargo", "duration": 10},
        {"name": "Unload Baggage", "duration": 10},
        {"name": "Baggage Scanning", "duration": 5},
        {"name": "Open Baggage Hold", "duration": 3},
        {"name": "Close Baggage Hold", "duration": 3},
        {"name": "Bring Baggage to Aircraft", "duration": 8},
        {"name": "Load Baggage into Aircraft", "duration": 8},
        {"name": "Load Cargo into Aircraft", "duration": 8},
        {"name": "Position Ramp Agents", "duration": 3},
        {"name": "Towing Operation", "duration": 7},
        {"name": "Waste Disposal", "duration": 4},
        {"name": "Supply Potable Water", "duration": 2},
        {"name": "Cleaning", "duration": 5},
        {"name": "Final Checks", "duration": 3}
    ]

def initialize_turnaround_for_flight(index):
    """
    Initialize the turnaround sequence for a flight at the given index.
    Adds new fields to flights_data.
    """
    seq = get_turnaround_sequence()
    flights_data.at[index, "turnaround_sequence"] = seq
    flights_data.at[index, "current_task_index"] = 0
    flights_data.at[index, "task_remaining_time"] = seq[0]["duration"]
    simulation_state["event_log"].append(
        (simulation_state["current_time"], f"Flight {flights_data.at[index, 'flight_name']} initialized turnaround sequence.")
    )

###########################
# Weather and Time Utility Functions
###########################
def time_series_weather_generator(start_date, days=1):
    """
    Generate daily weather for Montreal with realistic seasonal variations.
    Uses winter ranges (Dec-Feb) and summer ranges otherwise.
    """
    weather_data = []
    month = start_date.month
    if month in [12, 1, 2]:
        wind_range = (10, 25)
        precip_range = (0, 5)
        temp_range = (-15, 5)
    else:
        wind_range = (5, 20)
        precip_range = (0, 10)
        temp_range = (15, 30)
    
    current_wind = random.uniform(*wind_range)
    current_precip = random.uniform(*precip_range)
    current_temp = random.uniform(*temp_range)
    
    for i in range(days):
        date_i = start_date + timedelta(days=i)
        if random.random() < 0.05:
            current_wind = random.uniform(wind_range[1] * 0.8, wind_range[1] * 1.2)
            current_precip = random.uniform(precip_range[1] * 0.8, precip_range[1] * 1.2)
            current_temp += random.uniform(-5, 0)
        else:
            current_wind = max(0, current_wind + random.uniform(-2, 2))
            current_precip = max(0.0, current_precip + random.uniform(-0.5, 0.5))
            current_temp += random.uniform(-1, 1)
        weather_data.append({
            "date": date_i,
            "wind_speed": round(current_wind, 2),
            "precipitation": round(current_precip, 2),
            "temperature": round(current_temp, 2)
        })
    return weather_data

def pick_departure_time(date_obj, wave="morning"):
    """
    Assign a departure time based on the time-of-day wave.
    """
    if wave == "morning":
        hour = random.randint(6, 9)
    elif wave == "midday":
        hour = random.randint(11, 14)
    else:
        hour = random.randint(16, 19)
    minute = random.randint(0, 59)
    return datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute)

def get_base_turnaround(aircraft_type, is_international=False):
    """
    Return a base turnaround time based on aircraft type and international status.
    """
    if aircraft_type in ["A320", "B737"]:
        base_min, base_max = 30, 45
    elif aircraft_type in ["A330", "B777", "A350"]:
        base_min, base_max = 60, 120
    else:
        base_min, base_max = 20, 30
    if is_international:
        base_min += 10
        base_max += 30
    return random.randint(base_min, base_max)

def compute_subprocess_times(aircraft_type, pax, cargo, weather, staff):
    """
    Compute fueling, baggage, and cleaning times influenced by various factors.
    """
    temp_factor = 1.0
    if weather["temperature"] < 0:
        temp_factor += 0.1
    staff_baggage = max(1, staff.get("baggage", 5))
    staff_fueling = max(1, staff.get("fueling", 3))
    staff_cleaning = max(1, staff.get("cleaning", 4))
    if aircraft_type in ["A330", "B777", "A350"]:
        fueling_time = random.randint(15, 30) - staff_fueling
    elif aircraft_type in ["A320", "B737"]:
        fueling_time = random.randint(10, 20) - staff_fueling
    else:
        fueling_time = random.randint(5, 15) - staff_fueling
    fueling_time = max(5, fueling_time) * temp_factor
    baggage_time = (cargo / 200.0) + random.randint(5, 10) - (0.5 * staff_baggage)
    baggage_time = max(5, baggage_time) * temp_factor
    cleaning_time = (pax / 50.0) + random.randint(5, 10) - staff_cleaning
    cleaning_time = max(5, cleaning_time) * temp_factor
    return {
        "fueling_time": round(fueling_time, 2),
        "baggage_time": round(baggage_time, 2),
        "cleaning_time": round(cleaning_time, 2)
    }

###########################
# Ultra Hyperrealistic Flight Data Generation for Montreal (YUL)
###########################
def generate_flight_data(num=50, start_date=None, days=1):
    """
    Generate ultra hyperrealistic flight scheduling data for Montreal (YUL).
    All flights are arrivals to Montreal from a mix of Canadian and international origins.
    """
    if start_date is None:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    weather_series = time_series_weather_generator(start_date, days=days)
    weather_map = {w["date"].date(): w for w in weather_series}
    
    flights = []
    flight_id_counter = 1
    possible_aircraft = ["A320", "B737", "A330", "B777", "A350", "CRJ", "Q400"]
    airlines = ["Air Canada", "WestJet", "Air Transat", "Delta", "United", "American Airlines"]
    origins = ["Toronto (YYZ)", "Calgary (YYC)", "Ottawa (YOW)", "Vancouver (YVR)",
               "New York (JFK)", "London (LHR)", "Paris (CDG)", "Dubai (DXB)"]
    destination = "Montreal (YUL)"
    waves = ["morning", "midday", "evening"]
    
    for d in range(days):
        current_day = start_date + timedelta(days=d)
        wdata = weather_map[current_day.date()]
        flights_per_day = int(num // days)
        flights_per_wave = max(1, flights_per_day // 3)
        for wave in waves:
            for _ in range(flights_per_wave):
                a_type = random.choice(possible_aircraft)
                a_line = random.choice(airlines)
                origin_airport = random.choice(origins)
                is_int = 0 if origin_airport in ["Toronto (YYZ)", "Calgary (YYC)", "Ottawa (YOW)", "Vancouver (YVR)"] else 1
                pax = random.randint(50, 350) if not is_int else random.randint(150, 400)
                cargo = random.randint(0, 5000)
                dep_time = pick_departure_time(current_day, wave=wave)
                flight_code = f"{a_line.split()[0][:2].upper()}{random.randint(100,999)}"
                base_turn = get_base_turnaround(a_type, is_international=bool(is_int))
                sp_times = compute_subprocess_times(a_type, pax, cargo, wdata, simulation_state["staff_scheduling"])
                total_sub = sp_times["fueling_time"] + sp_times["baggage_time"] + sp_times["cleaning_time"]
                base_delay = base_turn - total_sub
                if base_delay < 0:
                    base_delay = abs(base_delay) + random.randint(0, 10)
                predicted_delay = round(max(base_delay * 0.2, 0), 2)
                gate_capacity = "Wide" if a_type in ["A330", "B777", "A350"] else "Narrow"
                status = "Landing"
                flights.append({
                    "flight_id": flight_id_counter,
                    "flight_name": flight_code,
                    "eta": dep_time - timedelta(minutes=random.randint(30, 90)),
                    "etd": dep_time,
                    "status": status,
                    "refueling_status": "Not Started",
                    "baggage_status": "Not Started",
                    "taxiing_status": "Ready",
                    "gate_efficiency": random.choice(["Efficient", "Moderate", "Inefficient"]),
                    "maintenance_status": random.choice(["Up-to-date", "Pending", "Overdue"]),
                    "crew_availability": random.choice(["Available", "Delayed", "Not Available"]),
                    "cleaning_status": "Not Started",
                    "turnaround_analytics": f"Fuel:{sp_times['fueling_time']}|Bag:{sp_times['baggage_time']}|Clean:{sp_times['cleaning_time']}",
                    "predicted_delay": predicted_delay,
                    "crew_pool": random.randint(1, simulation_state["num_crew_pools"]),
                    "completed_flights": 0,
                    "flight_category": "International" if is_int else "Domestic",
                    "security_status": random.choice(["Not Cleared", "In Progress", "Cleared"]),
                    "fuel_status": random.randint(50, 100),
                    "emissions": random.randint(1000, 2000),
                    "maintenance_counter": 0,
                    "airline": a_line,
                    "origin": origin_airport,
                    "destination": destination,
                    "passengers": pax,
                    "cargo_weight": cargo,
                    "runway_choice": random.randint(0, 1),
                    "resource_conflict": False,
                    "connection_misconnect": False,
                    "seat_capacity": random.randint(60, 350),
                    "wide_body": (a_type in ["A330", "B777", "A350"]),
                    "overbooked": False,
                    "staff_shortage_factor": random.randint(0, 2),
                    "previous_delay": 0,
                    "is_international": int(is_int),
                    "gate_index": 0,
                    "required_gate_capacity": gate_capacity,
                    "aircraft_type": a_type,
                    "day_wind_speed": wdata["wind_speed"],
                    "day_precipitation": wdata["precipitation"],
                    "day_temperature": wdata["temperature"],
                    "turnaround_sequence": None,
                    "current_task_index": None,
                    "task_remaining_time": None
                })
                flight_id_counter += 1
    df = pd.DataFrame(flights)
    gate_coords = [
        (45.4710, -73.7420),
        (45.4712, -73.7410),
        (45.4708, -73.7400),
        (45.4705, -73.7390)
    ]
    def create_taxi_route(gate_coord, runway_choice):
        runway_coords = [(45.4712, -73.7490), (45.4720, -73.7515)]
        chosen_runway = runway_coords[runway_choice]
        return [chosen_runway, (45.4710, -73.7460), (45.4708, -73.7430), gate_coord]
    df["gate_index"] = df.apply(lambda row: random.randint(0, len(gate_coords)-1), axis=1)
    df["taxi_route"] = [
        create_taxi_route(gate_coords[row["gate_index"]], row["runway_choice"])
        if row["status"] == "Taxiing" else []
        for _, row in df.iterrows()
    ]
    df["overbooked"] = df.apply(lambda r: r["passengers"] > r["seat_capacity"], axis=1)
    return df

###########################
# Initialize flights_data and dynamic_positions
###########################
flights_data = generate_flight_data(num=50, start_date=datetime.now(), days=1)
for col in ["turnaround_sequence", "current_task_index", "task_remaining_time"]:
    if col not in flights_data.columns:
        flights_data[col] = None

dynamic_positions = pd.DataFrame({
    "flight_id": flights_data["flight_id"],
    "lat": [0.0] * len(flights_data),
    "lon": [0.0] * len(flights_data),
    "progress": [0.0] * len(flights_data),
    "segment": [0] * len(flights_data)
})

def get_flight_position_static(f):
    hanger_coord = (45.4695, -73.7410)
    runway_coords = [(45.4712, -73.7490), (45.4720, -73.7515)]
    gate_coords = [
        (45.4710, -73.7420),
        (45.4712, -73.7410),
        (45.4708, -73.7400),
        (45.4705, -73.7390)
    ]
    airport_center = (45.4710, -73.7408)
    if f["status"] == "Runway":
        return runway_coords[f["runway_choice"]]
    elif f["status"] == "Hanger":
        return hanger_coord
    elif f["status"] == "Gate":
        return gate_coords[f["gate_index"]]
    elif f["status"] == "Maintenance":
        return (45.4692, -73.7405)
    return airport_center

for i in range(len(flights_data)):
    row = flights_data.iloc[i]
    pos = get_flight_position_static(row)
    dynamic_positions.at[i, "lat"] = pos[0]
    dynamic_positions.at[i, "lon"] = pos[1]

###########################
# Ground Vehicles
###########################
def initialize_vehicles(num_veh=5):
    from collections import deque
    def shortest_path(graph, start, end):
        visited = {start}
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path
            neighbors = ([k[1] for k in simulation_state["airport_edges"].keys() if k[0] == node] +
                         [k[0] for k in simulation_state["airport_edges"].keys() if k[1] == node])
            for neigh in neighbors:
                if neigh not in visited:
                    visited.add(neigh)
                    new_path = list(path)
                    new_path.append(neigh)
                    queue.append(new_path)
        return [start]
    
    node_names = list(simulation_state["airport_nodes"].keys())
    for v_id in range(1, num_veh+1):
        start_node = random.choice(node_names)
        end_node = random.choice(node_names)
        while end_node == start_node:
            end_node = random.choice(node_names)
        p_nodes = shortest_path(simulation_state["airport_edges"], start_node, end_node)
        rcoords = [simulation_state["airport_nodes"][n] for n in p_nodes]
        simulation_state["vehicles"][v_id] = {
            "type": random.choice(["Fuel Truck", "Baggage Cart", "Passenger Bus", "Pushback Tractor"]),
            "node_path": p_nodes,
            "segment": 0,
            "progress": 0.0,
            "lat": simulation_state["airport_nodes"][start_node][0],
            "lon": simulation_state["airport_nodes"][start_node][1],
            "route_coords": rcoords
        }

initialize_vehicles()

def update_ground_vehicles():
    for v_id, v_info in simulation_state["vehicles"].items():
        route = v_info["route_coords"]
        seg = v_info["segment"]
        prog = v_info["progress"] + 0.1
        if seg >= len(route) - 1:
            simulation_state["event_log"].append(
                (simulation_state["current_time"], f"{v_info['type']} (ID {v_id}) finished route; re-init.")
            )
            initialize_vehicles(1)
        else:
            if prog >= 1.0:
                v_info["segment"] += 1
                v_info["progress"] = 0.0
                new_seg = v_info["segment"]
                v_info["lat"] = route[new_seg][0]
                v_info["lon"] = route[new_seg][1]
            else:
                start_c = route[seg]
                end_c = route[seg+1]
                v_info["lat"] = start_c[0] + prog * (end_c[0] - start_c[0])
                v_info["lon"] = start_c[1] + prog * (end_c[1] - start_c[1])
                v_info["progress"] = prog

###########################
# Anomaly Detection
###########################
anomaly_forest = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
sample_delays = np.random.normal(10, 5, (50, 1))
anomaly_forest.fit(sample_delays)
simulation_state["anomaly_detector"] = anomaly_forest

###########################
# Weather & Simulation Updates
###########################
def mocked_fetch_weather_api_data():
    return {
        "wind_speed": random.uniform(5, 35),
        "precipitation": random.choice([0, 0.1, 0.3, 0.5]),
        "temperature": random.uniform(-10, 30)
    }

def update_weather():
    neww = mocked_fetch_weather_api_data()
    simulation_state["weather"].update(neww)
    simulation_state["weather_history"].append({
        "time": simulation_state["current_time"],
        "wind_speed": neww["wind_speed"],
        "precipitation": neww["precipitation"],
        "temperature": neww["temperature"]
    })

def update_runway_conditions():
    for r_idx, r_info in simulation_state["runways"].items():
        if r_info["status"] == "Open":
            if simulation_state["current_time"] >= r_info["next_maintenance"]:
                r_info["status"] = "Maintenance"
                simulation_state["event_log"].append((simulation_state["current_time"], f"Runway {r_idx} in Maintenance."))
            if simulation_state["weather"]["precipitation"] > 0.5 or simulation_state["weather"]["wind_speed"] > 30:
                r_info["status"] = "Closed"
                simulation_state["event_log"].append((simulation_state["current_time"], f"Runway {r_idx} closed (weather)."))
        else:
            if random.random() < 0.05:
                r_info["status"] = "Open"
                r_info["next_maintenance"] = simulation_state["current_time"] + timedelta(minutes=random.randint(60, 120))
                simulation_state["event_log"].append((simulation_state["current_time"], f"Runway {r_idx} reopened."))

def update_crew_pools():
    before = simulation_state["crew_pools"].copy()
    for pool, status in simulation_state["crew_pools"].items():
        if status == "Available":
            if simulation_state["weather"]["precipitation"] > 0.3 and random.random() < 0.1:
                simulation_state["crew_pools"][pool] = "Delayed"
            elif random.random() < 0.05:
                simulation_state["crew_pools"][pool] = "Delayed"
        elif status == "Delayed":
            if random.random() < 0.1:
                simulation_state["crew_pools"][pool] = "Not Available"
            elif random.random() < 0.2:
                simulation_state["crew_pools"][pool] = "Available"
        else:
            if random.random() < 0.2:
                simulation_state["crew_pools"][pool] = "Delayed"
    after = simulation_state["crew_pools"].copy()
    simulation_state["crew_history"].append((simulation_state["current_time"], after))

def staff_shift_logic():
    if not simulation_state["disable_advanced_scheduling"]:
        for duty in simulation_state["staff_hours_remaining"]:
            usage = random.uniform(0.1, 0.5)
            simulation_state["staff_hours_remaining"][duty] = max(0, simulation_state["staff_hours_remaining"][duty] - usage)
            if simulation_state["staff_hours_remaining"][duty] <= 0:
                simulation_state["staff_scheduling"][duty] = 0
        if random.random() < 0.01:
            for duty in simulation_state["staff_hours_remaining"]:
                simulation_state["staff_hours_remaining"][duty] = random.randint(20, 50)
            simulation_state["staff_scheduling"] = {"baggage": 5, "fueling": 3, "pushback": 2, "cleaning": 4, "security": 6}
            simulation_state["event_log"].append((simulation_state["current_time"], "Staff shift changed, hours reset."))

def free_gates():
    for idx, gate in enumerate(simulation_state["gates"]):
        occ = gate["flight_id"]
        if occ is not None:
            f_idx = flights_data.index[flights_data["flight_id"] == occ].tolist()
            if f_idx:
                i = f_idx[0]
                f = flights_data.iloc[i]
                if simulation_state["current_time"] > f["etd"]:
                    if simulation_state["current_time"] <= f["etd"]:
                        simulation_state["on_time_departures"] += 1
                    simulation_state["total_departures"] += 1
                    flights_data.at[i, "status"] = "Runway"
                    simulation_state["gates"][idx]["flight_id"] = None
                    flights_data.at[i, "completed_flights"] += 1
                    flights_data.at[i, "maintenance_counter"] += 1
                    simulation_state["event_log"].append(
                        (simulation_state["current_time"], f"{f['flight_name']} departed Gate {idx}.")
                    )

###########################
# Turnaround Sequence Update Function (Updated)
###########################
def update_turnaround_sequences():
    """
    Process flights in 'Turnaround' status by decrementing the current task's remaining time.
    When a task is complete, advance to the next task.
    When all tasks are completed, mark the flight as 'Ready for Departure'.
    This version includes a check to ensure task_remaining_time is not None.
    """
    time_step_minutes = simulation_state["time_step"].seconds / 60  # 5 minutes
    for i in range(len(flights_data)):
        if flights_data.at[i, "status"] == "Turnaround":
            remaining = flights_data.at[i, "task_remaining_time"]
            # Safety check: if remaining is None, skip processing this flight.
            if remaining is None:
                continue
            new_remaining = remaining - time_step_minutes
            seq = flights_data.at[i, "turnaround_sequence"]
            cur_index = flights_data.at[i, "current_task_index"]
            if new_remaining <= 0:
                simulation_state["event_log"].append(
                    (simulation_state["current_time"], f"Flight {flights_data.at[i, 'flight_name']}: Completed task '{seq[cur_index]['name']}'")
                )
                cur_index += 1
                if cur_index >= len(seq):
                    flights_data.at[i, "status"] = "Ready for Departure"
                    flights_data.at[i, "turnaround_sequence"] = None
                    flights_data.at[i, "current_task_index"] = None
                    flights_data.at[i, "task_remaining_time"] = None
                    simulation_state["event_log"].append(
                        (simulation_state["current_time"], f"Flight {flights_data.at[i, 'flight_name']} is Ready for Departure.")
                    )
                else:
                    flights_data.at[i, "current_task_index"] = cur_index
                    flights_data.at[i, "task_remaining_time"] = seq[cur_index]["duration"] + new_remaining
            else:
                flights_data.at[i, "task_remaining_time"] = new_remaining

###########################
# Updated Positions Function
###########################
def update_positions():
    weather = simulation_state["weather"]
    speed_factor = 1.0
    if weather["wind_speed"] > 15:
        speed_factor -= 0.2
    if weather["precipitation"] > 0.3:
        speed_factor -= 0.1
    approach_coord = (45.4740, -73.7550)
    runway_coords = [(45.4712, -73.7490), (45.4720, -73.7515)]
    
    for i in range(len(flights_data)):
        f = flights_data.iloc[i]
        if f["status"] == "Landing":
            flights_data.at[i, "status"] = "Turnaround"
            initialize_turnaround_for_flight(i)
            simulation_state["event_log"].append(
                (simulation_state["current_time"], f"Flight {f['flight_name']} changed from Landing to Turnaround.")
            )
            continue
        
        if f["status"] == "Turnaround":
            update_turnaround_sequences()
            ramp_position = (45.4700, -73.7400)
            dynamic_positions.at[i, "lat"] = ramp_position[0]
            dynamic_positions.at[i, "lon"] = ramp_position[1]
            continue
        
        if f["status"] == "Runway":
            if random.random() < 0.1:
                flights_data.at[i, "status"] = "Taxiing"
                dynamic_positions.at[i, "segment"] = 0
                dynamic_positions.at[i, "progress"] = 0.0
        elif f["status"] == "Taxiing":
            route = f.get("taxi_route", [])
            if not route:
                pos = get_flight_position_static(f)
                dynamic_positions.at[i, "lat"] = pos[0]
                dynamic_positions.at[i, "lon"] = pos[1]
            else:
                seg = dynamic_positions.at[i, "segment"]
                if seg >= len(route) - 1:
                    g_idx = f["gate_index"]
                    if (g_idx < len(simulation_state["gates"]) and simulation_state["gates"][g_idx]["flight_id"] is None):
                        flights_data.at[i, "status"] = "Gate"
                        flights_data.at[i, "eta"] = simulation_state["current_time"]
                        simulation_state["gates"][g_idx]["flight_id"] = f["flight_id"]
                    else:
                        flights_data.at[i, "status"] = "Hanger"
                    dynamic_positions.at[i, "lat"] = route[-1][0]
                    dynamic_positions.at[i, "lon"] = route[-1][1]
                else:
                    start_coord = route[seg]
                    end_coord = route[seg + 1]
                    prog = dynamic_positions.at[i, "progress"] + (0.1 * speed_factor)
                    if prog >= 1.0:
                        dynamic_positions.at[i, "segment"] = seg + 1
                        dynamic_positions.at[i, "progress"] = 0.0
                        dynamic_positions.at[i, "lat"] = end_coord[0]
                        dynamic_positions.at[i, "lon"] = end_coord[1]
                    else:
                        lat0, lon0 = start_coord
                        lat1, lon1 = end_coord
                        new_lat = lat0 + prog * (lat1 - lat0)
                        new_lon = lon0 + prog * (lon1 - lon0)
                        dynamic_positions.at[i, "lat"] = new_lat
                        dynamic_positions.at[i, "lon"] = new_lon
                        dynamic_positions.at[i, "progress"] = prog
        else:
            pos = get_flight_position_static(f)
            dynamic_positions.at[i, "lat"] = pos[0]
            dynamic_positions.at[i, "lon"] = pos[1]

###########################
# Remaining Functions
###########################
def check_flight_connections():
    pass

def detect_anomalies_on_delay():
    for idx, f in flights_data.iterrows():
        dd = f["predicted_delay"]
        simulation_state["delay_buffer"].append([dd])
        if len(simulation_state["delay_buffer"]) > 200:
            simulation_state["delay_buffer"].pop(0)
        arr = np.array(simulation_state["delay_buffer"])
        preds = simulation_state["anomaly_detector"].predict(arr[-1:])
        if preds[0] == -1:
            simulation_state["event_log"].append(
                (simulation_state["current_time"], f"ANOMALY DETECTED: {f['flight_name']} => delay={dd}.")
            )

def run_digital_twin_simulation():
    simulation_state["current_time"] += simulation_state["time_step"]
    update_weather()
    update_runway_conditions()
    update_crew_pools()
    free_gates()
    staff_shift_logic()
    update_ground_vehicles()
    update_positions()
    check_flight_connections()
    detect_anomalies_on_delay()

###########################
# Crew & Maintenance Factor Helpers
###########################
def crew_factor_for(status):
    if status == "Available":
        return 0
    elif status == "Delayed":
        return 1
    else:
        return 2

def maintenance_factor_for(status):
    if status == "Up-to-date":
        return 0
    elif status == "Pending":
        return 1
    else:
        return 2

###########################
# Colors, Map, and Layout Settings
###########################
mapbox_token = "YOUR_MAPBOX_TOKEN"  # Replace with a valid token
airport_center = (45.4706, -73.7408)
status_colors = {
    "Runway": "#ff4d4d",
    "Hanger": "#9966ff",
    "Gate": "#33cc33",
    "Taxiing": "#ffa500",
    "Maintenance": "#778899"
}
misconnect_color = "#800080"
map_styles = [
    {"label": "Open Street Map", "value": "open-street-map"},
    {"label": "Stamen Terrain", "value": "stamen-terrain"},
    {"label": "Carto Positron", "value": "carto-positron"},
    {"label": "Carto Darkmatter", "value": "carto-darkmatter"}
]
runway_polygons = [
    [(45.4715, -73.7500), (45.4715, -73.7480), (45.4709, -73.7480), (45.4709, -73.7500)],
    [(45.4723, -73.7520), (45.4723, -73.7500), (45.4717, -73.7500), (45.4717, -73.7520)]
]
terminal_polygons = [
    [(45.4708, -73.7415), (45.4708, -73.7385), (45.4695, -73.7385), (45.4695, -73.7415)],
    [(45.4692, -73.7420), (45.4692, -73.7370), (45.4685, -73.7370), (45.4685, -73.7420)]
]

###########################
# Navbar, Modal, and Download Components
###########################
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Simulation Dashboard", href="/")),
        dbc.NavItem(dbc.NavLink("Manual Input", href="/manual")),
        dbc.NavItem(dbc.NavLink("About Us", href="/about")),
        dbc.NavItem(dbc.NavLink("Logout", href="/logout", id="logout-link", style={"display": "none"}))
    ],
    brand="YUL Digital Twin",
    brand_href="/",
    color="dark",
    dark=True
)

download_link_input = dcc.Download(id="download-input-csv")
download_link_predictions = dcc.Download(id="download-predictions-csv")

modal = dbc.Modal(
    [
        dbc.ModalHeader("Event Log"),
        dbc.ModalBody(id="event-log-body", style={"maxHeight": "400px", "overflowY": "auto"}),
        dbc.ModalFooter(dbc.Button("Close", id="close-event-log", className="ml-auto", n_clicks=0)),
    ],
    id="event-log-modal",
    size="lg"
)

analysis_card = dbc.Card([
    dbc.CardHeader("YUL Airport Control Center", style={"fontWeight": "bold", "fontSize": "1.2em"}),
    dbc.CardBody([
        html.Div(id="digital-clock", style={"fontSize": "1.5em", "textAlign": "center", "marginBottom": "10px"}),
        html.P(id="runway-count"),
        html.P(id="gate-count"),
        html.P(id="taxiing-count"),
        html.P(id="hanger-count"),
        html.P(id="avg-delay"),
        html.P(id="avg-emissions"),
        html.P(id="security-cleared-percent"),
        html.Span("Operational Cost: ", style={"fontWeight": "bold"}),
        html.Span(id="operational-cost", style={"fontWeight": "bold", "marginLeft": "5px"}),
        html.Br(),
        html.P(id="departed-flights-count"),
        html.P(id="on-time-percent"),
        dbc.Button("View Event Log", id="open-event-log", className="mt-2", color="secondary", outline=True),
        html.Br(),
        dbc.Checklist(
            options=[{"label": "Disable Advanced Scheduling", "value": "disable"}],
            value=[],
            id="advanced-scheduling-checklist",
            inline=True
        ),
        html.P("Multi-day staff scheduling, advanced gate constraints, seat capacity checks, pushback/taxi times, etc.")
    ])
], className="my-4")

download_card = dbc.Card([
    dbc.CardHeader("Download Data", style={"fontWeight": "bold", "fontSize": "1.2em"}),
    dbc.CardBody([
        html.P("Download flight input data (numeric features):", style={"marginBottom": "10px"}),
        dbc.Button("Download Input CSV", id="btn_download_input", color="primary", style={"marginBottom": "10px"}),
        html.Br(),
        html.P("Download predictions (actual vs. predicted), sub-delays, etc.:", style={"marginBottom": "10px"}),
        dbc.Button("Download Predictions CSV", id="btn_download_predictions", color="success")
    ])
], className="my-4")

###########################
# Simulation Dashboard Layout
###########################
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Average Delay Time"),
                dbc.CardBody(html.H2(id="avg-delay-kpi", className="card-text"))
            ], color="info", inverse=True),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Flights On Time"),
                dbc.CardBody(html.H2(id="flights-on-time-kpi", className="card-text"))
            ], color="success", inverse=True),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Gate Efficiency"),
                dbc.CardBody(html.H2(id="gate-efficiency-kpi", className="card-text"))
            ], color="warning", inverse=True),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Runway Efficiency"),
                dbc.CardBody(html.H2(id="runway-efficiency-kpi", className="card-text"))
            ], color="primary", inverse=True),
            width=3
        )
    ], className="my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Flight Selector"),
                dbc.CardBody([
                    html.Label("Select Flight:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Dropdown(
                        id="flight-selector",
                        options=[
                            {"label": f"{flights_data.iloc[i]['flight_name']} ({flights_data.iloc[i]['origin']} -> {flights_data.iloc[i]['destination']})", "value": i}
                            for i in range(len(flights_data))
                        ],
                        value=0,
                        style={"marginBottom": "20px"}
                    ),
                    html.Div(id="flight-details")
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Process Tracker"),
                dbc.CardBody([
                    dcc.Graph(
                        id="gantt-chart",
                        style={
                            "height": "700px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=9)
    ], className="my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Flight Status Distribution"),
                dbc.CardBody([
                    dcc.Graph(
                        id="status-pie-chart",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Gate Efficiency Trends"),
                dbc.CardBody([
                    dcc.Graph(
                        id="efficiency-line-chart",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predicted vs Actual Delays"),
                dbc.CardBody([
                    dcc.Graph(
                        id="prediction-scatter",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4)
    ], className="my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Maintenance Prediction"),
                dbc.CardBody([
                    dcc.Graph(
                        id="maintenance-prediction",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Emissions per Flight"),
                dbc.CardBody([
                    dcc.Graph(
                        id="emissions-bar-chart",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Crew Pool Status Over Time"),
                dbc.CardBody([
                    dcc.Graph(
                        id="crew-status-chart",
                        style={
                            "height": "400px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    )
                ])
            ], className="mb-4")
        ], width=4)
    ], className="my-4"),
    
    html.Hr(),
    dbc.Row([dbc.Col(analysis_card, width=12)]),
    html.Hr(),
    dbc.Row([dbc.Col(download_card, width=12)]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Map Settings & Visualization"),
                dbc.CardBody([
                    html.Label("Select Map Style:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Dropdown(
                        id="map-style-dropdown",
                        options=map_styles,
                        value="open-street-map",
                        clearable=False
                    ),
                    html.Br(),
                    html.Label("Map Bearing:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Slider(id="bearing-slider", min=0, max=360, step=10, value=0),
                    html.Br(),
                    html.Label("Map Pitch:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Slider(id="pitch-slider", min=0, max=60, step=5, value=0),
                    html.Br(),
                    html.Label("Show Airport Polygons:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                    dcc.Checklist(
                        id="polygon-checklist",
                        options=[{"label": "Show Runways & Terminals", "value": "show"}],
                        value=["show"]
                    ),
                    html.H4("Airport Map (YUL)", className="text-center", style={"marginTop": "20px"}),
                    dcc.Graph(
                        id="airport-map",
                        style={
                            "height": "600px",
                            "border": "2px solid #ccc",
                            "borderRadius": "5px",
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5"
                        }
                    ),
                    dcc.Interval(id="position-update-interval", interval=5000, n_intervals=0)
                ])
            ], className="mb-4")
        ], width=12)
    ], className="my-4"),
    
    modal,
    download_link_input,
    download_link_predictions
], fluid=True)

####################
# Manual Input Layout
####################
manual_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H3("Manual Delay Prediction", className="text-center text-primary")
                        ),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Select Flight:", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id="manual-flight-selector",
                                                    options=[
                                                        {"label": f"{flights_data.iloc[i]['flight_name']} (ID:{flights_data.iloc[i]['flight_id']})", "value": i}
                                                        for i in range(len(flights_data))
                                                    ],
                                                    value=0,
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Wind Speed (km/h):", className="fw-bold"),
                                                dcc.Input(id="input-wind-speed", type="number", value=10, min=0, max=60, step=1),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Precipitation (mm):", className="fw-bold"),
                                                dcc.Input(id="input-precipitation", type="number", value=0.1, step=0.1),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Maintenance Status (0=Up-to-date, 1=Pending, 2=Overdue):", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id="input-maintenance",
                                                    options=[
                                                        {"label": "Up-to-date (0)", "value": 0},
                                                        {"label": "Pending (1)", "value": 1},
                                                        {"label": "Overdue (2)", "value": 2}
                                                    ],
                                                    value=0,
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Crew Status (0=Available, 1=Delayed, 2=Not Available):", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id="input-crew",
                                                    options=[
                                                        {"label": "Available (0)", "value": 0},
                                                        {"label": "Delayed (1)", "value": 1},
                                                        {"label": "Not Available (2)", "value": 2}
                                                    ],
                                                    value=0,
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Previous Delay (mins):", className="fw-bold"),
                                                dcc.Input(id="input-prev-delay", type="number", value=0, min=0, step=1),
                                            ],
                                            md=4,
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("International Flight? (0=No, 1=Yes):", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id="input-international",
                                                    options=[
                                                        {"label": "No (0)", "value": 0},
                                                        {"label": "Yes (1)", "value": 1}
                                                    ],
                                                    value=0,
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Overbooked? (0=No, 1=Yes):", className="fw-bold"),
                                                 dcc.Dropdown(
                                                    id="input-overbooked",
                                                    options=[
                                                        {"label": "No (0)", "value": 0},
                                                        {"label": "Yes (1)", "value": 1}
                                                    ],
                                                    value=0,
                                                    clearable=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Staff Shortage Factor (0-2):", className="fw-bold"),
                                                dcc.Input(id="input-staff-shortage", type="number", value=0, min=0, max=2, step=1),
                                            ],
                                            md=4,
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Passenger Load:", className="fw-bold"),
                                                dcc.Input(id="input-passenger-load", type="number", value=150, min=0, step=1),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Congestion Index (0-10):", className="fw-bold"),
                                                dcc.Input(id="input-congestion-index", type="number", value=5, min=0, max=10, step=0.1),
                                            ],
                                            md=6,
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H5("Process Staff Override:", className="text-center mb-2"),
                                                html.Label("Baggage Staff:", className="fw-bold"),
                                                dcc.Input(id="manual-baggage-staff", type="number", value=5, min=0, step=1),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Fueling Staff:", className="fw-bold"),
                                                dcc.Input(id="manual-fueling-staff", type="number", value=3, min=0, step=1),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Pushback Staff:", className="fw-bold"),
                                                dcc.Input(id="manual-pushback-staff", type="number", value=2, min=0, step=1),
                                            ],
                                            md=4,
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Cleaning Staff:", className="fw-bold"),
                                                dcc.Input(id="manual-cleaning-staff", type="number", value=4, min=0, step=1),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Security Staff:", className="fw-bold"),
                                                dcc.Input(id="manual-security-staff", type="number", value=6, min=0, step=1),
                                            ],
                                            md=6,
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Checklist(
                                                options=[{"label": "Apply Special Event Delay (+5 mins)", "value": "special"}],
                                                value=[],
                                                id="input-special-event",
                                                inline=True
                                            ),
                                            md=12
                                        )
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Button("Calculate Delay", id="btn-manual-calc", color="primary", className="w-100"),
                                html.Br(), html.Br(),
                                html.Div(id="manual-prediction-result", style={"border": "1px solid #ccc", "padding": "10px", "borderRadius": "5px"}),
                                html.Br(),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Calculation Explanation"),
                                        dbc.CardBody(
                                            html.P(
                                                "This calculation uses two approaches:\n\n"
                                                "1. ML Model Prediction:\n"
                                                "   - Your inputs (weather, maintenance, crew, etc.) are fed into a trained ML model\n"
                                                "     to predict a base turnaround delay for the flight.\n"
                                                "   - If a special event is selected, an additional 5 minutes is added.\n\n"
                                                "2. Optimization Model Prediction:\n"
                                                "   - The ML-predicted delay is used as the required turnaround duration in a mathematical\n"
                                                "     optimization model that determines an optimized departure time (minimizing delay and cost).\n"
                                                "   - The optimized delay is the difference between this optimized departure time and the\n"
                                                "     scheduled departure time (set here as 20 minutes for demonstration).\n\n"
                                                "Both results are shown for you to compare the direct prediction with the optimized schedule."
                                            )
                                        )
                                    ],
                                    className="mt-4"
                                )
                            ]
                        )
                    ],
                    style={
                        "maxWidth": "900px",
                        "margin": "auto",
                        "marginTop": "50px",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.1)",
                        "borderRadius": "10px",
                        "backgroundColor": "#ffffff"
                    }
                )
            )
        )
    ],
    fluid=True,
    style={
        "background": "linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), url('/assets/yul_runway_hazy.jpg')",
        "backgroundSize": "cover",
        "backgroundPosition": "center",
        "minHeight": "100vh",
        "paddingTop": "20px",
        "paddingBottom": "20px"
    }
)

####################
# Manual Input Callback (Integrated ML and Optimization Models)
####################
# Train the main ML model once (global variable)
best_main_model, _ = train_main_models(1500)

@callback(
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
    State("input-passenger-load", "value"),
    State("input-congestion-index", "value"),
    State("input-special-event", "value"),
    prevent_initial_call=True
)
def calculate_manual_prediction(n_clicks, flight_sel, wind_sp, precip, maint, crew,
                                prev_delay, international, overbooked, staff_shortage,
                                passenger_load, congestion_index, special_event):
    """
    This callback uses both the ML model and the optimization model to compute
    a manual delay prediction for Montreal (YUL).
    
    1. It creates a feature dictionary from your inputs and uses the ML model to predict a base turnaround delay.
       If a special event is applied, 5 minutes are added.
    2. It then builds a parameter dictionary for an optimization model where the ML-predicted delay is used as the
       required turnaround duration (T_req).
    3. The optimization model computes an optimized departure time, and the optimized delay is the difference between
       that time and the scheduled departure time.
    
    Both delay estimates are shown for comparison.
    """
    current_dt = datetime.now()
    features = {
        "wind_speed": wind_sp,
        "precipitation": precip,
        "maintenance_factor": maint,
        "crew_factor": crew,
        "previous_delay": prev_delay,
        "is_international": international,
        "overbooked": overbooked,
        "staff_shortage_factor": staff_shortage,
        "day_of_week": current_dt.weekday(),
        "hour": current_dt.hour,
        "passenger_load": passenger_load,
        "congestion_index": congestion_index
    }
    
    ml_delay = rf_predict_delay(best_main_model, features)
    special_adjustment = 5 if "special" in special_event else 0
    ml_delay_adjusted = ml_delay + special_adjustment

    params = {
        'L': {1: 0},
        'U': {1: 100},
        'D0': {1: 20},  # Scheduled departure time for flight 1
        'T_req': {1: ml_delay_adjusted},  # Required turnaround time from ML prediction
        'p': {1: 1},
        'delta': {(1, 'Gate1'): 1, (1, 'Gate2'): 1},
        'M': 1000,
        'r': {1: 2},
        'R': {t: 10 for t in range(1, 11)},
        'alpha': {p: 5 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security']},
        'beta': {(p, 1): 1 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security']},
        'a': {(g, t): 1 for g in ['Gate1', 'Gate2'] for t in range(1, 11)},
        'C_gate': {(1, g): 1 for g in ['Gate1', 'Gate2']},
        'C_proc': {(p, 1): 1 for p in ['baggage', 'fueling', 'pushback', 'cleaning', 'security']}
    }
    flights_dict = {1: {}}
    gates = ['Gate1', 'Gate2']
    time_slots = list(range(1, 11))
    processes = ['baggage', 'fueling', 'pushback', 'cleaning', 'security']
    crews = ['Crew1', 'Crew2']
    
    opt_solution = optimize_turnaround_schedule(flights_dict, gates, time_slots, processes, crews, params)
    optimized_departure = opt_solution["x"][1]
    optimized_delay = optimized_departure - params["D0"][1]
    
    output_text = (
        "Manual Delay Prediction Results:\n\n"
        "1. ML Model Prediction:\n"
        f"   Base predicted turnaround delay: {ml_delay:.2f} minutes\n"
        f"   (After special event adjustment: {ml_delay_adjusted:.2f} minutes)\n\n"
        "2. Optimization Model Prediction:\n"
        f"   Optimized departure time: {optimized_departure:.2f} minutes\n"
        f"   Scheduled departure time: {params['D0'][1]} minutes\n"
        f"   Optimized delay: {optimized_delay:.2f} minutes\n\n"
        "Explanation:\n"
        "   The ML model uses your inputs to estimate how long the turnaround will take.\n"
        "   That estimated delay is then used in a mathematical optimization model that determines\n"
        "   an optimized departure time (minimizing delay penalties and resource costs).\n"
        "   The difference between the optimized departure time and the scheduled departure time\n"
        "   is the optimized delay. Both results are shown for comparison."
    )
    
    return html.Pre(output_text, style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})

###########################
# Additional Visual Functions
###########################
def get_model_performance_figure(model_scores):
    metrics = []
    for name, score in model_scores.items():
        metrics.append({"Model": name, "Metric": "RMSE", "Value": score["rmse"]})
        metrics.append({"Model": name, "Metric": "MAPE (%)", "Value": score["mape"]})
        metrics.append({"Model": name, "Metric": "MAD", "Value": score["mad"]})
    df_metrics = pd.DataFrame(metrics)
    import plotly.express as px
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

###########################
# Flight Details Callback
###########################
@callback(
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
            rows.append(html.Tr([html.Td(html.B(label), style={"padding": "4px 8px"}),
                                  html.Td(str(value), style={"padding": "4px 8px"})]))
        return html.Div([
            html.H5(title, className="mt-3 mb-2"),
            dbc.Table([html.Tbody(rows)],
                      bordered=True,
                      striped=True,
                      hover=True,
                      responsive=True,
                      style={"fontSize": "14px", "marginBottom": "10px"})
        ])
    
    content = html.Div([
        html.H4("Flight Details", className="text-center mb-3"),
        make_section("Basic Information", basic_info),
        make_section("Timing Information", timing_info),
        make_section("Status Information", status_info),
        make_section("Delay Information", delay_info),
        make_section("Additional Information", extra_info)
    ], style={"maxHeight": "500px", "overflowY": "auto", "padding": "10px", "border": "1px solid #ccc", "borderRadius": "5px"})
    
    return content

###########################
# Update Airport Map Callback
###########################
@callback(
    Output("airport-map", "figure"),
    Input("flight-selector", "value"),
    Input("map-style-dropdown", "value"),
    Input("position-update-interval", "n_intervals"),
    Input("bearing-slider", "value"),
    Input("pitch-slider", "value"),
    Input("polygon-checklist", "value")
)
def update_airport_map(sel_flight, map_style, n_intervals, bearing, pitch, polygon_vals):
    from dash import dcc
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
    if "show" in polygon_vals:
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
        hov_txt = (
            f"{f['flight_name']}<br>"
            f"Airline: {f['airline']}<br>"
            f"Route: {f['origin']} -> {f['destination']}<br>"
            f"ETA: {f['eta'].strftime('%H:%M')} / ETD: {f['etd'].strftime('%H:%M')}"
        )
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=25, symbol="airport-15"),
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

###########################
# Position Update Interval Callback
###########################
@callback(
    Output("position-update-interval", "n_clicks"),
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
    Input("position-update-interval", "n_clicks"),
    State("advanced-scheduling-checklist", "value")
)
def trigger_position_update(n_clicks, adv_values):
    simulation_state["disable_advanced_scheduling"] = ("disable" in adv_values)
    run_digital_twin_simulation()
    r_count = (flights_data["status"] == "Runway").sum()
    g_count = (flights_data["status"] == "Gate").sum()
    t_count = (flights_data["status"] == "Taxiing").sum()
    h_count = (flights_data["status"] == "Hanger").sum()
    avg_delay = 5
    on_time_pct = 95
    gate_eff = 98
    runway_eff = 99
    a_em = round(flights_data["emissions"].mean(), 2)
    tot = len(flights_data)
    sec_str = "Security Cleared: 100%" if tot > 0 else "Security Cleared: N/A"
    cost_str = f"${round(simulation_state['operational_cost'], 2)}"
    departed_str = f"Departed: {simulation_state['total_departures']}"
    ot_str = f"On-Time Departures: {on_time_pct}%"
    return (
        n_clicks,
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

###########################
# Clock Callback
###########################
@callback(
    Output("digital-clock", "children"),
    Input("position-update-interval", "n_clicks")
)
def update_digital_clock(_):
    return simulation_state["current_time"].strftime("%Y-%m-%d %H:%M:%S")

###########################
# Event Log Modal Callback
###########################
@callback(
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

###########################
# Download Input CSV Callback
###########################
@callback(
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

###########################
# Download Predictions CSV Callback
###########################
@callback(
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

###########################
# Main Execution Block (for testing)
###########################
if __name__ == "__main__":
    from dash import dcc, html
    app_layout = dbc.Container([
        dcc.Location(id="url"),
        navbar,
        home_layout  # default view for testing
    ], fluid=True)
    from dash import Dash
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = app_layout
    app.run_server(debug=True)