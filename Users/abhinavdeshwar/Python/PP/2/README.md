# Supply Chain Demand Forecasting App

This Dash application provides interactive forecasting for supply-chain planning.

## Setup

1. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run the app**

```bash
python app.py
```

3. **(VS Code)** Switch interpreter to the `venv` under workspace settings.

## Manual Test Checklist

- **Add Row should** append the entered data to the history table and update the chart.
- **Clear All should** remove all rows from the history table and chart.
- **Run Forecast should** compute forecasts and display the best model with metrics.

## Screenshots

*Add screenshots here*

## Requirements

See `requirements.txt` for exact package versions.
