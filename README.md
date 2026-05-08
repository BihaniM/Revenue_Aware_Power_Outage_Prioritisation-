# Revenue-Aware Power Outage Prioritisation

FastAPI application for telecom NOC support that ranks power-related site outages by combining live alarm severity, traffic-derived revenue impact, and machine learning predictions of outage progression.

## What It Does

This system is designed for situations where field resources are limited and operations teams must decide which sites to restore first. It combines:

1. KPI-derived revenue importance per site and hour.
2. Power alarm progression across mains failure, DC degradation, and site-down stages.
3. Predicted outage timings using a multi-output regression model.
4. A dashboard for operators to review priorities, predicted loss windows, and recent imports.

## Core Features

- Import KPI and power alarm files through the web UI.
- Recalculate revenue-aware site priority rankings.
- Predict `mains_to_dc`, `dc_to_down`, and `site_down_duration` for active incidents.
- Export ranked prediction results as CSV.
- Manage users with simple role-based access for admin and common users.
- Run offline with SQLite or against PostgreSQL/Supabase through `DATABASE_URL`.
- Generate offline evaluation reports for time-split model validation through [Evaluations/evaluate_model_time_split.py](Evaluations/evaluate_model_time_split.py).

## Stack

- Backend: FastAPI, SQLAlchemy
- Frontend: Jinja2 templates, HTML, CSS, JavaScript
- ML: scikit-learn `RandomForestRegressor` in a `Pipeline` with `ColumnTransformer`
- Data processing: pandas
- Optional local AI assistant: Ollama + local MCP server

## Repository Scope

This repository is intentionally code-only.

Excluded from version control:

- sample datasets
- documentation and deployment artifacts
- local databases and model files
- evaluation outputs
- environment secrets and local virtual environments

## Project Structure

```text
app/            FastAPI app, templates, static assets, services, models
scripts/        Training, seeding, import, and utility scripts
Evaluations/    Evaluation code for offline model validation
requirements.txt
run_local.sh
```

## Local Setup

1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and adjust values if needed.
4. Run the seed script if you want a local demo database.
5. Start the FastAPI app.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python scripts/seed_demo_data.py
uvicorn app.main:app --host 0.0.0.0 --port 8011
```

Open `http://127.0.0.1:8011`.

Default demo login:

```text
username: admin
password: admin
```

## Model Logic

The prediction workflow uses site context and event timing to estimate outage progression:

- categorical features: `site_name`, `region`
- numeric features: `hour_of_day`, `day_of_week`
- targets: `mains_to_dc_min`, `dc_to_down_min`, `site_down_duration_min`

The application then combines predicted durations with revenue rates to estimate potential loss over the next 1 to 4 hours.

## Evaluation Workflow

The repository includes an offline evaluation script that reconstructs complete three-stage incidents from private alarm workbooks and performs a strict time-based split:

- training window: 16 Apr 2026 to 26 Apr 2026
- validation window: 27 Apr 2026 to 30 Apr 2026

Generated artifacts include descriptive statistics, validation metrics, and split summaries. Private datasets themselves are not committed.

## Security Note

This project currently keeps demo passwords in plain text because that was a project requirement for the academic prototype. For any real deployment, replace this with proper password hashing and stronger session/secret management.
