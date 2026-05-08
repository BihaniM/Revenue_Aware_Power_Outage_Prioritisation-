from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
ALARM_SOURCE_PATH = ROOT / "sample_data" / "Power-Alarm-db.xlsx"
KPI_SOURCE_PATH = ROOT / "sample_data" / "KPI-db.xlsx"
OUTPUT_DIR = ROOT / "Evaluations"
DESCRIPTIVE_STATS_PATH = OUTPUT_DIR / "descriptive_stats.txt"
MODEL_EVAL_PATH = OUTPUT_DIR / "model_evaluation_results.txt"
FIVE_SITE_SAMPLE_PATH = OUTPUT_DIR / "five_site_sample.csv"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "training_dataset_summary.txt"
VALIDATION_SUMMARY_PATH = OUTPUT_DIR / "validation_dataset_summary.txt"

TRAIN_START = pd.Timestamp("2026-04-16").date()
TRAIN_END = pd.Timestamp("2026-04-26").date()
TEST_START = pd.Timestamp("2026-04-27").date()
TEST_END = pd.Timestamp("2026-04-30").date()

TARGET_COLUMNS = [
    "mains_to_dc_min",
    "dc_to_down_min",
    "site_down_duration_min",
]
FEATURE_COLUMNS = ["site_name", "region", "hour_of_day", "day_of_week"]

DAY_NAME_BY_INDEX = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


@dataclass
class PreparedData:
    source_records: pd.DataFrame
    filtered_records: pd.DataFrame
    incident_groups: pd.DataFrame
    complete_incidents: pd.DataFrame
    incomplete_count: int


def section_title(title: str) -> str:
    return f"\n{'=' * 88}\n{title}\n{'=' * 88}"


def subsection_title(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def format_frame(frame: pd.DataFrame, index: bool = False) -> str:
    if frame.empty:
        return "No data available."
    return frame.to_string(index=index)


def safe_datetime_parse(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    retry_mask = parsed.isna() & values.notna()
    if retry_mask.any():
        parsed.loc[retry_mask] = pd.to_datetime(values.loc[retry_mask], errors="coerce", dayfirst=True)
    return parsed


def coerce_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    aliases = {
        "incident_id": "incident_id",
        "site_name": "site_name",
        "region": "region",
        "event_date": "event_date",
        "alarm_sequence": "alarm_sequence",
        "alarm_name": "alarm_name",
        "occurred_on": "occurred_on",
        "cleared_on": "cleared_on",
        "alarm_duration_min": "alarm_duration_min",
    }
    return aliases.get(normalized, normalized)


def load_alarm_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Alarm dataset not found at {data_path}")

    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported alarm source format: {data_path.suffix}")

    df = df.rename(columns={column: normalize_column_name(column) for column in df.columns})
    required_columns = {
        "incident_id",
        "site_name",
        "region",
        "alarm_sequence",
        "occurred_on",
        "cleared_on",
        "alarm_duration_min",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Alarm workbook is missing required columns: {', '.join(missing_columns)}")

    df = df.copy()
    df["occurred_on"] = safe_datetime_parse(df["occurred_on"])
    df["cleared_on"] = safe_datetime_parse(df["cleared_on"])
    df["alarm_sequence"] = coerce_numeric(df["alarm_sequence"]).astype("Int64")
    df["alarm_duration_min"] = coerce_numeric(df["alarm_duration_min"])
    df["incident_id"] = df["incident_id"].astype(str).str.strip()
    df["site_name"] = df["site_name"].astype(str).str.strip()
    df["region"] = df["region"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

    return df


def reconstruct_incidents(df: pd.DataFrame) -> PreparedData:
    filtered = df[df["alarm_sequence"].isin([1, 2, 3])].copy()
    filtered = filtered.dropna(subset=["incident_id", "site_name", "occurred_on"]).copy()

    group_summary_rows: list[dict] = []
    complete_rows: list[dict] = []

    for (incident_id, site_name), group in filtered.groupby(["incident_id", "site_name"], dropna=False):
        sequences_present = set(group["alarm_sequence"].dropna().astype(int).tolist())
        ordered = (
            group.sort_values("occurred_on")
            .groupby("alarm_sequence", as_index=False)
            .first()
            .set_index("alarm_sequence")
        )

        region = str(group["region"].dropna().iloc[0]).strip() if group["region"].notna().any() else "Unknown"
        first_seen = group["occurred_on"].min()
        group_summary_rows.append(
            {
                "incident_id": incident_id,
                "site_name": site_name,
                "region": region or "Unknown",
                "sequence_count": len(sequences_present),
                "is_complete": {1, 2, 3}.issubset(sequences_present),
                "first_seen": first_seen,
            }
        )

        if not {1, 2, 3}.issubset(sequences_present):
            continue

        stage_1 = ordered.loc[1]
        stage_2 = ordered.loc[2]
        stage_3 = ordered.loc[3]

        mains_to_dc = (stage_2["occurred_on"] - stage_1["occurred_on"]).total_seconds() / 60.0
        dc_to_down = (stage_3["occurred_on"] - stage_2["occurred_on"]).total_seconds() / 60.0

        site_down_duration = stage_3.get("alarm_duration_min")
        if pd.isna(site_down_duration) or site_down_duration <= 0:
            cleared_on = stage_3.get("cleared_on")
            occurred_on = stage_3.get("occurred_on")
            if pd.notna(cleared_on) and pd.notna(occurred_on):
                site_down_duration = (cleared_on - occurred_on).total_seconds() / 60.0

        incident_date = stage_1["occurred_on"].date()
        complete_rows.append(
            {
                "incident_id": incident_id,
                "site_name": site_name,
                "region": region or "Unknown",
                "actual_mains_to_dc_min": mains_to_dc,
                "actual_dc_to_down_min": dc_to_down,
                "actual_site_down_duration_min": site_down_duration,
                "mains_to_dc_min": mains_to_dc,
                "dc_to_down_min": dc_to_down,
                "site_down_duration_min": site_down_duration,
                "hour_of_day": int(stage_1["occurred_on"].hour),
                "day_of_week": int(stage_1["occurred_on"].weekday()),
                "incident_date": incident_date,
                "sequence_1_occurred_on": stage_1["occurred_on"],
                "sequence_2_occurred_on": stage_2["occurred_on"],
                "sequence_3_occurred_on": stage_3["occurred_on"],
            }
        )

    incident_groups = pd.DataFrame(group_summary_rows)
    complete = pd.DataFrame(complete_rows)
    if not complete.empty:
        complete = complete.replace([np.inf, -np.inf], np.nan)
        complete = complete.dropna(
            subset=[
                "actual_mains_to_dc_min",
                "actual_dc_to_down_min",
                "actual_site_down_duration_min",
            ]
        )
        complete = complete[
            (complete["actual_mains_to_dc_min"] > 0)
            & (complete["actual_dc_to_down_min"] > 0)
            & (complete["actual_site_down_duration_min"] > 0)
        ].copy()

    incomplete_count = int((~incident_groups["is_complete"]).sum()) if not incident_groups.empty else 0
    return PreparedData(
        source_records=df.copy(),
        filtered_records=filtered,
        incident_groups=incident_groups,
        complete_incidents=complete,
        incomplete_count=incomplete_count,
    )


def descriptive_summary(prepared: PreparedData) -> str:
    source = prepared.source_records
    raw = prepared.filtered_records
    complete = prepared.complete_incidents
    incident_groups = prepared.incident_groups

    parts: list[str] = [section_title("PART 2 - DESCRIPTIVE STATISTICS")]

    total_grouped_incidents = len(incident_groups)
    incomplete_pct = (prepared.incomplete_count / total_grouped_incidents * 100.0) if total_grouped_incidents else 0.0
    date_min = raw["occurred_on"].min()
    date_max = raw["occurred_on"].max()
    summary_lines = [
        f"Total alarm records in file: {len(source):,}",
        f"Total unique sites: {source['site_name'].nunique():,}",
        f"Total unique incidents: {source['incident_id'].nunique():,}",
        f"Total complete three-stage incidents: {len(complete):,}",
        f"Incomplete incidents: {prepared.incomplete_count:,} ({incomplete_pct:.2f}%)",
        (
            "Date range of dataset: "
            f"{date_min.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(date_min) else 'N/A'} "
            f"to {date_max.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(date_max) else 'N/A'}"
        ),
    ]
    parts.append(subsection_title("2a - Overall dataset summary"))
    parts.append("\n".join(summary_lines))

    stage_stats = pd.DataFrame(
        {
            "target": ["mains_to_dc_min", "dc_to_down_min", "site_down_duration_min"],
            "mean": [complete["mains_to_dc_min"].mean(), complete["dc_to_down_min"].mean(), complete["site_down_duration_min"].mean()],
            "median": [complete["mains_to_dc_min"].median(), complete["dc_to_down_min"].median(), complete["site_down_duration_min"].median()],
            "std_dev": [complete["mains_to_dc_min"].std(), complete["dc_to_down_min"].std(), complete["site_down_duration_min"].std()],
            "min": [complete["mains_to_dc_min"].min(), complete["dc_to_down_min"].min(), complete["site_down_duration_min"].min()],
            "max": [complete["mains_to_dc_min"].max(), complete["dc_to_down_min"].max(), complete["site_down_duration_min"].max()],
            "p25": [complete["mains_to_dc_min"].quantile(0.25), complete["dc_to_down_min"].quantile(0.25), complete["site_down_duration_min"].quantile(0.25)],
            "p75": [complete["mains_to_dc_min"].quantile(0.75), complete["dc_to_down_min"].quantile(0.75), complete["site_down_duration_min"].quantile(0.75)],
        }
    )
    stage_stats = stage_stats.round(2)
    parts.append(subsection_title("2b - Stage duration statistics across all complete incidents"))
    parts.append(format_frame(stage_stats))

    site_counts = (
        complete.groupby("site_name", dropna=False)
        .size()
        .reset_index(name="complete_incident_count")
        .sort_values(["complete_incident_count", "site_name"], ascending=[False, True])
    )
    sparse_sites = site_counts[site_counts["complete_incident_count"] < 3].copy()
    max_count = int(site_counts["complete_incident_count"].max()) if not site_counts.empty else 0
    min_count = int(site_counts["complete_incident_count"].min()) if not site_counts.empty else 0
    most_sites = site_counts[site_counts["complete_incident_count"] == max_count]["site_name"].tolist() if not site_counts.empty else []
    fewest_sites = site_counts[site_counts["complete_incident_count"] == min_count]["site_name"].tolist() if not site_counts.empty else []
    parts.append(subsection_title("2c - Per-site incident count"))
    parts.append(format_frame(site_counts))
    parts.append(
        "\n".join(
            [
                f"Sites with fewer than 3 complete incidents: {len(sparse_sites):,}",
                f"Sparse sites: {', '.join(sparse_sites['site_name'].tolist()) if not sparse_sites.empty else 'None'}",
                f"Sites with most complete incidents ({max_count}): {', '.join(most_sites) if most_sites else 'N/A'}",
                f"Sites with fewest complete incidents ({min_count}): {', '.join(fewest_sites) if fewest_sites else 'N/A'}",
            ]
        )
    )

    region_breakdown = (
        complete.groupby("region", dropna=False)
        .agg(
            complete_incidents=("incident_id", "count"),
            mean_mains_to_dc_min=("mains_to_dc_min", "mean"),
            mean_dc_to_down_min=("dc_to_down_min", "mean"),
            mean_site_down_duration_min=("site_down_duration_min", "mean"),
        )
        .reset_index()
        .sort_values("complete_incidents", ascending=False)
        .round(2)
    )
    parts.append(subsection_title("2d - Regional breakdown"))
    parts.append(format_frame(region_breakdown))

    hourly_distribution = (
        complete.groupby("hour_of_day")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index(name="incident_count")
    )
    peak_hours = hourly_distribution[hourly_distribution["incident_count"] == hourly_distribution["incident_count"].max()]["hour_of_day"].tolist() if not hourly_distribution.empty else []
    off_peak_hours = hourly_distribution[hourly_distribution["incident_count"] == hourly_distribution["incident_count"].min()]["hour_of_day"].tolist() if not hourly_distribution.empty else []
    parts.append(subsection_title("2e - Time of day distribution"))
    parts.append(format_frame(hourly_distribution))
    parts.append(
        "\n".join(
            [
                f"Peak hours for power failures: {', '.join(map(str, peak_hours)) if peak_hours else 'N/A'}",
                f"Off-peak hours for power failures: {', '.join(map(str, off_peak_hours)) if off_peak_hours else 'N/A'}",
            ]
        )
    )

    weekday_distribution = (
        complete.groupby("day_of_week")
        .size()
        .reindex(range(7), fill_value=0)
        .reset_index(name="incident_count")
    )
    weekday_distribution["day_name"] = weekday_distribution["day_of_week"].map(DAY_NAME_BY_INDEX)
    max_weekday_count = int(weekday_distribution["incident_count"].max()) if not weekday_distribution.empty else 0
    min_weekday_count = int(weekday_distribution["incident_count"].min()) if not weekday_distribution.empty else 0
    busiest_days = weekday_distribution[weekday_distribution["incident_count"] == max_weekday_count]["day_name"].tolist()
    quietest_days = weekday_distribution[weekday_distribution["incident_count"] == min_weekday_count]["day_name"].tolist()
    parts.append(subsection_title("2f - Day of week distribution"))
    parts.append(format_frame(weekday_distribution[["day_of_week", "day_name", "incident_count"]]))
    parts.append(
        "\n".join(
            [
                f"Days with most incidents ({max_weekday_count}): {', '.join(busiest_days) if busiest_days else 'N/A'}",
                f"Days with fewest incidents ({min_weekday_count}): {', '.join(quietest_days) if quietest_days else 'N/A'}",
            ]
        )
    )

    site_variability = (
        complete.groupby("site_name", dropna=False)
        .agg(
            incident_count=("incident_id", "count"),
            site_down_duration_std=("site_down_duration_min", lambda values: float(pd.Series(values).std(ddof=0))),
        )
        .reset_index()
        .sort_values(["site_down_duration_std", "incident_count", "site_name"], ascending=[False, False, True])
        .round(2)
    )
    most_variable = site_variability.head(5)
    most_consistent = site_variability.sort_values(["site_down_duration_std", "incident_count", "site_name"], ascending=[True, False, True]).head(5)
    parts.append(subsection_title("2g - Site variability"))
    parts.append("Top 5 most variable sites")
    parts.append(format_frame(most_variable))
    parts.append("\nTop 5 most consistent sites")
    parts.append(format_frame(most_consistent))

    return "\n".join(parts)


def split_time_windows(complete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = complete["incident_date"].between(TRAIN_START, TRAIN_END)
    test_mask = complete["incident_date"].between(TEST_START, TEST_END)
    return complete[train_mask].copy(), complete[test_mask].copy()


def dataset_split_summary(frame: pd.DataFrame, split_name: str, start_date, end_date) -> str:
    parts = [section_title(f"{split_name.upper()} DATASET SUMMARY")]

    if frame.empty:
        parts.append(f"No incidents found for {split_name.lower()} period {start_date} to {end_date}.")
        return "\n".join(parts)

    date_min = frame["incident_date"].min()
    date_max = frame["incident_date"].max()
    parts.append(subsection_title("Overview"))
    parts.append(
        "\n".join(
            [
                f"Configured period: {start_date} to {end_date}",
                f"Observed incident date range: {date_min} to {date_max}",
                f"Complete incidents: {len(frame):,}",
                f"Unique incidents: {frame['incident_id'].nunique():,}",
                f"Unique sites: {frame['site_name'].nunique():,}",
                f"Unique regions: {frame['region'].nunique():,}",
            ]
        )
    )

    duration_stats = pd.DataFrame(
        {
            "target": TARGET_COLUMNS,
            "mean": [frame[target].mean() for target in TARGET_COLUMNS],
            "median": [frame[target].median() for target in TARGET_COLUMNS],
            "std_dev": [frame[target].std() for target in TARGET_COLUMNS],
            "min": [frame[target].min() for target in TARGET_COLUMNS],
            "max": [frame[target].max() for target in TARGET_COLUMNS],
        }
    ).round(2)
    parts.append(subsection_title("Stage duration statistics"))
    parts.append(format_frame(duration_stats))

    region_counts = (
        frame.groupby("region", dropna=False)
        .agg(incident_count=("incident_id", "count"))
        .reset_index()
        .sort_values(["incident_count", "region"], ascending=[False, True])
    )
    parts.append(subsection_title("Regional distribution"))
    parts.append(format_frame(region_counts))

    site_counts = (
        frame.groupby("site_name", dropna=False)
        .size()
        .reset_index(name="incident_count")
        .sort_values(["incident_count", "site_name"], ascending=[False, True])
    )
    parts.append(subsection_title("Top 10 sites by incident count"))
    parts.append(format_frame(site_counts.head(10)))

    hour_counts = (
        frame.groupby("hour_of_day")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index(name="incident_count")
    )
    parts.append(subsection_title("Incidents by hour of day"))
    parts.append(format_frame(hour_counts))

    weekday_counts = (
        frame.groupby("day_of_week")
        .size()
        .reindex(range(7), fill_value=0)
        .reset_index(name="incident_count")
    )
    weekday_counts["day_name"] = weekday_counts["day_of_week"].map(DAY_NAME_BY_INDEX)
    parts.append(subsection_title("Incidents by day of week"))
    parts.append(format_frame(weekday_counts[["day_of_week", "day_name", "incident_count"]]))

    return "\n".join(parts)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["site_name", "region"]),
                        ("num", "passthrough", ["hour_of_day", "day_of_week"]),
                    ]
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def metric_row(actual: pd.Series, predicted: np.ndarray, baseline: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": rmse,
        "R2": r2,
        "Baseline_MAE": mean_absolute_error(actual, baseline),
    }


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, str]:
    if train_df.empty:
        raise ValueError("Training split is empty for 2026-04-16 to 2026-04-26.")
    if test_df.empty:
        raise ValueError("Test split is empty for 2026-04-27 to 2026-04-30.")

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMNS].copy()

    warnings_list: list[str] = []
    if len(train_df) < 10:
        warnings_list.append(
            f"Warning: only {len(train_df)} training incidents are available. Metrics may be unstable."
        )

    preview_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site_name", "region"]),
            ("num", "passthrough", ["hour_of_day", "day_of_week"]),
        ]
    )
    encoded_train = preview_preprocessor.fit_transform(X_train)
    feature_count = int(encoded_train.shape[1])

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = pd.DataFrame(model.predict(X_test), columns=[f"predicted_{name}" for name in TARGET_COLUMNS], index=test_df.index)
    results = test_df.copy()
    results = pd.concat([results, predictions], axis=1)

    training_target_means = y_train.mean()
    metrics_rows: list[dict] = []
    for target in TARGET_COLUMNS:
        predicted_col = f"predicted_{target}"
        baseline_predictions = np.repeat(training_target_means[target], len(results))
        metrics = metric_row(results[target], results[predicted_col], baseline_predictions)
        metrics_rows.append(
            {
                "target": target,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "baseline_MAE": metrics["Baseline_MAE"],
            }
        )
        results[f"abs_error_{target}"] = (results[target] - results[predicted_col]).abs()

    metrics_df = pd.DataFrame(metrics_rows).round(2)
    results.attrs["metrics_df"] = metrics_df

    training_summary_lines = [
        section_title("PART 3 - MODEL TRAINING"),
        f"Training period: {TRAIN_START} to {TRAIN_END}",
        f"Training set size: {len(train_df):,} complete incidents",
        f"Encoded feature count: {feature_count:,}",
    ]
    training_summary_lines.extend(warnings_list)
    training_summary = "\n".join(training_summary_lines)

    return model, results, training_summary + "\n"


def per_site_test_rmse(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby("site_name", dropna=False)
        .agg(
            test_incident_count=("site_name", "size"),
            site_down_duration_rmse=(
                "site_down_duration_min",
                lambda actual: float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            results.loc[actual.index, "predicted_site_down_duration_min"],
                        )
                    )
                ),
            ),
        )
        .reset_index()
        .sort_values(["site_down_duration_rmse", "test_incident_count", "site_name"], ascending=[False, False, True])
        .round(2)
    )
    return grouped


def tolerance_rows(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    work = frame.copy()
    work["site_down_abs_error"] = (work["site_down_duration_min"] - work["predicted_site_down_duration_min"]).abs()

    def summarise(group: pd.DataFrame) -> dict:
        total = len(group)
        within_15 = int((group["site_down_abs_error"] <= 15).sum())
        within_30 = int((group["site_down_abs_error"] <= 30).sum())
        within_60 = int((group["site_down_abs_error"] <= 60).sum())
        return {
            "incident_count": total,
            "within_15_count": within_15,
            "within_15_pct": (within_15 / total * 100.0) if total else 0.0,
            "within_30_count": within_30,
            "within_30_pct": (within_30 / total * 100.0) if total else 0.0,
            "within_60_count": within_60,
            "within_60_pct": (within_60 / total * 100.0) if total else 0.0,
        }

    if group_column is None:
        return pd.DataFrame([summarise(work)]).round(2)

    summary_rows: list[dict] = []
    for group_value, group in work.groupby(group_column, dropna=False):
        row = {group_column: group_value}
        row.update(summarise(group))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("within_60_pct", ascending=False).round(2)


def evaluation_report(results: pd.DataFrame) -> str:
    primary_table = results.attrs.get("metrics_df")
    if primary_table is None:
        raise ValueError("Evaluation metrics were not attached to the results frame.")
    site_rmse = per_site_test_rmse(results)
    best_sites = site_rmse.tail(1)
    worst_sites = site_rmse.head(1)
    tolerance_overall = tolerance_rows(results)
    tolerance_by_region = tolerance_rows(results, group_column="region")

    parts = [section_title("PART 4 - MODEL EVALUATION")]
    parts.append(subsection_title("5 - Primary results on 27-30 April test incidents"))
    parts.append(format_frame(primary_table))
    parts.append(subsection_title("6 - Per-site evaluation on test set"))
    parts.append(format_frame(site_rmse))
    parts.append("\nWorst predicted site(s)")
    parts.append(format_frame(worst_sites))
    parts.append("\nBest predicted site(s)")
    parts.append(format_frame(best_sites))
    parts.append(subsection_title("7 - Tolerance analysis for site_down_duration_min"))
    parts.append("Overall tolerance")
    parts.append(format_frame(tolerance_overall))
    parts.append("\nPer-region tolerance")
    parts.append(format_frame(tolerance_by_region))
    return "\n".join(parts)


def choose_five_sites(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    eligible_sites = sorted(set(train_df["site_name"]) & set(test_df["site_name"]))
    rng = random.Random(42)
    if len(eligible_sites) <= 5:
        return eligible_sites
    return sorted(rng.sample(eligible_sites, 5))


def five_site_sample(results: pd.DataFrame, selected_sites: list[str]) -> tuple[pd.DataFrame, str]:
    parts = [section_title("PART 5 - 5-SITE ILLUSTRATION SAMPLE")]
    if not selected_sites:
        parts.append("No sites met the requirement of appearing in both the training and test periods.")
        return pd.DataFrame(), "\n".join(parts)

    sample = results[results["site_name"].isin(selected_sites)].copy()
    sample["within_15_min"] = sample["abs_error_site_down_duration_min"] <= 15
    sample["within_30_min"] = sample["abs_error_site_down_duration_min"] <= 30
    sample["within_60_min"] = sample["abs_error_site_down_duration_min"] <= 60

    export_columns = [
        "site_name",
        "incident_date",
        "hour_of_day",
        "mains_to_dc_min",
        "dc_to_down_min",
        "site_down_duration_min",
        "predicted_mains_to_dc_min",
        "predicted_dc_to_down_min",
        "predicted_site_down_duration_min",
        "abs_error_mains_to_dc_min",
        "abs_error_dc_to_down_min",
        "abs_error_site_down_duration_min",
        "within_15_min",
        "within_30_min",
        "within_60_min",
    ]
    sample = sample.sort_values(["site_name", "incident_date", "hour_of_day"]).copy()
    sample_export = sample[export_columns].rename(
        columns={
            "mains_to_dc_min": "actual_mains_to_dc_min",
            "dc_to_down_min": "actual_dc_to_down_min",
            "site_down_duration_min": "actual_site_down_duration_min",
        }
    )

    per_site = tolerance_rows(sample, group_column="site_name")
    total = len(sample)
    within_15 = int(sample["within_15_min"].sum())
    within_30 = int(sample["within_30_min"].sum())
    within_60 = int(sample["within_60_min"].sum())

    parts.append(f"Selected sites (seed 42): {', '.join(selected_sites)}")
    parts.append("\nIncident-level detail")
    parts.append(format_frame(sample_export.round(2)))
    parts.append(subsection_title("10 - 5-site tolerance summary"))
    parts.append(
        "\n".join(
            [
                f"Within 15 min: {within_15} of {total} incidents ({(within_15 / total * 100.0) if total else 0.0:.2f}%)",
                f"Within 30 min: {within_30} of {total} incidents ({(within_30 / total * 100.0) if total else 0.0:.2f}%)",
                f"Within 60 min: {within_60} of {total} incidents ({(within_60 / total * 100.0) if total else 0.0:.2f}%)",
            ]
        )
    )
    parts.append("\nPer-site breakdown")
    parts.append(format_frame(per_site))

    return sample_export, "\n".join(parts)


def revenue_context_report(complete: pd.DataFrame) -> str:
    time_to_down = complete["mains_to_dc_min"] + complete["dc_to_down_min"]
    within_60 = float((time_to_down <= 60).mean() * 100.0) if len(time_to_down) else 0.0
    within_90 = float((time_to_down <= 90).mean() * 100.0) if len(time_to_down) else 0.0
    within_120 = float((time_to_down <= 120).mean() * 100.0) if len(time_to_down) else 0.0

    lines = [
        section_title("PART 6 - REVENUE CONTEXT STATISTICS"),
        f"Average total time from first alarm to site down: {time_to_down.mean():.2f} minutes",
        f"Minimum total time to site down: {time_to_down.min():.2f} minutes",
        f"Maximum total time to site down: {time_to_down.max():.2f} minutes",
        f"Incidents reaching site down within 60 minutes: {within_60:.2f}%",
        f"Incidents reaching site down within 90 minutes: {within_90:.2f}%",
        f"Incidents reaching site down within 120 minutes: {within_120:.2f}%",
    ]
    return "\n".join(lines)


def write_outputs(
    descriptive_text: str,
    evaluation_text: str,
    five_site_df: pd.DataFrame,
    training_summary_text: str,
    validation_summary_text: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DESCRIPTIVE_STATS_PATH.write_text(descriptive_text + "\n", encoding="utf-8")
    MODEL_EVAL_PATH.write_text(evaluation_text + "\n", encoding="utf-8")
    TRAINING_SUMMARY_PATH.write_text(training_summary_text + "\n", encoding="utf-8")
    VALIDATION_SUMMARY_PATH.write_text(validation_summary_text + "\n", encoding="utf-8")
    if five_site_df.empty:
        pd.DataFrame().to_csv(FIVE_SITE_SAMPLE_PATH, index=False)
    else:
        five_site_df.to_csv(FIVE_SITE_SAMPLE_PATH, index=False)


def main() -> None:
    data_path = ALARM_SOURCE_PATH
    raw_df = load_alarm_data(data_path)
    prepared = reconstruct_incidents(raw_df)
    if prepared.complete_incidents.empty:
        raise ValueError("No complete three-stage incidents remained after reconstruction and filtering.")

    descriptive_text = descriptive_summary(prepared)
    train_df, test_df = split_time_windows(prepared.complete_incidents)
    training_dataset_text = dataset_split_summary(train_df, "Training", TRAIN_START, TRAIN_END)
    validation_dataset_text = dataset_split_summary(test_df, "Validation", TEST_START, TEST_END)
    _model, results, training_text = evaluate_model(train_df, test_df)

    evaluation_text = evaluation_report(results)
    selected_sites = choose_five_sites(train_df, test_df)
    five_site_df, five_site_text = five_site_sample(results, selected_sites)
    revenue_text = revenue_context_report(prepared.complete_incidents)

    print(section_title("PART 1 - DATA PREPARATION"))
    print(f"Loaded alarm source: {data_path}")
    print(f"KPI workbook detected: {KPI_SOURCE_PATH.exists()} ({KPI_SOURCE_PATH})")
    print(f"Alarm records after filtering to sequences 1, 2, 3: {len(prepared.filtered_records):,}")
    print(f"Complete incidents retained: {len(prepared.complete_incidents):,}")
    print(f"Training incidents: {len(train_df):,}")
    print(f"Test incidents: {len(test_df):,}")
    print(descriptive_text)
    print(training_text)
    print(evaluation_text)
    print(five_site_text)
    print(revenue_text)

    write_outputs(
        descriptive_text,
        evaluation_text,
        five_site_df,
        training_dataset_text,
        validation_dataset_text,
    )
    print(section_title("PART 7 - OUTPUT FILES"))
    print(f"Descriptive statistics written to: {DESCRIPTIVE_STATS_PATH}")
    print(f"Model evaluation results written to: {MODEL_EVAL_PATH}")
    print(f"Five-site sample written to: {FIVE_SITE_SAMPLE_PATH}")
    print(f"Training dataset summary written to: {TRAINING_SUMMARY_PATH}")
    print(f"Validation dataset summary written to: {VALIDATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()