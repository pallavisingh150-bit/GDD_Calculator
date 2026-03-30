"""
Growing Degree Day (GDD) Calculator
====================================
A Streamlit web application for calculating Growing Degree Days (GDD)
from daily weather data. Supports multiple experiments, stage-wise
analysis, visualization, and export.

Author: GDD Tool
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import io
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GDD Calculator",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }

    .main-header {
        background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.4rem;
        margin: 0 0 0.4rem 0;
        color: #d8f3dc;
    }

    .main-header p {
        margin: 0;
        color: #b7e4c7;
        font-size: 1rem;
        font-weight: 300;
    }

    .metric-card {
        background: #f8fdf9;
        border: 1px solid #d8f3dc;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-card .label {
        color: #52b788;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .metric-card .value {
        color: #1b4332;
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
    }

    .metric-card .unit {
        color: #74c69d;
        font-size: 0.85rem;
        margin-left: 0.2rem;
    }

    .section-header {
        font-family: 'DM Serif Display', serif !important;
        color: #1b4332;
        border-bottom: 2px solid #d8f3dc;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .stage-badge {
        display: inline-block;
        background: #d8f3dc;
        color: #1b4332;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .info-box {
        background: #f0faf4;
        border-left: 4px solid #52b788;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #2d6a4f;
    }

    .warning-box {
        background: #fff8e6;
        border-left: 4px solid #f4a261;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #7c4a03;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0faf4;
        padding: 6px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        color: #2d6a4f;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1b4332 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        padding: 0.5rem 1.4rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        box-shadow: 0 4px 12px rgba(45,106,79,0.3);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #d8f3dc;
    }

    .footer-note {
        text-align: center;
        color: #74c69d;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #d8f3dc;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GDD CORE FUNCTIONS
# ─────────────────────────────────────────────

def cap_temperature(temp: float, t_base: float, t_upper: float = 30.0) -> float:
    """
    Apply upper and lower capping to a temperature value.

    Rules:
      - If temp >= t_upper  → use t_upper
      - If temp <= t_base   → use t_base
      - Otherwise           → use temp as-is

    Parameters
    ----------
    temp    : observed temperature (°C)
    t_base  : base (threshold) temperature (°C)
    t_upper : upper (ceiling) temperature (°C), default 30
    """
    if temp >= t_upper:
        return t_upper
    elif temp <= t_base:
        return t_base
    return temp


def calculate_daily_gdd(tmax: float, tmin: float, t_base: float, t_upper: float = 30.0) -> float:
    """
    Calculate GDD for a single day.

    Formula:
        GDD = ((Tmax_eff + Tmin_eff) / 2) - Tbase
        GDD is floored at 0 (no negative accumulation).

    Parameters
    ----------
    tmax    : daily maximum temperature
    tmin    : daily minimum temperature
    t_base  : base temperature
    t_upper : upper ceiling temperature
    """
    tmax_eff = cap_temperature(tmax, t_base, t_upper)
    tmin_eff = cap_temperature(tmin, t_base, t_upper)
    gdd = ((tmax_eff + tmin_eff) / 2.0) - t_base
    return max(gdd, 0.0)   # GDD cannot be negative


def process_experiment(
    weather_df: pd.DataFrame,
    sowing_date: date,
    harvest_date: date,
    t_base: float,
    t_upper: float = 30.0,
    stages: list = None,
) -> dict:
    """
    Run GDD analysis for one sowing–harvest experiment.

    Parameters
    ----------
    weather_df   : DataFrame with columns [date, tmax, tmin]
    sowing_date  : crop sowing date
    harvest_date : crop harvest date
    t_base       : base temperature
    t_upper      : upper threshold temperature
    stages       : list of dicts {name, date} for crop growth stages

    Returns
    -------
    dict with keys: daily_df, total_gdd, stage_gdd, missing_days
    """
    # Filter to the crop period
    mask = (
        (weather_df["date"] >= pd.Timestamp(sowing_date)) &
        (weather_df["date"] <= pd.Timestamp(harvest_date))
    )
    crop_df = weather_df.loc[mask].copy()

    if crop_df.empty:
        return None

    # Track missing rows
    expected_dates = pd.date_range(sowing_date, harvest_date, freq="D")
    present_dates  = set(crop_df["date"].dt.date)
    missing_days   = [d for d in expected_dates.date if d not in present_dates]

    # Drop rows where tmax or tmin is NaN
    n_before = len(crop_df)
    crop_df.dropna(subset=["tmax", "tmin"], inplace=True)
    n_after  = len(crop_df)
    rows_dropped = n_before - n_after

    # Daily GDD
    crop_df["daily_gdd"] = crop_df.apply(
        lambda row: calculate_daily_gdd(row["tmax"], row["tmin"], t_base, t_upper),
        axis=1,
    )

    # Cumulative GDD (reset from sowing)
    crop_df["cumulative_gdd"] = crop_df["daily_gdd"].cumsum()

    # Effective temps for transparency
    crop_df["tmax_eff"] = crop_df["tmax"].apply(lambda x: cap_temperature(x, t_base, t_upper))
    crop_df["tmin_eff"] = crop_df["tmin"].apply(lambda x: cap_temperature(x, t_base, t_upper))

    total_gdd = crop_df["cumulative_gdd"].iloc[-1] if not crop_df.empty else 0.0

    # Stage-wise GDD
    stage_gdd = {}
    if stages:
        for stage in stages:
            s_date = pd.Timestamp(stage["date"])
            if s_date < pd.Timestamp(sowing_date) or s_date > pd.Timestamp(harvest_date):
                stage_gdd[stage["name"]] = None   # outside crop period
            else:
                sub = crop_df[crop_df["date"] <= s_date]
                stage_gdd[stage["name"]] = round(sub["cumulative_gdd"].iloc[-1], 2) if not sub.empty else 0.0

    return {
        "daily_df":    crop_df.reset_index(drop=True),
        "total_gdd":   round(total_gdd, 2),
        "stage_gdd":   stage_gdd,
        "missing_days": missing_days,
        "rows_dropped": rows_dropped,
    }


def load_weather_data(uploaded_file) -> pd.DataFrame | None:
    """
    Load and standardise weather data from CSV or Excel upload.

    Expected columns (case-insensitive):
        date, tmax / t_max / maximum temperature
        tmin / t_min / minimum temperature

    Returns cleaned DataFrame with columns: date, tmax, tmin
    """
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None

        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Map flexible column names → standard names
        col_map = {}
        for col in df.columns:
            if "date" in col:
                col_map[col] = "date"
            elif col in ("tmax", "t_max", "max_temp", "maximum_temperature",
                         "temp_max", "tempmax"):
                col_map[col] = "tmax"
            elif col in ("tmin", "t_min", "min_temp", "minimum_temperature",
                         "temp_min", "tempmin"):
                col_map[col] = "tmin"

        df.rename(columns=col_map, inplace=True)

        required = {"date", "tmax", "tmin"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            st.error(f"Missing required columns: {missing}. Found: {list(df.columns)}")
            return None

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
        df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")

        n_invalid_dates = df["date"].isna().sum()
        if n_invalid_dates:
            st.warning(f"{n_invalid_dates} rows had unparseable dates and were removed.")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as exc:
        st.error(f"Error loading file: {exc}")
        return None


def build_export_excel(results: dict) -> bytes:
    """
    Build a multi-sheet Excel export from all experiment results.

    Returns raw bytes of the .xlsx file.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []

        for exp_name, res in results.items():
            if res is None:
                continue

            df = res["daily_df"].copy()
            # Pretty column names for export
            export_df = df[["date", "tmax", "tmin", "tmax_eff", "tmin_eff",
                             "daily_gdd", "cumulative_gdd"]].copy()
            export_df.columns = [
                "Date", "Tmax (°C)", "Tmin (°C)",
                "Tmax Effective", "Tmin Effective",
                "Daily GDD", "Cumulative GDD"
            ]
            sheet_name = exp_name[:31]   # Excel sheet name limit
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)

            row = {"Experiment": exp_name, "Total GDD": res["total_gdd"]}
            for stage, val in res.get("stage_gdd", {}).items():
                row[f"GDD @ {stage}"] = val
            summary_rows.append(row)

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name="Summary", index=False
            )

    return buffer.getvalue()


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

PALETTE = [
    "#2d6a4f", "#74c69d", "#40916c",
    "#1b4332", "#95d5b2", "#52b788",
]


def plot_cumulative_gdd(results: dict, stages_list: list = None) -> go.Figure:
    """Line chart of cumulative GDD vs date for all experiments."""
    fig = go.Figure()

    for i, (exp_name, res) in enumerate(results.items()):
        if res is None:
            continue
        df = res["daily_df"]
        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["cumulative_gdd"],
            mode="lines",
            name=exp_name,
            line=dict(color=color, width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Cumulative GDD: %{y:.1f} °C·d<extra></extra>",
        ))

    # Add stage vertical lines (from first experiment if available)
    first_res = next((v for v in results.values() if v), None)
    if stages_list and first_res:
        for stage in stages_list:
            fig.add_vline(
                x=pd.Timestamp(stage["date"]),
                line_dash="dot",
                line_color="#f4a261",
                line_width=1.5,
                annotation_text=stage["name"],
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="#7c4a03",
            )

    fig.update_layout(
        title=dict(text="Cumulative GDD Over Crop Period", font=dict(size=16, color="#1b4332")),
        xaxis_title="Date",
        yaxis_title="Cumulative GDD (°C·days)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#f8fdf9",
        paper_bgcolor="white",
        font=dict(family="DM Sans"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#d8f3dc"),
        yaxis=dict(showgrid=True, gridcolor="#d8f3dc"),
        margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


def plot_daily_gdd(results: dict) -> go.Figure:
    """Bar chart of daily GDD for each experiment."""
    fig = go.Figure()

    for i, (exp_name, res) in enumerate(results.items()):
        if res is None:
            continue
        df = res["daily_df"]
        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(go.Bar(
            x=df["date"], y=df["daily_gdd"],
            name=exp_name,
            marker_color=color,
            opacity=0.85,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Daily GDD: %{y:.2f} °C·d<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Daily GDD", font=dict(size=16, color="#1b4332")),
        xaxis_title="Date",
        yaxis_title="Daily GDD (°C·days)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#f8fdf9",
        paper_bgcolor="white",
        font=dict(family="DM Sans"),
        xaxis=dict(showgrid=False, gridcolor="#d8f3dc"),
        yaxis=dict(showgrid=True, gridcolor="#d8f3dc"),
        margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


def plot_temperature_gdd(df: pd.DataFrame, exp_name: str, t_base: float) -> go.Figure:
    """Combined chart: Tmax/Tmin bars + GDD line for one experiment."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df["date"], y=df["tmax"],
        name="Tmax", marker_color="#f4a261", opacity=0.6,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["tmin"],
        name="Tmin", marker_color="#74b9d3", opacity=0.6,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_gdd"],
        name="Cumulative GDD",
        line=dict(color="#2d6a4f", width=2.5),
        mode="lines",
    ), secondary_y=True)

    fig.add_hline(y=t_base, line_dash="dash", line_color="#e63946",
                  annotation_text=f"Base Temp ({t_base}°C)",
                  annotation_font_color="#e63946",
                  secondary_y=False)

    fig.update_layout(
        title=dict(text=f"Temperature & GDD — {exp_name}", font=dict(size=15, color="#1b4332")),
        barmode="group",
        plot_bgcolor="#f8fdf9",
        paper_bgcolor="white",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=80, b=40, l=60, r=60),
    )
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False,
                     gridcolor="#d8f3dc")
    fig.update_yaxes(title_text="Cumulative GDD (°C·days)", secondary_y=True)
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def metric_card(label: str, value, unit: str = "°C·days"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}<span class="unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Weather data upload ──────────────────
    st.markdown("### 📂 Weather Data")
    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Must contain columns: Date, Tmax, Tmin",
    )

    weather_df = None
    if uploaded_file:
        weather_df = load_weather_data(uploaded_file)
        if weather_df is not None:
            st.success(f"✅ Loaded {len(weather_df):,} rows  "
                       f"({weather_df['date'].min().date()} → "
                       f"{weather_df['date'].max().date()})")

    st.markdown("---")

    # ── Base temperature ─────────────────────
    st.markdown("### 🌡️ Temperature Settings")
    t_base = st.number_input(
        "Base Temperature (°C)",
        min_value=-10.0, max_value=25.0, value=10.0, step=0.5,
        help="Minimum temperature below which crop growth stops.",
    )
    t_upper = st.number_input(
        "Upper Threshold (°C)",
        min_value=25.0, max_value=45.0, value=30.0, step=0.5,
        help="Temperature above which no extra growth occurs.",
    )

    st.markdown("---")

    # ── Number of experiments ────────────────
    st.markdown("### 🌱 Experiments")
    n_experiments = st.slider(
        "Number of sowing dates",
        min_value=1, max_value=5, value=1,
    )

    st.markdown("---")

    # ── Crop growth stages ───────────────────
    st.markdown("### 📅 Crop Growth Stages")
    n_stages = st.slider("Number of stages", 0, 8, 0)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🌾 GDD Calculator</h1>
  <p>Growing Degree Day analysis · Multiple experiments · Stage-wise comparison · Export ready</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────
tab_setup, tab_results, tab_charts, tab_export, tab_guide = st.tabs([
    "🔧 Setup", "📊 Results", "📈 Charts", "💾 Export", "📖 Guide"
])


# ═══════════════════════════════════════════════
# TAB 1 — SETUP
# ═══════════════════════════════════════════════
with tab_setup:
    st.markdown('<h2 class="section-header">Experiment Setup</h2>', unsafe_allow_html=True)

    # Default date boundaries
    if weather_df is not None:
        min_d = weather_df["date"].min().date()
        max_d = weather_df["date"].max().date()
    else:
        min_d = date(2020, 1, 1)
        max_d = date(2025, 12, 31)

    experiments = []
    exp_cols = st.columns(min(n_experiments, 3))

    for i in range(n_experiments):
        col_idx = i % 3
        with exp_cols[col_idx]:
            st.markdown(f"**Experiment {i+1}**")
            sow = st.date_input(
                f"Sowing date",
                value=min_d + timedelta(days=i * 15),
                min_value=min_d, max_value=max_d,
                key=f"sow_{i}",
            )
            harv = st.date_input(
                f"Harvest date",
                value=min(min_d + timedelta(days=120 + i * 15), max_d),
                min_value=min_d, max_value=max_d,
                key=f"harv_{i}",
            )
            label = st.text_input(
                "Label",
                value=f"Experiment {i+1}",
                key=f"label_{i}",
            )
            experiments.append({"sow": sow, "harv": harv, "label": label})

    st.markdown("---")

    # ── Stages setup ────────────────────────
    stages = []
    if n_stages > 0:
        st.markdown('<h2 class="section-header">Crop Growth Stages</h2>', unsafe_allow_html=True)
        stage_cols = st.columns(min(n_stages, 4))
        default_stage_names = [
            "Emergence", "Tillering", "Jointing", "Booting",
            "Heading", "Anthesis", "Grain Fill", "Maturity"
        ]
        for j in range(n_stages):
            col_idx = j % 4
            with stage_cols[col_idx]:
                s_name = st.text_input(
                    "Stage name",
                    value=default_stage_names[j] if j < len(default_stage_names) else f"Stage {j+1}",
                    key=f"sname_{j}",
                )
                s_date = st.date_input(
                    "Date",
                    value=min_d + timedelta(days=30 + j * 15),
                    min_value=min_d, max_value=max_d,
                    key=f"sdate_{j}",
                )
                stages.append({"name": s_name, "date": s_date})

    st.markdown("---")

    # ── Run button ───────────────────────────
    run_ready = weather_df is not None
    if not run_ready:
        st.markdown('<div class="warning-box">⬅️ Upload a weather data file in the sidebar to enable analysis.</div>',
                    unsafe_allow_html=True)

    run_btn = st.button("▶ Run GDD Analysis", disabled=not run_ready, use_container_width=True)


# ═══════════════════════════════════════════════
# COMPUTE RESULTS (shared state via session)
# ═══════════════════════════════════════════════

if run_btn and weather_df is not None:
    all_results = {}
    for exp in experiments:
        if exp["sow"] >= exp["harv"]:
            st.warning(f"'{exp['label']}': sowing date must be before harvest date. Skipped.")
            continue
        res = process_experiment(
            weather_df,
            sowing_date=exp["sow"],
            harvest_date=exp["harv"],
            t_base=t_base,
            t_upper=t_upper,
            stages=stages if stages else None,
        )
        if res is None:
            st.warning(f"'{exp['label']}': No weather data found for the selected period.")
        all_results[exp["label"]] = res

    st.session_state["results"]     = all_results
    st.session_state["stages"]      = stages
    st.session_state["t_base"]      = t_base
    st.session_state["t_upper"]     = t_upper
    st.session_state["experiments"] = experiments
    st.success("✅ Analysis complete! View results in the other tabs.")


# ═══════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════
with tab_results:
    if "results" not in st.session_state:
        st.info("Run the analysis first (Setup tab).")
    else:
        results = st.session_state["results"]
        stages  = st.session_state["stages"]

        for exp_label, res in results.items():
            if res is None:
                continue

            st.markdown(f'<h2 class="section-header">📌 {exp_label}</h2>',
                        unsafe_allow_html=True)

            # Missing data warnings
            if res["missing_days"]:
                st.markdown(
                    f'<div class="warning-box">⚠️ {len(res["missing_days"])} date(s) missing '
                    f'from weather data for this period.</div>',
                    unsafe_allow_html=True,
                )
            if res["rows_dropped"]:
                st.markdown(
                    f'<div class="warning-box">⚠️ {res["rows_dropped"]} row(s) removed due '
                    f'to missing Tmax/Tmin values.</div>',
                    unsafe_allow_html=True,
                )

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            df = res["daily_df"]
            with c1:
                metric_card("Total GDD", f"{res['total_gdd']:.1f}")
            with c2:
                metric_card("Days in Period", len(df), unit=" days")
            with c3:
                metric_card("Avg Daily GDD", f"{df['daily_gdd'].mean():.2f}")
            with c4:
                metric_card("Peak Daily GDD", f"{df['daily_gdd'].max():.2f}")

            # Stage-wise GDD
            if res["stage_gdd"]:
                st.markdown("**Stage-wise Cumulative GDD**")
                stage_data = [
                    {"Stage": name, "Cumulative GDD (°C·days)": val if val is not None else "Outside Period"}
                    for name, val in res["stage_gdd"].items()
                ]
                st.dataframe(
                    pd.DataFrame(stage_data).style.format(
                        {"Cumulative GDD (°C·days)": lambda x: f"{x:.2f}" if isinstance(x, float) else x}
                    ),
                    use_container_width=True, hide_index=True,
                )

            # Full daily table
            with st.expander(f"📋 Full daily table — {exp_label}"):
                display_df = df[["date", "tmax", "tmin", "tmax_eff", "tmin_eff",
                                  "daily_gdd", "cumulative_gdd"]].copy()
                display_df.columns = [
                    "Date", "Tmax (°C)", "Tmin (°C)",
                    "Tmax Eff.", "Tmin Eff.",
                    "Daily GDD", "Cumulative GDD"
                ]
                display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
                st.dataframe(
                    display_df.style.format({
                        "Tmax (°C)": "{:.1f}", "Tmin (°C)": "{:.1f}",
                        "Tmax Eff.": "{:.1f}", "Tmin Eff.": "{:.1f}",
                        "Daily GDD": "{:.2f}", "Cumulative GDD": "{:.2f}",
                    }).background_gradient(subset=["Daily GDD"], cmap="YlGn"),
                    use_container_width=True, hide_index=True,
                )

            st.markdown("---")


# ═══════════════════════════════════════════════
# TAB 3 — CHARTS
# ═══════════════════════════════════════════════
with tab_charts:
    if "results" not in st.session_state:
        st.info("Run the analysis first (Setup tab).")
    else:
        results = st.session_state["results"]
        stages  = st.session_state["stages"]
        valid   = {k: v for k, v in results.items() if v}

        if not valid:
            st.warning("No valid results to plot.")
        else:
            st.plotly_chart(
                plot_cumulative_gdd(valid, stages if stages else None),
                use_container_width=True,
            )
            st.plotly_chart(plot_daily_gdd(valid), use_container_width=True)

            st.markdown("---")
            st.markdown("#### Temperature vs GDD by Experiment")
            for exp_label, res in valid.items():
                st.plotly_chart(
                    plot_temperature_gdd(res["daily_df"], exp_label,
                                         st.session_state["t_base"]),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════
# TAB 4 — EXPORT
# ═══════════════════════════════════════════════
with tab_export:
    if "results" not in st.session_state:
        st.info("Run the analysis first (Setup tab).")
    else:
        results = st.session_state["results"]
        valid   = {k: v for k, v in results.items() if v}

        st.markdown('<h2 class="section-header">Export Results</h2>',
                    unsafe_allow_html=True)

        if valid:
            excel_bytes = build_export_excel(valid)
            st.download_button(
                label="⬇️ Download Excel Workbook (.xlsx)",
                data=excel_bytes,
                file_name="gdd_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("**Or download individual CSV files:**")
            for exp_label, res in valid.items():
                df = res["daily_df"].copy()
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ {exp_label} — CSV",
                    data=csv_bytes,
                    file_name=f"gdd_{exp_label.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No valid results to export.")


# ═══════════════════════════════════════════════
# TAB 5 — GUIDE
# ═══════════════════════════════════════════════
with tab_guide:
    st.markdown('<h2 class="section-header">User Guide</h2>', unsafe_allow_html=True)

    with st.expander("📐 GDD Formula & Temperature Capping", expanded=True):
        st.markdown("""
**Basic formula:**
```
GDD = ((Tmax_eff + Tmin_eff) / 2) - Tbase
```
*(floored at 0 — GDD is never negative)*

**Temperature capping rules applied to both Tmax and Tmin:**

| Condition | Effective value used |
|---|---|
| Temperature ≥ Upper threshold (default 30 °C) | Upper threshold |
| Temperature ≤ Base temperature | Base temperature |
| Base temp < Temperature < Upper threshold | Actual temperature |
""")

    with st.expander("📂 Required CSV / Excel format"):
        sample = pd.DataFrame({
            "Date":  ["2023-06-01", "2023-06-02", "2023-06-03"],
            "Tmax":  [28.5, 31.2, 29.8],
            "Tmin":  [14.2, 16.0, 13.5],
        })
        st.dataframe(sample, hide_index=True)
        st.markdown("Column names are case-insensitive. Accepted variants: "
                    "`tmax` / `T_max` / `max_temp` / `maximum_temperature` etc.")

        csv_sample = sample.to_csv(index=False).encode()
        st.download_button("⬇️ Download sample CSV", csv_sample,
                           "sample_weather.csv", "text/csv")

    with st.expander("🚀 How to run this app"):
        st.code("""
# 1. Install dependencies
pip install streamlit pandas numpy plotly openpyxl

# 2. Run the app
streamlit run app.py

# 3. Open in browser (auto-opens at http://localhost:8501)
""", language="bash")

    with st.expander("💡 Future improvements"):
        st.markdown("""
- **NetCDF / ERA5 integration** — Auto-fetch gridded reanalysis data by lat/lon using `xarray` + `cfgrib`
- **NASA POWER API** — Pull daily weather data directly from NASA satellite estimates
- **Crop library** — Pre-loaded base temperatures for wheat, maize, rice, cotton, etc.
- **Phenology prediction** — Predict stage dates from accumulated GDD targets
- **Map visualisation** — Spatial GDD maps for regional trials
- **Soil temperature** — Switch from air to soil temperature for germination stages
- **Heat stress index** — Count days above upper threshold
- **GDD interpolation** — Hourly or sub-daily GDD calculation option
        """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">GDD Calculator · Built with Streamlit · '
    'Base temperature capping per CIMMYT/USDA modified protocol</div>',
    unsafe_allow_html=True,
)
