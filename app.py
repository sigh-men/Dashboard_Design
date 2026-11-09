"""
Clinic Wait Times & No-Shows Dashboard
Goal: help the clinic operations manager cut wait times and no-shows
without lowering patient satisfaction.
Data: data/clinic_visits.csv (one row per scheduled visit, 2025)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Clinic Wait Times & No-Shows", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "clinic_visits.csv"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
AGE_ORDER = ["0-17", "18-34", "35-49", "50-64", "65+"]
WAIT_BANDS = ["0–10 min", "11–30 min", "31–60 min", "60+ min"]

# Larger, readable fonts on every Plotly chart
FONT = dict(size=15)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV once and cache it so filters re-run instantly."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["is_no_show"] = df["no_show"].eq("Yes")
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    # Blank wait_minutes / satisfaction only happen on no-show rows: the
    # patient never arrived, so there was no wait and no survey. We keep
    # those rows (they count toward visits and no-show rate) but leave the
    # values blank, and pandas skips NaN when averaging wait/satisfaction.
    df["wait_band"] = pd.cut(
        df["wait_minutes"],
        bins=[-1, 10, 30, 60, float("inf")],
        labels=WAIT_BANDS,
    )
    return df


df = load_data(DATA_PATH)


# ---------------------------------------------------------------------------
# 2. FILTERS (sidebar)
# ---------------------------------------------------------------------------
def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_d, max_d = data["date"].min().date(), data["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )

    clinics = st.sidebar.multiselect(
        "Clinic", sorted(data["clinic"].unique()), default=sorted(data["clinic"].unique())
    )
    depts = st.sidebar.multiselect(
        "Department", sorted(data["department"].unique()),
        default=sorted(data["department"].unique()),
    )
    visit_type = st.sidebar.radio("Visit type", ["All", "In person", "Telehealth"])

    # date_input returns a 1-tuple while the user is mid-selection;
    # fall back to the full range until both ends are picked.
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    mask = (
        data["date"].dt.date.between(start, end)
        & data["clinic"].isin(clinics)
        & data["department"].isin(depts)
    )
    if visit_type != "All":
        mask &= data["visit_type"].eq(visit_type)

    st.sidebar.caption(
        "Blank values: wait time and satisfaction are blank only for no-shows. "
        "Those visits count toward total visits and no-show rate but are left out "
        "of wait and satisfaction averages."
    )
    return data[mask]


filtered = sidebar_filters(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Clinic Wait Times & No-Shows")
st.caption(
    "For the clinic operations manager (weekly) and department leads (monthly). "
    "Source: clinic_visits.csv, Jan–Dec 2025."
)

# Stop early if the filters remove everything
if filtered.empty:
    st.warning("No visits match these filters. Try selecting more clinics, "
               "departments, or a wider date range.")
    st.stop()

attended = filtered.dropna(subset=["wait_minutes"])  # rows with a recorded wait


# ---------------------------------------------------------------------------
# 3. KPIs
# ---------------------------------------------------------------------------
def show_kpis(data: pd.DataFrame, seen: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total visits", f"{len(data):,}")
    c2.metric("Median wait", f"{seen['wait_minutes'].median():.0f} min" if len(seen) else "No data")
    c3.metric("No-show rate", f"{data['is_no_show'].mean():.1%}")
    c4.metric("Avg satisfaction", f"{seen['satisfaction_1to5'].mean():.2f} / 5" if len(seen) else "No data")
    st.caption("Median wait and satisfaction use attended visits only.")


show_kpis(filtered, attended)
st.divider()


# ---------------------------------------------------------------------------
# 4. CHARTS
# ---------------------------------------------------------------------------
def chart_wait_by_department(seen: pd.DataFrame):
    """Q1: Which departments have the longest waits?"""
    g = (seen.groupby("department")["wait_minutes"].median()
         .sort_values().reset_index())
    fig = px.bar(
        g, x="wait_minutes", y="department", orientation="h",
        text=g["wait_minutes"].map(lambda v: f"{v:.0f} min"),
        labels={"wait_minutes": "Median wait (minutes)", "department": "Department"},
        color_discrete_sequence=["#3B6EA8"],
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="%{y}: %{x:.0f} min<extra></extra>")
    fig.update_layout(font=FONT, title="Q1. Median wait by department",
                      xaxis_range=[0, g["wait_minutes"].max() * 1.25],  # room for labels
                      margin=dict(l=10, r=40, t=50, b=10))
    return fig


def chart_satisfaction_by_wait(seen: pd.DataFrame):
    """Q2: Do long waits lower satisfaction?"""
    g = (seen.groupby("wait_band", observed=False)
         .agg(avg_sat=("satisfaction_1to5", "mean"), n=("satisfaction_1to5", "size"))
         .reset_index())
    fig = px.bar(
        g, x="wait_band", y="avg_sat",
        text=g["avg_sat"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "no data"),
        labels={"wait_band": "Wait time", "avg_sat": "Avg satisfaction (1–5)"},
        custom_data=["n"], color_discrete_sequence=["#3B6EA8"],
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x}<br>Avg: %{y:.2f} / 5<br>Visits: %{customdata[0]}<extra></extra>")
    fig.update_layout(font=FONT, title="Q2. Satisfaction by wait time",
                      yaxis_range=[0, 5.5], margin=dict(t=50, b=10))
    return fig


def chart_monthly_trend(data: pd.DataFrame, seen: pd.DataFrame):
    """Q3: When is demand highest, and do waits rise with it?"""
    visits = data.groupby("month", observed=False).size()
    wait = seen.groupby("month", observed=False)["wait_minutes"].median()
    fig = go.Figure()
    # Visits = bars, wait = dashed line with markers: the two series differ
    # by shape and line style, not just color.
    fig.add_bar(x=MONTH_ORDER, y=visits.values, name="Visits (count)",
                marker_color="#B8C7DA",
                hovertemplate="%{x}: %{y} visits<extra></extra>")
    fig.add_scatter(x=MONTH_ORDER, y=wait.values, name="Median wait (min)",
                    yaxis="y2", mode="lines+markers",
                    line=dict(color="#1F3B5C", width=3, dash="dash"),
                    marker=dict(symbol="diamond", size=9),
                    hovertemplate="%{x}: %{y:.0f} min median wait<extra></extra>")
    fig.update_layout(
        font=FONT, title="Q3. Monthly visits vs. median wait",
        xaxis_title="Month (2025)",
        yaxis=dict(title="Visits (count)"),
        yaxis2=dict(title="Median wait (minutes)", overlaying="y", side="right",
                    rangemode="tozero"),
        legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10),
    )
    return fig


def chart_no_show_heatmap(data: pd.DataFrame):
    """Q4: Which patient groups miss appointments most?"""
    pivot = (data.pivot_table(index="insurance", columns="age_group",
                              values="is_no_show", aggfunc="mean")
             .reindex(columns=[a for a in AGE_ORDER if a in data["age_group"].unique()])
             * 100)
    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Age group", y="Insurance", color="No-show %"),
    )
    fig.update_traces(hovertemplate="%{y}, age %{x}: %{z:.1f}% no-show<extra></extra>")
    fig.update_layout(font=FONT, title="Q4. No-show rate (%) by insurance and age",
                      margin=dict(t=50, b=10))
    return fig


left, right = st.columns(2)
with left:
    if attended.empty:
        st.info("No data: every visit in this selection was a no-show.")
    else:
        st.plotly_chart(chart_wait_by_department(attended), width="stretch")
        st.caption("Horizontal bars show the median wait in minutes for each department; "
                   "the value is printed at the end of each bar. Hover for details.")
with right:
    if not attended.empty:
        st.plotly_chart(chart_satisfaction_by_wait(attended), width="stretch")
        st.caption("Bars show average patient satisfaction (1 = worst, 5 = best) for each "
                   "wait-time range, with the average printed above each bar.")

left, right = st.columns(2)
with left:
    st.plotly_chart(chart_monthly_trend(filtered, attended), width="stretch")
    st.caption("Light bars show visit count per month (left axis); the dashed line with "
               "diamond markers shows median wait in minutes (right axis).")
with right:
    st.plotly_chart(chart_no_show_heatmap(filtered), width="stretch")
    st.caption("Each cell shows the percent of visits that were no-shows for one insurance "
               "type and age group. The number is printed in each cell; darker = higher.")

st.divider()


# ---------------------------------------------------------------------------
# 5. INSIGHTS (computed from the filtered data so the numbers stay accurate)
# ---------------------------------------------------------------------------
def build_insights(data: pd.DataFrame, seen: pd.DataFrame) -> list[str]:
    out = []

    # Q1: longest-wait department vs. the rest
    if seen["department"].nunique() >= 2:
        by_dept = seen.groupby("department")["wait_minutes"].median().sort_values()
        top, top_val = by_dept.index[-1], by_dept.iloc[-1]
        rest = seen[seen["department"] != top]["wait_minutes"].median()
        out.append(f"**Longest waits:** {top} has a median wait of {top_val:.0f} minutes, "
                   f"compared with {rest:.0f} minutes across the other departments.")

    # Q2: satisfaction at short vs. long waits
    sat = seen.groupby("wait_band", observed=False)["satisfaction_1to5"].mean().dropna()
    if len(sat) >= 2:
        corr = seen["wait_minutes"].corr(seen["satisfaction_1to5"])
        out.append(f"**Waits and satisfaction:** Average satisfaction is {sat.iloc[0]:.1f}/5 "
                   f"for {sat.index[0]} waits but {sat.iloc[-1]:.1f}/5 for {sat.index[-1]} "
                   f"waits (correlation {corr:.2f}).")

    # Q3: busiest month vs. typical month
    visits = data.groupby("month", observed=False).size()
    visits = visits[visits > 0]
    if len(visits) >= 2:
        busiest = visits.idxmax()
        pct = (visits.max() / visits.median() - 1) * 100
        out.append(f"**Peak demand:** {busiest} is the busiest month with {visits.max()} visits, "
                   f"{pct:.0f}% above a typical month ({visits.median():.0f} visits).")

    # Q4: highest vs. lowest no-show insurance group
    ns = data.groupby("insurance")["is_no_show"].mean().sort_values()
    if len(ns) >= 2:
        out.append(f"**No-shows:** {ns.index[-1]} patients miss {ns.iloc[-1]:.1%} of visits, "
                   f"the highest rate, versus {ns.iloc[0]:.1%} for {ns.index[0]} patients.")
    return out


st.subheader("Key takeaways")
st.caption("These update with the filters.")
insights = build_insights(filtered, attended)
if insights:
    for line in insights:
        st.markdown(f"- {line}")
else:
    st.info("Not enough data in this selection to compare groups.")
