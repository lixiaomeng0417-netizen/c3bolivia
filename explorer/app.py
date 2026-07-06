from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR.parent / "c3databolivia"
if not DATA_DIR.exists():
    DATA_DIR = APP_DIR / "data"
DEFAULT_DATASETS = {
    "Bolivia 112 provinces + SDG/satellite indicators": "bolivia112_v20260622.csv",
    "GDP per capita, wide 1990-2024": "gdp_perCapita_1990_2024.csv",
    "GDP per capita, long 1990-2024": "gdp_perCapita_long.csv",
    "SDG indexes + satellite embeddings 2017": "sdgs_satelliteEmbeddings2017.csv",
}
ID_COLS = ["prov_id", "prov", "dep", "dep_id", "dep_prov"]


st.set_page_config(
    page_title="C3 Bolivia Data Explorer",
    page_icon="🇧🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e7e2d8;}
    h1, h2, h3 {letter-spacing: 0;}
    div[data-testid="stMetric"] {
        border: 1px solid #e5e2dc;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        background: #fffefa;
    }
    .small-note {color: #5e635f; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def dataset_path(filename: str) -> Path:
    return DATA_DIR / filename


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def category_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in numeric_cols(df) and df[c].nunique(dropna=True) <= 60
    ]


def friendly(col: str) -> str:
    labels = {
        "prov": "Province",
        "dep": "Department",
        "imds": "IMDS",
        "gdppc": "GDP per capita",
        "log_gdppc": "Log GDP per capita",
        "population_2020": "Population 2020",
        "urbano_2012": "Urban share 2012",
    }
    if col in labels:
        return f"{labels[col]} ({col})"
    if col.startswith("index_sdg"):
        return f"SDG {col.replace('index_sdg', '')} index ({col})"
    return col


def year_from_col(col: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", col)
    return int(match.group(0)) if match else None


def infer_years(df: pd.DataFrame) -> list[int]:
    years = set()
    if "year" in df.columns:
        years.update(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int))
    for col in df.columns:
        year = year_from_col(col)
        if year:
            years.add(year)
    return sorted(years)


def make_panel(df: pd.DataFrame) -> pd.DataFrame | None:
    if {"year", "gdppc"}.issubset(df.columns):
        keep = [c for c in ["prov_id", "prov", "dep", "dep_id", "year", "gdppc", "log_gdppc"] if c in df.columns]
        return df[keep].copy()

    gdppc_cols = [c for c in df.columns if re.fullmatch(r"gdppc\d{4}", c)]
    if not gdppc_cols:
        return None

    id_cols = [c for c in ["prov_id", "prov", "dep", "dep_id"] if c in df.columns]
    long_df = df.melt(id_vars=id_cols, value_vars=gdppc_cols, var_name="indicator", value_name="gdppc")
    long_df["year"] = long_df["indicator"].str.extract(r"(\d{4})").astype(int)
    long_df["log_gdppc"] = np.log(long_df["gdppc"].where(long_df["gdppc"] > 0))
    return long_df.drop(columns=["indicator"])


def winsorize(df: pd.DataFrame, cols: list[str], lower: float, upper: float) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            lo, hi = out[col].quantile([lower, upper])
            out[col] = out[col].clip(lo, hi)
    return out


def apply_filters(df: pd.DataFrame, range_col: str | None, range_values, outlier_mode: str) -> pd.DataFrame:
    filtered = df.copy()
    if "dep" in filtered.columns and st.session_state.get("filter_dep"):
        filtered = filtered[filtered["dep"].isin(st.session_state["filter_dep"])]
    if "prov" in filtered.columns and st.session_state.get("filter_prov"):
        filtered = filtered[filtered["prov"].isin(st.session_state["filter_prov"])]
    if "year" in filtered.columns and st.session_state.get("year_range"):
        lo, hi = st.session_state["year_range"]
        filtered = filtered[(filtered["year"] >= lo) & (filtered["year"] <= hi)]
    if range_col and range_values:
        lo, hi = range_values
        filtered = filtered[(filtered[range_col] >= lo) & (filtered[range_col] <= hi)]

    if outlier_mode == "Winsorize 1%-99%":
        filtered = winsorize(filtered, numeric_cols(filtered), 0.01, 0.99)
    elif outlier_mode == "Winsorize 5%-95%":
        filtered = winsorize(filtered, numeric_cols(filtered), 0.05, 0.95)
    return filtered


def download_frame(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


def pca_2d(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat = (mat - mat.mean()) / mat.std(ddof=0).replace(0, 1)
    u, s, _ = np.linalg.svd(mat.to_numpy(), full_matrices=False)
    coords = u[:, :2] * s[:2]
    out = df[[c for c in ["prov", "dep", "imds"] if c in df.columns]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    return out


def metric_row(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows kept", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric variables", f"{len(numeric_cols(df)):,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")


def overview(df: pd.DataFrame, original: pd.DataFrame, panel: pd.DataFrame | None):
    st.header("Overview & Data")
    st.write(
        f"**Active sample** - {len(df):,} rows, {df.shape[1]:,} columns after subsetting and outlier treatment."
    )
    metric_row(df)

    with st.expander("Preview the analysis sample", expanded=False):
        st.dataframe(df.head(80), width="stretch", height=360)
        download_frame(df, "Download filtered sample", "c3bolivia_filtered_sample.csv")

    st.subheader("Panel balance & coverage")
    if panel is not None:
        coverage = panel.groupby("year")["prov_id" if "prov_id" in panel.columns else panel.columns[0]].nunique().reset_index(name="provinces")
        fig = px.bar(coverage, x="year", y="provinces", title="Observed provinces by year")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Balance summary"):
            st.dataframe(coverage.describe().T, width="stretch")
    else:
        present = pd.DataFrame({"column": original.columns, "non_missing": original.notna().sum().values})
        fig = px.bar(present.head(60), x="column", y="non_missing", title="Non-missing values by column")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Missing values")
    miss = df.isna().mean().sort_values(ascending=False).head(40).reset_index()
    miss.columns = ["variable", "missing_share"]
    fig = px.bar(miss, x="missing_share", y="variable", orientation="h", title="Top missingness rates")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    if panel is not None and "gdppc" in panel.columns:
        st.subheader("Value heatmap")
        value = st.selectbox("Variable", ["gdppc", "log_gdppc"], format_func=friendly)
        pivot = panel.pivot_table(index="dep", columns="year", values=value, aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", title=f"{friendly(value)} by department and year")
        st.plotly_chart(fig, width="stretch")


def describe_variables(df: pd.DataFrame):
    st.header("Describe Variables")
    nums = numeric_cols(df)
    if not nums:
        st.info("No numeric variables available in this filtered sample.")
        return
    selected = st.multiselect("Variables", nums, default=nums[: min(4, len(nums))], format_func=friendly)
    if not selected:
        st.stop()
    summary = df[selected].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary["missing"] = df[selected].isna().sum()
    st.dataframe(summary, width="stretch")

    var = st.selectbox("Distribution variable", selected, format_func=friendly)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x=var, color="dep" if "dep" in df.columns else None, marginal="box"), width="stretch")
    with c2:
        st.plotly_chart(px.box(df, y=var, x="dep" if "dep" in df.columns else None, points="outliers"), width="stretch")


def within_between(df: pd.DataFrame, panel: pd.DataFrame | None):
    st.header("Within & Between")
    if panel is not None:
        value = st.selectbox("Panel variable", [c for c in ["gdppc", "log_gdppc"] if c in panel.columns], format_func=friendly)
        unit = "prov" if "prov" in panel.columns else "prov_id"
        within = panel.groupby(unit)[value].std().rename("within_sd")
        between = panel.groupby("year")[value].std().rename("between_sd")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(within.reset_index(), x="within_sd", title="Within-province variation over time"), width="stretch")
        with c2:
            st.plotly_chart(px.line(between.reset_index(), x="year", y="between_sd", markers=True, title="Between-province dispersion by year"), width="stretch")
        st.dataframe(pd.concat([within.describe(), between.describe()], axis=1), width="stretch")
    else:
        nums = numeric_cols(df)
        group = st.selectbox("Group", category_cols(df), index=0 if category_cols(df) else None)
        value = st.selectbox("Variable", nums, format_func=friendly)
        agg = df.groupby(group)[value].agg(["count", "mean", "std", "min", "max"]).reset_index()
        st.plotly_chart(px.bar(agg, x=group, y="mean", error_y="std", title=f"{friendly(value)} by {group}"), width="stretch")
        st.dataframe(agg, width="stretch")


def trends(df: pd.DataFrame, panel: pd.DataFrame | None):
    st.header("Trends")
    if panel is None:
        st.info("Trend view is available for datasets with a year column or GDP year columns.")
        return
    value = st.selectbox("Trend variable", [c for c in ["gdppc", "log_gdppc"] if c in panel.columns], format_func=friendly)
    by = st.radio("Aggregate by", ["Department mean", "Selected provinces"], horizontal=True)
    if by == "Department mean" and "dep" in panel.columns:
        trend = panel.groupby(["year", "dep"], as_index=False)[value].mean()
        fig = px.line(trend, x="year", y=value, color="dep", markers=True, title=f"{friendly(value)} by department")
    else:
        provinces = sorted(panel["prov"].dropna().unique()) if "prov" in panel.columns else []
        default = provinces[:6]
        chosen = st.multiselect("Provinces", provinces, default=default)
        trend = panel[panel["prov"].isin(chosen)] if chosen else panel.head(0)
        fig = px.line(trend, x="year", y=value, color="prov", markers=True, title=f"{friendly(value)} for selected provinces")
    st.plotly_chart(fig, width="stretch")

    if {"prov", "year", value}.issubset(panel.columns):
        latest_year = int(panel["year"].max())
        latest = panel[panel["year"] == latest_year].nlargest(15, value)
        st.plotly_chart(px.bar(latest, x=value, y="prov", color="dep" if "dep" in latest else None, orientation="h", title=f"Top provinces in {latest_year}"), width="stretch")


def by_group(df: pd.DataFrame):
    st.header("By Group")
    cats = category_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        st.info("This view needs at least one categorical column and one numeric column.")
        return
    group = st.selectbox("Group", cats, index=cats.index("dep") if "dep" in cats else 0)
    value = st.selectbox("Value", nums, index=nums.index("imds") if "imds" in nums else 0, format_func=friendly)
    fig = px.box(df, x=group, y=value, color=group, points="all", title=f"{friendly(value)} distribution by {group}")
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")
    agg = df.groupby(group)[value].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    st.dataframe(agg.sort_values("mean", ascending=False), width="stretch")


def composition(df: pd.DataFrame):
    st.header("Composition")
    if "dep" not in df.columns:
        st.info("Composition view needs a department column.")
        return
    size_options = [c for c in ["population_2020", "imds", "gdppc"] if c in df.columns]
    size_options += [c for c in numeric_cols(df) if c not in size_options][:20]
    size = st.selectbox("Tile size", size_options, format_func=friendly)
    color = st.selectbox("Color", [c for c in ["imds", "population_2020", size] if c in df.columns] + [c for c in numeric_cols(df) if c not in [size]][:15], format_func=friendly)
    path = ["dep", "prov"] if "prov" in df.columns else ["dep"]
    fig = px.treemap(df, path=path, values=size, color=color, color_continuous_scale="RdYlGn", title="Department/province composition")
    st.plotly_chart(fig, width="stretch")


def relationships(df: pd.DataFrame):
    st.header("Relationships")
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.info("This view needs at least two numeric variables.")
        return
    default_x = "imds" if "imds" in nums else nums[0]
    default_y = "population_2020" if "population_2020" in nums else nums[1]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", nums, index=nums.index(default_x), format_func=friendly)
    y = c2.selectbox("Y", nums, index=nums.index(default_y), format_func=friendly)
    color = c3.selectbox("Color", ["dep"] + nums if "dep" in df.columns else nums, format_func=friendly)
    size_candidates = ["population_2020"] + nums
    size = st.selectbox("Point size", ["None"] + [c for c in size_candidates if c in df.columns], format_func=friendly)
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=None if size == "None" else size,
        hover_name="prov" if "prov" in df.columns else None,
        title=f"{friendly(y)} vs {friendly(x)}",
    )
    st.plotly_chart(fig, width="stretch")

    corr_vars = st.multiselect("Correlation variables", nums, default=nums[: min(12, len(nums))], format_func=friendly)
    if len(corr_vars) >= 2:
        corr = df[corr_vars].corr(numeric_only=True)
        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu", title="Correlation heatmap")
        st.plotly_chart(fig, width="stretch")


def dynamics(df: pd.DataFrame, panel: pd.DataFrame | None):
    st.header("Dynamics")
    if panel is not None and {"year", "gdppc", "prov"}.issubset(panel.columns):
        years = sorted(panel["year"].dropna().astype(int).unique())
        start, end = st.select_slider("Compare years", options=years, value=(years[0], years[-1]))
        wide = panel[panel["year"].isin([start, end])].pivot_table(index=["prov", "dep"], columns="year", values="gdppc").reset_index()
        wide = wide.dropna(subset=[start, end])
        wide["change"] = wide[end] - wide[start]
        wide["pct_change"] = np.where(wide[start] != 0, wide["change"] / wide[start] * 100, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.scatter(wide, x=start, y=end, color="dep", hover_name="prov", title=f"GDP per capita: {start} vs {end}"), width="stretch")
        with c2:
            top = wide.reindex(wide["change"].abs().sort_values(ascending=False).index).head(20)
            st.plotly_chart(px.bar(top, x="change", y="prov", color="dep", orientation="h", title="Largest absolute changes"), width="stretch")
        st.dataframe(wide.sort_values("pct_change", ascending=False), width="stretch")
        return

    embed_cols = [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]
    if embed_cols:
        st.subheader("Satellite embedding projection")
        proj = pca_2d(df, embed_cols)
        fig = px.scatter(proj, x="PC1", y="PC2", color="dep" if "dep" in proj.columns else None, hover_name="prov" if "prov" in proj.columns else None, size="imds" if "imds" in proj.columns else None)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Dynamics view is available for panel data or embedding columns named A00-A63.")


def notebook_template(dataset_name: str, config: dict) -> bytes:
    text = f"""# Reproduce C3 Bolivia exploration
import pandas as pd

dataset = {dataset_name!r}
config = {json.dumps(config, indent=2)}
df = pd.read_csv('data/' + dataset)
print(df.shape)
df.head()
"""
    return text.encode("utf-8")


with st.sidebar:
    st.title("ExPdPy")
    st.subheader("Data")
    dataset_label = st.selectbox("Dataset", list(DEFAULT_DATASETS), index=0)
    uploaded = st.file_uploader("...or upload your own data", type=["csv"])
    if uploaded is not None:
        raw = load_csv(uploaded)
        active_name = uploaded.name
    else:
        active_name = DEFAULT_DATASETS[dataset_label]
        raw = load_csv(dataset_path(active_name))

    st.caption(f"**Active sample** - {active_name}")

    years = infer_years(raw)
    if "year" in raw.columns and years:
        st.subheader("Sample")
        st.slider("Period", min_value=min(years), max_value=max(years), value=(min(years), max(years)), key="year_range")
        st.caption("Drag the handles together for a single-year cross-section.")

    if "dep" in raw.columns:
        st.multiselect("Filter by category", sorted(raw["dep"].dropna().unique()), key="filter_dep", placeholder="Choose departments")
    if "prov" in raw.columns:
        st.multiselect("Filter by province", sorted(raw["prov"].dropna().unique()), key="filter_prov", placeholder="Choose provinces")

    nums_for_range = numeric_cols(raw)
    range_col = st.selectbox("Filter by range", ["None"] + nums_for_range, format_func=friendly)
    range_values = None
    if range_col != "None":
        min_v = float(raw[range_col].min())
        max_v = float(raw[range_col].max())
        if np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v:
            range_values = st.slider(f"{friendly(range_col)} range", min_v, max_v, (min_v, max_v))

    outlier_mode = st.selectbox("Outlier treatment", ["None", "Winsorize 1%-99%", "Winsorize 5%-95%"])

    with st.expander("Advanced: user-defined variables"):
        st.caption("Create one ratio from two numeric columns.")
        ratio_name = st.text_input("New variable name", value="")
        numerator = st.selectbox("Numerator", ["None"] + nums_for_range, format_func=friendly)
        denominator = st.selectbox("Denominator", ["None"] + nums_for_range, format_func=friendly)
        if ratio_name and numerator != "None" and denominator != "None":
            raw = raw.copy()
            raw[ratio_name] = raw[numerator] / raw[denominator].replace(0, np.nan)
            st.success(f"Created {ratio_name}")

    st.divider()
    config = {
        "dataset": active_name,
        "departments": st.session_state.get("filter_dep", []),
        "provinces": st.session_state.get("filter_prov", []),
        "year_range": st.session_state.get("year_range"),
        "range_col": None if range_col == "None" else range_col,
        "outlier_treatment": outlier_mode,
    }
    st.download_button("Save config", json.dumps(config, indent=2).encode("utf-8"), "c3bolivia_explorer_config.json", "application/json", width="stretch")
    st.download_button("Export notebook + data", notebook_template(active_name, config), "c3bolivia_reproduce.py", "text/x-python", width="stretch")


df = apply_filters(raw, None if range_col == "None" else range_col, range_values, outlier_mode)
panel = make_panel(df)

page = st.sidebar.radio(
    "Explore",
    [
        "Overview & Data",
        "Describe variables",
        "Within & between",
        "Trends",
        "By group",
        "Composition",
        "Relationships",
        "Dynamics",
    ],
    label_visibility="collapsed",
)

st.title("C3 Bolivia")
st.caption("Province-level SDG, satellite, population, night lights, and GDP per capita exploration.")

if page == "Overview & Data":
    overview(df, raw, panel)
elif page == "Describe variables":
    describe_variables(df)
elif page == "Within & between":
    within_between(df, panel)
elif page == "Trends":
    trends(df, panel)
elif page == "By group":
    by_group(df)
elif page == "Composition":
    composition(df)
elif page == "Relationships":
    relationships(df)
elif page == "Dynamics":
    dynamics(df, panel)
