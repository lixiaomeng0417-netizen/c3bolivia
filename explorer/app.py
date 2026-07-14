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
    "Region names and province metadata": "regionNames.csv",
    "SDG indexes": "sdg.csv",
    "SDG variables": "sdgVariables.csv",
    "GDP per capita, wide 1990-2024": "gdp_perCapita_1990_2024.csv",
    "GDP per capita, long 1990-2024": "gdp_perCapita_long.csv",
    "Population panel, 2001-2020": "pop.csv",
    "Night-time lights panel, 2012-2020": "ln_NTLpc.csv",
    "Satellite embeddings 2017": "satelliteEmbeddings2017.csv",
    "SDG indexes + satellite embeddings 2017": "sdgs_satelliteEmbeddings2017.csv",
}
ID_COLS = ["prov_id", "prov", "dep", "dep_id", "dep_prov"]
GEOJSON_FILE = "bolivia112provincesOpt.geojson"


st.set_page_config(
    page_title="C3 Bolivia Data Explorer",
    page_icon="🇧🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3.5rem; max-width: 1320px;}
    [data-testid="stSidebar"] {border-right: 1px solid #e6e8eb; background: #fbfbf8;}
    h1, h2, h3 {letter-spacing: 0; color: #1f2933;}
    h1 {font-size: 2rem;}
    h2 {font-size: 1.45rem; margin-top: 1.8rem;}
    h3 {font-size: 1.08rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e1e5ea;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(31, 41, 51, 0.04);
    }
    .section-caption {color: #5d6975; font-size: 0.94rem; margin-top: -0.4rem; margin-bottom: 1rem;}
    .dashboard-rule {border-top: 1px solid #e6e8eb; margin: 1.6rem 0 1.2rem;}
    .summary-box {
        border-left: 4px solid #567568;
        background: #f6f8f5;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        color: #263238;
        margin: 0.5rem 0 1.2rem;
    }
    .warning-note {
        border-left: 4px solid #b7791f;
        background: #fff8e8;
        padding: 0.7rem 0.9rem;
        border-radius: 6px;
        color: #3f3422;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


@st.cache_data(show_spinner=False)
def load_region_names() -> pd.DataFrame:
    path = DATA_DIR / "regionNames.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_dataset(filename: str) -> pd.DataFrame:
    df = load_csv(dataset_path(filename))
    regions = load_region_names()
    if "prov_id" not in df.columns or regions.empty:
        return add_province_labels(df)

    add_cols = [c for c in ID_COLS + ["capital", "n_mun"] if c in regions.columns and c not in df.columns]
    if not add_cols:
        return add_province_labels(df)
    return add_province_labels(df.merge(regions[["prov_id"] + add_cols], on="prov_id", how="left"))


def dataset_path(filename: str) -> Path:
    return DATA_DIR / filename


def geojson_path() -> Path:
    return DATA_DIR / GEOJSON_FILE


def add_province_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "prov_label" in df.columns or "prov" not in df.columns:
        return df
    out = df.copy()
    if "dep_prov" in out.columns:
        out["prov_label"] = out["dep_prov"]
        return out
    if "dep" not in out.columns:
        out["prov_label"] = out["prov"]
        return out
    unique_pairs = out[["prov", "dep"]].drop_duplicates()
    repeated = unique_pairs["prov"].value_counts()
    repeated_names = set(repeated[repeated > 1].index)
    out["prov_label"] = np.where(
        out["prov"].isin(repeated_names),
        out["dep"].astype(str) + "-" + out["prov"].astype(str),
        out["prov"].astype(str),
    )
    return out


def province_label_column(df: pd.DataFrame) -> str:
    if "prov_label" in df.columns:
        return "prov_label"
    if "dep_prov" in df.columns:
        return "dep_prov"
    if "prov" in df.columns:
        return "prov"
    return "prov_id"


def unique_existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys([c for c in columns if c in df.columns]))


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


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
        "value": "Value",
        "metric": "Metric",
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
        return add_province_labels(df[keep].copy())

    gdppc_cols = [c for c in df.columns if re.fullmatch(r"gdppc\d{4}", c)]
    if gdppc_cols:
        id_cols = [c for c in ["prov_id", "prov", "dep", "dep_id"] if c in df.columns]
        long_df = df.melt(id_vars=id_cols, value_vars=gdppc_cols, var_name="indicator", value_name="gdppc")
        long_df["year"] = long_df["indicator"].str.extract(r"(\d{4})").astype(int)
        long_df["log_gdppc"] = np.log(long_df["gdppc"].where(long_df["gdppc"] > 0))
        return add_province_labels(long_df.drop(columns=["indicator"]))

    return make_yearly_panel(df)


def make_yearly_panel(df: pd.DataFrame) -> pd.DataFrame | None:
    specs = [
        (r"pop(\d{4})", "population"),
        (r"ln_NTLpc(\d{4})", "ln_NTLpc"),
        (r"ln_t400NTLpc(\d{4})", "ln_t400NTLpc"),
    ]
    frames = []
    id_cols = [c for c in ["prov_id", "prov", "dep", "dep_id"] if c in df.columns]
    for pattern, metric in specs:
        value_cols = [c for c in df.columns if re.fullmatch(pattern, c)]
        if not value_cols:
            continue
        melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="indicator", value_name="value")
        melted["year"] = melted["indicator"].str.extract(r"(\d{4})").astype(int)
        melted["metric"] = metric
        frames.append(melted.drop(columns=["indicator"]))
    if not frames:
        return None
    return add_province_labels(pd.concat(frames, ignore_index=True))


def panel_value_options(panel: pd.DataFrame) -> list[str]:
    options = [c for c in ["gdppc", "log_gdppc", "value"] if c in panel.columns]
    return options or [c for c in numeric_cols(panel) if c != "year"]


def filter_panel_metric(panel: pd.DataFrame, key: str) -> pd.DataFrame:
    if "metric" not in panel.columns:
        return panel
    metrics = sorted(panel["metric"].dropna().unique())
    chosen = st.selectbox("Panel metric", metrics, key=key)
    return panel[panel["metric"] == chosen].copy()


@st.cache_data(show_spinner=False)
def load_indicator_panel() -> pd.DataFrame:
    frames = []

    gdp_path = dataset_path("gdp_perCapita_long.csv")
    if gdp_path.exists():
        gdp = pd.read_csv(gdp_path)
        gdp = add_province_labels(gdp)
        gdp = gdp.rename(columns={"gdppc": "value"})
        gdp["indicator"] = "GDP per capita"
        frames.append(gdp[[c for c in ["prov_id", "prov", "prov_label", "dep", "dep_id", "year", "indicator", "value"] if c in gdp.columns]])

    for filename, patterns in {
        "pop.csv": [(r"pop(\d{4})", "Population")],
        "ln_NTLpc.csv": [
            (r"ln_NTLpc(\d{4})", "Night-time lights per capita"),
            (r"ln_t400NTLpc(\d{4})", "Trimmed night-time lights per capita"),
        ],
    }.items():
        path = dataset_path(filename)
        if not path.exists():
            continue
        source = load_dataset(filename)
        id_cols = [c for c in ["prov_id", "prov", "prov_label", "dep", "dep_id"] if c in source.columns]
        for pattern, indicator in patterns:
            value_cols = [c for c in source.columns if re.fullmatch(pattern, c)]
            if not value_cols:
                continue
            long_df = source.melt(id_vars=id_cols, value_vars=value_cols, var_name="source_column", value_name="value")
            long_df["year"] = long_df["source_column"].str.extract(r"(\d{4})").astype(int)
            long_df["indicator"] = indicator
            frames.append(long_df[id_cols + ["year", "indicator", "value"]])

    if not frames:
        return pd.DataFrame(columns=["prov_id", "prov", "prov_label", "dep", "dep_id", "year", "indicator", "value"])
    panel = pd.concat(frames, ignore_index=True)
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    return add_province_labels(panel.dropna(subset=["value", "year", "indicator"]))


@st.cache_data(show_spinner=False)
def load_gdp_panel() -> pd.DataFrame:
    path = dataset_path("gdp_perCapita_long.csv")
    if path.exists():
        return add_province_labels(pd.read_csv(path))
    wide_path = dataset_path("gdp_perCapita_1990_2024.csv")
    if not wide_path.exists():
        return pd.DataFrame()
    panel = make_panel(pd.read_csv(wide_path))
    return add_province_labels(panel) if panel is not None else pd.DataFrame()


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
    if st.session_state.get("filter_prov"):
        label_col = province_label_column(filtered)
        if label_col in filtered.columns:
            filtered = filtered[filtered[label_col].isin(st.session_state["filter_prov"])]
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
    out = df[[c for c in ["prov", "prov_label", "dep", "imds"] if c in df.columns]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    return out


def metric_row(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows kept", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric variables", f"{len(numeric_cols(df)):,}")


def panel_balance_status(panel: pd.DataFrame | None) -> tuple[str, pd.DataFrame | None]:
    if panel is None or "year" not in panel.columns:
        return "Not panel", None
    unit = "prov_id" if "prov_id" in panel.columns else panel.columns[0]
    counts = panel.groupby("year")[unit].nunique().reset_index(name="observed_units")
    if counts.empty:
        return "No panel rows", counts
    return ("Balanced" if counts["observed_units"].nunique() == 1 else "Unbalanced"), counts


def dataset_summary_sentence(df: pd.DataFrame, panel: pd.DataFrame | None, active_name: str) -> str:
    provinces = df["prov_id"].nunique() if "prov_id" in df.columns else "no province id"
    years = sorted(panel["year"].dropna().astype(int).unique()) if panel is not None and "year" in panel.columns else []
    year_text = f"{min(years)}-{max(years)}" if years else "cross-sectional"
    return (
        f"{active_name} contains {len(df):,} observations across {provinces} provinces, "
        f"covering {year_text} with {len(numeric_cols(df)):,} numeric variables available for analysis."
    )


def kpi_overview(df: pd.DataFrame, panel: pd.DataFrame | None):
    years = sorted(panel["year"].dropna().astype(int).unique()) if panel is not None and "year" in panel.columns else []
    balance, _ = panel_balance_status(panel)
    cards = st.columns(6)
    cards[0].metric("Observations", f"{len(df):,}")
    cards[1].metric("Provinces", f"{df['prov_id'].nunique():,}" if "prov_id" in df.columns else "N/A")
    cards[2].metric("Years", f"{len(years):,}" if years else "N/A")
    cards[3].metric("Variables", f"{df.shape[1]:,}")
    cards[4].metric("Numeric Variables", f"{len(numeric_cols(df)):,}")
    cards[5].metric("Panel", balance)


def outlier_summary(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows = []
    for col in numeric_cols(df):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isclose(iqr, 0):
            count = 0
        else:
            count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        rows.append({"variable": col, "outlier_share": count / len(series), "outliers": count})
    if not rows:
        return pd.DataFrame(columns=["variable", "outlier_share", "outliers"])
    return pd.DataFrame(rows).sort_values("outlier_share", ascending=False).head(limit)


def missing_summary(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).head(limit).reset_index()
    miss.columns = ["variable", "missing_share"]
    return miss[miss["missing_share"] > 0]


def selected_panel_view(panel: pd.DataFrame | None, key_prefix: str) -> tuple[pd.DataFrame | None, str | None]:
    if panel is None:
        return None, None
    panel_for_view = filter_panel_metric(panel, f"{key_prefix}_metric")
    value_options = panel_value_options(panel_for_view)
    if not value_options:
        return panel_for_view, None
    value = st.selectbox("Panel value", value_options, format_func=friendly, key=f"{key_prefix}_value")
    return panel_for_view, value


def data_catalog():
    st.header("Data Catalog")
    rows = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() == ".csv":
            sample = pd.read_csv(path, nrows=5)
            total_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
            rows.append(
                {
                    "file": path.name,
                    "type": "CSV",
                    "rows": total_rows,
                    "columns": sample.shape[1],
                    "sample_columns": ", ".join(sample.columns[:8]),
                }
            )
        elif path.suffix.lower() == ".geojson":
            geo = json.loads(path.read_text())
            rows.append(
                {
                    "file": path.name,
                    "type": "GeoJSON",
                    "rows": len(geo.get("features", [])),
                    "columns": len(geo.get("features", [{}])[0].get("properties", {})) if geo.get("features") else 0,
                    "sample_columns": ", ".join(geo.get("features", [{}])[0].get("properties", {}).keys()) if geo.get("features") else "",
                }
            )
    catalog = pd.DataFrame(rows)
    st.dataframe(catalog, width="stretch", hide_index=True)

    st.subheader("Quick file preview")
    csv_files = [r["file"] for r in rows if r["type"] == "CSV"]
    if csv_files:
        chosen = st.selectbox("Local file", csv_files)
        preview = load_dataset(chosen)
        metric_row(preview)
        st.dataframe(preview.head(50), width="stretch", height=320)

    if geojson_path().exists():
        with st.expander("Province boundary file"):
            geo = json.loads(geojson_path().read_text())
            props = pd.DataFrame([feature["properties"] for feature in geo.get("features", [])])
            st.write(f"GeoJSON contains **{len(props):,} province features**.")
            st.dataframe(props.head(50), width="stretch", hide_index=True)


def overview(df: pd.DataFrame, original: pd.DataFrame, panel: pd.DataFrame | None, active_name: str):
    st.header("Dataset Overview")
    st.markdown("<p class='section-caption'>A compact research-data summary before moving into diagnostics and variable-level analysis.</p>", unsafe_allow_html=True)
    kpi_overview(df, panel)
    st.markdown(f"<div class='summary-box'>{dataset_summary_sentence(df, panel, active_name)}</div>", unsafe_allow_html=True)

    with st.expander("Preview analysis sample", expanded=False):
        st.dataframe(df.head(80), width="stretch", height=340)
        download_frame(df, "Download filtered sample", "c3bolivia_filtered_sample.csv")

    st.markdown("<div class='dashboard-rule'></div>", unsafe_allow_html=True)
    st.header("Data Quality")
    st.markdown("<p class='section-caption'>Panel coverage, balance, missingness, and outlier checks are grouped here so the main analysis remains readable.</p>", unsafe_allow_html=True)
    balance, coverage = panel_balance_status(panel)
    left, right = st.columns(2)
    with left:
        st.subheader("Coverage & Balance")
        if coverage is not None and not coverage.empty:
            fig = px.bar(coverage, x="year", y="observed_units", title="Observed provinces by year", color_discrete_sequence=["#567568"])
            st.plotly_chart(fig, width="stretch")
            if balance != "Balanced":
                st.markdown("<div class='warning-note'>Panel is unbalanced across years; inspect coverage before interpreting trends.</div>", unsafe_allow_html=True)
        else:
            dtype_summary = original.dtypes.astype(str).value_counts().reset_index()
            dtype_summary.columns = ["dtype", "columns"]
            fig = px.bar(dtype_summary, x="dtype", y="columns", title="Column types", color_discrete_sequence=["#567568"])
            st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Missing Data")
        miss = missing_summary(df)
        if miss.empty:
            st.success("No missing values in the current filtered sample.")
        else:
            fig = px.bar(miss, x="missing_share", y="variable", orientation="h", title="Highest missing-value rates", color_discrete_sequence=["#b7791f"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with st.expander("Advanced diagnostics: outliers and balance table", expanded=False):
        diag_left, diag_right = st.columns(2)
        with diag_left:
            outliers = outlier_summary(df)
            if outliers.empty:
                st.info("Outlier summary is not available for this sample.")
            else:
                fig = px.bar(outliers, x="outlier_share", y="variable", orientation="h", title="Largest IQR outlier shares", color_discrete_sequence=["#9f6b54"])
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, width="stretch")
        with diag_right:
            if coverage is not None and not coverage.empty:
                st.dataframe(coverage.describe().T, width="stretch")
            else:
                st.dataframe(original.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "variable"}), width="stretch", hide_index=True)

    st.markdown("<div class='dashboard-rule'></div>", unsafe_allow_html=True)
    st.header("Variable Explorer")
    st.markdown("<p class='section-caption'>Choose a variable first, then inspect summary statistics, distribution, trend, and the higher-load heatmap last.</p>", unsafe_allow_html=True)

    panel_for_view, panel_value = selected_panel_view(panel, "explorer") if panel is not None else (None, None)
    if panel_for_view is not None and panel_value is not None:
        variable = panel_value
        explorer_df = panel_for_view.copy()
    else:
        nums = numeric_cols(df)
        if not nums:
            st.info("No numeric variables available for exploration.")
            return
        variable = st.selectbox("Variable", nums, format_func=friendly, key="explorer_static_variable")
        explorer_df = df.copy()

    stat_col, dist_col = st.columns(2)
    with stat_col:
        st.subheader("Summary Statistics")
        stats = pd.to_numeric(explorer_df[variable], errors="coerce").describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("value")
        st.dataframe(stats, width="stretch")
    with dist_col:
        st.subheader("Distribution")
        fig = px.histogram(
            explorer_df,
            x=variable,
            color="dep" if "dep" in explorer_df.columns else None,
            nbins=28,
            title=f"Distribution of {friendly(variable)}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, width="stretch")

    trend_col, heatmap_col = st.columns(2)
    with trend_col:
        st.subheader("Time Trend")
        if panel_for_view is not None and panel_value is not None and "year" in explorer_df.columns:
            trend = explorer_df.groupby(["year", "dep"], as_index=False)[variable].mean() if "dep" in explorer_df.columns else explorer_df.groupby("year", as_index=False)[variable].mean()
            fig = px.line(trend, x="year", y=variable, color="dep" if "dep" in trend.columns else None, markers=True, title=f"{friendly(variable)} over time")
            st.plotly_chart(fig, width="stretch")
        else:
            group = "dep" if "dep" in explorer_df.columns else None
            if group:
                grouped = explorer_df.groupby(group, as_index=False)[variable].mean()
                fig = px.bar(grouped, x=group, y=variable, title=f"Mean {friendly(variable)} by department", color_discrete_sequence=["#567568"])
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Time trend requires a panel dataset.")
    with heatmap_col:
        st.subheader("Heatmap")
        if panel_for_view is not None and panel_value is not None and "year" in explorer_df.columns:
            index_col = "dep" if "dep" in explorer_df.columns else "prov"
            pivot = explorer_df.pivot_table(index=index_col, columns="year", values=variable, aggfunc="mean")
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", title=f"{friendly(variable)} by {index_col} and year")
            st.plotly_chart(fig, width="stretch")
        elif geojson_path().exists() and "prov_id" in explorer_df.columns:
            geo = json.loads(geojson_path().read_text())
            fig = px.choropleth(
                explorer_df,
                geojson=geo,
                locations="prov_id",
                featureidkey="properties.prov_id",
                color=variable,
                hover_name=province_label_column(explorer_df),
                color_continuous_scale="Viridis",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(title=f"{friendly(variable)} by province", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Heatmap requires panel years or province geography.")


def indicator_sample():
    st.header("Indicator Sample")
    panel = load_indicator_panel()
    if panel.empty:
        st.info("No annual indicator data found.")
        return

    if st.session_state.get("filter_dep") and "dep" in panel.columns:
        panel = panel[panel["dep"].isin(st.session_state["filter_dep"])]
    if st.session_state.get("filter_prov") and "prov" in panel.columns:
        panel = panel[panel[province_label_column(panel)].isin(st.session_state["filter_prov"])]

    c1, c2, c3 = st.columns([1.4, 1, 1])
    indicators = sorted(panel["indicator"].dropna().unique())
    indicator = c1.selectbox("Indicator", indicators)
    indicator_panel = panel[panel["indicator"] == indicator].copy()
    years = sorted(indicator_panel["year"].dropna().astype(int).unique())
    year = c2.selectbox("Year", years, index=len(years) - 1)
    mode = c3.selectbox("Sample mode", ["Top values", "Bottom values", "Random sample"])

    year_df = indicator_panel[indicator_panel["year"] == year].dropna(subset=["value"]).copy()
    max_n = max(int(len(year_df)), 1)
    default_n = min(20, max_n)
    sample_size = st.slider("Sample size", min_value=1, max_value=max_n, value=default_n)

    if mode == "Top values":
        sample = year_df.nlargest(sample_size, "value")
    elif mode == "Bottom values":
        sample = year_df.nsmallest(sample_size, "value")
    else:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        sample = year_df.sample(n=sample_size, random_state=int(seed))

    sample = sample.sort_values("value", ascending=True)
    label_col = province_label_column(sample)
    fig = px.bar(
        sample,
        x="value",
        y=label_col,
        color="dep" if "dep" in sample.columns else None,
        orientation="h",
        hover_data=[c for c in ["prov", "prov_id", "dep", "year"] if c in sample.columns and c != label_col],
        title=f"{indicator}: {year} sample ({sample_size} of {max_n})",
    )
    fig.update_layout(yaxis_title="", xaxis_title=indicator)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available years", f"{min(years)}-{max(years)}")
    c2.metric("Year sample", f"{len(year_df):,}")
    c3.metric("Shown", f"{len(sample):,}")
    c4.metric("Median", f"{year_df['value'].median():,.2f}")

    with st.expander("Sample table"):
        st.dataframe(sample.sort_values("value", ascending=False), width="stretch", hide_index=True)
        download_frame(sample, "Download shown sample", f"c3bolivia_{indicator.lower().replace(' ', '_')}_{year}_sample.csv")


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
        panel_for_view = filter_panel_metric(panel, "within_metric")
        value = st.selectbox("Panel variable", panel_value_options(panel_for_view), format_func=friendly)
        unit = "prov" if "prov" in panel_for_view.columns else "prov_id"
        within = panel_for_view.groupby(unit)[value].std().rename("within_sd")
        between = panel_for_view.groupby("year")[value].std().rename("between_sd")
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
    path = ["dep", province_label_column(df)] if "prov" in df.columns else ["dep"]
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
        hover_name=province_label_column(df) if "prov" in df.columns or "prov_label" in df.columns else None,
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
    if panel is not None and {"year", "prov"}.issubset(panel.columns):
        panel_for_view = filter_panel_metric(panel, "dynamics_metric")
        label_col = province_label_column(panel_for_view)
        value = st.selectbox("Dynamic variable", panel_value_options(panel_for_view), format_func=friendly)
        years = sorted(panel_for_view["year"].dropna().astype(int).unique())
        start, end = st.select_slider("Compare years", options=years, value=(years[0], years[-1]))
        wide = panel_for_view[panel_for_view["year"].isin([start, end])].pivot_table(index=[label_col, "dep"], columns="year", values=value).reset_index()
        wide = wide.dropna(subset=[start, end])
        wide["change"] = wide[end] - wide[start]
        wide["pct_change"] = np.where(wide[start] != 0, wide["change"] / wide[start] * 100, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.scatter(wide, x=start, y=end, color="dep", hover_name=label_col, title=f"{friendly(value)}: {start} vs {end}"), width="stretch")
        with c2:
            top = wide.reindex(wide["change"].abs().sort_values(ascending=False).index).head(20)
            st.plotly_chart(px.bar(top, x="change", y=label_col, color="dep", orientation="h", title="Largest absolute changes"), width="stretch")
        st.dataframe(wide.sort_values("pct_change", ascending=False), width="stretch")
        return

    embed_cols = [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]
    if embed_cols:
        st.subheader("Satellite embedding projection")
        proj = pca_2d(df, embed_cols)
        fig = px.scatter(proj, x="PC1", y="PC2", color="dep" if "dep" in proj.columns else None, hover_name=province_label_column(proj) if "prov" in proj.columns or "prov_label" in proj.columns else None, size="imds" if "imds" in proj.columns else None)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Dynamics view is available for panel data or embedding columns named A00-A63.")


def gini(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or np.isclose(x.sum(), 0):
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def polynomial_terms(series: pd.Series, degree: int, prefix: str = "log_gdp_pc") -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce")
    terms = pd.DataFrame(index=series.index)
    for power in range(1, degree + 1):
        name = prefix if power == 1 else f"{prefix}_{power}"
        terms[name] = x ** power
    return terms


def fit_kuznets_model(data: pd.DataFrame, estimator: str, degree: int) -> dict:
    y_col = "gini_regional"
    x_col = "log_gdp_pc"
    use = data.dropna(subset=[y_col, x_col, "dep", "year"]).copy()
    if estimator == "Between":
        use = use.groupby("dep", as_index=False)[[y_col, x_col]].mean()

    x_terms = polynomial_terms(use[x_col], degree)
    design = x_terms.copy()
    if estimator == "Within two-way FE":
        dep_dummies = pd.get_dummies(use["dep"], prefix="dep", drop_first=True, dtype=float)
        year_dummies = pd.get_dummies(use["year"].astype(int), prefix="year", drop_first=True, dtype=float)
        design = pd.concat([design, dep_dummies, year_dummies], axis=1)
        include_intercept = True
    else:
        include_intercept = True

    if include_intercept:
        design.insert(0, "Intercept", 1.0)

    X = design.to_numpy(dtype=float)
    y = use[y_col].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if np.isclose(ss_tot, 0) else 1 - ss_res / ss_tot

    coefficients = pd.Series(beta, index=design.columns)
    poly_names = list(x_terms.columns)
    return {
        "estimator": estimator,
        "data": use,
        "coefficients": coefficients,
        "poly_names": poly_names,
        "n_obs": len(use),
        "n_units": use["dep"].nunique() if "dep" in use.columns else len(use),
        "r2": r2,
    }


def kuznets_turning_points(model: dict, x_min: float, x_max: float) -> pd.DataFrame:
    coefs = model["coefficients"]
    names = model["poly_names"]
    b1 = float(coefs.get(names[0], 0.0)) if names else 0.0
    b2 = float(coefs.get(names[1], 0.0)) if len(names) >= 2 else 0.0
    b3 = float(coefs.get(names[2], 0.0)) if len(names) >= 3 else 0.0
    roots = []
    if len(names) == 2 and not np.isclose(b2, 0):
        roots = [-b1 / (2 * b2)]
    elif len(names) >= 3:
        roots = [r.real for r in np.roots([3 * b3, 2 * b2, b1]) if np.isclose(r.imag, 0)]

    rows = []
    for root in roots:
        if x_min <= root <= x_max:
            second = 2 * b2 + 6 * b3 * root
            rows.append(
                {
                    "estimator": model["estimator"],
                    "log_gdp_pc": root,
                    "gdp_pc": float(np.exp(root)),
                    "type": "Peak" if second < 0 else "Trough",
                }
            )
    return pd.DataFrame(rows)


def predict_kuznets_curve(model: dict, x_grid: np.ndarray) -> pd.DataFrame:
    coefs = model["coefficients"]
    names = model["poly_names"]
    component = np.zeros_like(x_grid, dtype=float)
    for power, name in enumerate(names, start=1):
        component += float(coefs.get(name, 0.0)) * (x_grid ** power)

    if model["estimator"] == "Within two-way FE":
        sample_x = model["data"]["log_gdp_pc"]
        sample_component = np.zeros(len(sample_x), dtype=float)
        for power, name in enumerate(names, start=1):
            sample_component += float(coefs.get(name, 0.0)) * (sample_x.to_numpy(dtype=float) ** power)
        y_hat = model["data"]["gini_regional"].mean() + component - sample_component.mean()
    else:
        y_hat = float(coefs.get("Intercept", 0.0)) + component

    return pd.DataFrame(
        {
            "log_gdp_pc": x_grid,
            "gdp_pc": np.exp(x_grid),
            "fitted_gini": y_hat,
            "estimator": model["estimator"],
        }
    )


def build_kuznets_wave_panel(gdp: pd.DataFrame, min_provinces: int) -> pd.DataFrame:
    use = gdp.copy()
    use["gdppc"] = pd.to_numeric(use["gdppc"], errors="coerce")
    use = use.dropna(subset=["dep", "year", "prov_id", "gdppc"])
    use = use[use["gdppc"] > 0]
    panel = (
        use.groupby(["dep", "year"])
        .agg(
            gini_regional=("gdppc", gini),
            mean_gdp_pc=("gdppc", "mean"),
            median_gdp_pc=("gdppc", "median"),
            sd_log_gdp_pc=("gdppc", lambda s: np.log(s).std(ddof=0)),
            provinces=("prov_id", "nunique"),
        )
        .reset_index()
    )
    panel = panel[panel["provinces"] >= min_provinces].copy()
    panel["log_gdp_pc"] = np.log(panel["mean_gdp_pc"])
    panel["log_gdp_pc_2"] = panel["log_gdp_pc"] ** 2
    panel["log_gdp_pc_3"] = panel["log_gdp_pc"] ** 3
    return panel.dropna(subset=["gini_regional", "log_gdp_pc"])


def kuznets_waves():
    st.header("Kuznets-Waves Curve")
    st.markdown(
        "<p class='section-caption'>Modeled after the expdpy Kuznets workflow: regional inequality is regressed on a polynomial in log GDP per capita under pooled, between, and two-way fixed-effect specifications.</p>",
        unsafe_allow_html=True,
    )
    gdp = load_gdp_panel()
    if gdp.empty:
        st.info("GDP per capita files were not found in the local c3databolivia folder.")
        return

    if st.session_state.get("filter_dep") and "dep" in gdp.columns:
        gdp = gdp[gdp["dep"].isin(st.session_state["filter_dep"])]

    c1, c2, c3 = st.columns(3)
    years = sorted(gdp["year"].dropna().astype(int).unique())
    start, end = c1.select_slider("Kuznets period", options=years, value=(years[0], years[-1]))
    degree = c2.selectbox("Polynomial degree", [3, 2], format_func=lambda d: "Cubic / N-shaped wave" if d == 3 else "Quadratic / inverted-U")
    min_provinces = c3.slider("Minimum provinces per department-year", 2, 8, 3)

    gdp = gdp[(gdp["year"] >= start) & (gdp["year"] <= end)].copy()
    panel = build_kuznets_wave_panel(gdp, min_provinces)
    if len(panel) < degree + 5 or panel["dep"].nunique() < 2:
        st.warning("Not enough department-year observations for a stable Kuznets-waves estimate after filtering.")
        return

    estimators = st.multiselect(
        "Estimators",
        ["Pooled OLS", "Between", "Within two-way FE"],
        default=["Pooled OLS", "Between", "Within two-way FE"],
    )
    models = [fit_kuznets_model(panel, estimator, degree) for estimator in estimators]
    x_grid = np.linspace(panel["log_gdp_pc"].min(), panel["log_gdp_pc"].max(), 160)
    curves = pd.concat([predict_kuznets_curve(model, x_grid) for model in models], ignore_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Department-years", f"{len(panel):,}")
    c2.metric("Departments", f"{panel['dep'].nunique():,}")
    c3.metric("Period", f"{int(panel['year'].min())}-{int(panel['year'].max())}")
    c4.metric("Median regional Gini", f"{panel['gini_regional'].median():.3f}")

    fig = go.Figure()
    for dep, group in panel.groupby("dep"):
        fig.add_trace(
            go.Scatter(
                x=group["log_gdp_pc"],
                y=group["gini_regional"],
                mode="markers",
                name=dep,
                marker=dict(size=8, opacity=0.58),
                hovertemplate="dep=%{text}<br>year=%{customdata[0]}<br>Gini=%{y:.3f}<br>log GDPpc=%{x:.2f}<extra></extra>",
                text=[dep] * len(group),
                customdata=group[["year"]].to_numpy(),
                legendgroup="departments",
                showlegend=False,
            )
        )
    palette = {"Pooled OLS": "#2f6f73", "Between": "#8a5a44", "Within two-way FE": "#8b3a62"}
    for estimator, curve in curves.groupby("estimator"):
        fig.add_trace(
            go.Scatter(
                x=curve["log_gdp_pc"],
                y=curve["fitted_gini"],
                mode="lines",
                name=estimator,
                line=dict(width=3, color=palette.get(estimator)),
                hovertemplate=f"{estimator}<br>Gini=%{{y:.3f}}<br>log GDPpc=%{{x:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Kuznets-waves: regional GDPpc inequality vs development",
        xaxis_title="log(mean GDP per capita), department-year",
        yaxis_title="Within-department province GDPpc Gini",
        legend_title="Estimator",
    )
    st.plotly_chart(fig, width="stretch")

    coef_rows = []
    turning_rows = []
    for model in models:
        coefs = model["coefficients"]
        for name in ["Intercept"] + model["poly_names"]:
            if name in coefs.index:
                coef_rows.append({"estimator": model["estimator"], "term": name, "coefficient": coefs[name]})
        turning = kuznets_turning_points(model, panel["log_gdp_pc"].min(), panel["log_gdp_pc"].max())
        if not turning.empty:
            turning_rows.append(turning)

    left, right = st.columns(2)
    with left:
        st.subheader("Model Summary")
        summary = pd.DataFrame(
            [
                {
                    "estimator": model["estimator"],
                    "observations": model["n_obs"],
                    "departments": model["n_units"],
                    "r2": model["r2"],
                }
                for model in models
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)
    with right:
        st.subheader("Implied Turning Points")
        if turning_rows:
            st.dataframe(pd.concat(turning_rows, ignore_index=True), width="stretch", hide_index=True)
        else:
            st.info("No peak or trough falls inside the observed log GDPpc range.")

    with st.expander("Coefficient table and constructed Kuznets panel"):
        st.dataframe(pd.DataFrame(coef_rows), width="stretch", hide_index=True)
        st.dataframe(panel.sort_values(["dep", "year"]), width="stretch", hide_index=True)
        download_frame(panel, "Download Kuznets department-year panel", "c3bolivia_kuznets_waves_panel.csv")


def gdp_deep_dive():
    st.header("GDP Per Capita Deep Dive")
    gdp = load_gdp_panel()
    if gdp.empty:
        st.info("GDP per capita files were not found in the local c3databolivia folder.")
        return

    if st.session_state.get("filter_dep") and "dep" in gdp.columns:
        gdp = gdp[gdp["dep"].isin(st.session_state["filter_dep"])]
    if st.session_state.get("filter_prov") and "prov" in gdp.columns:
        gdp = gdp[gdp[province_label_column(gdp)].isin(st.session_state["filter_prov"])]

    years = sorted(gdp["year"].dropna().astype(int).unique())
    start, end = st.select_slider("GDP analysis period", options=years, value=(years[0], years[-1]))
    gdp = gdp[(gdp["year"] >= start) & (gdp["year"] <= end)].copy()
    gdp["gdppc"] = pd.to_numeric(gdp["gdppc"], errors="coerce")
    gdp["log_gdppc"] = np.log(gdp["gdppc"].where(gdp["gdppc"] > 0))
    label_col = province_label_column(gdp)

    latest_year = int(gdp["year"].max())
    first_year = int(gdp["year"].min())
    latest = gdp[gdp["year"] == latest_year].copy()
    first = gdp[gdp["year"] == first_year].copy()
    merged = first[[c for c in ["prov_id", "prov", "prov_label", "dep", "gdppc"] if c in first.columns]].merge(
        latest[["prov_id", "gdppc"]],
        on="prov_id",
        suffixes=("_start", "_end"),
    )
    merged_label_col = province_label_column(merged)
    merged["change"] = merged["gdppc_end"] - merged["gdppc_start"]
    merged["pct_change"] = np.where(merged["gdppc_start"] > 0, merged["change"] / merged["gdppc_start"] * 100, np.nan)
    years_elapsed = max(latest_year - first_year, 1)
    merged["cagr"] = np.where(
        (merged["gdppc_start"] > 0) & (merged["gdppc_end"] > 0),
        (merged["gdppc_end"] / merged["gdppc_start"]) ** (1 / years_elapsed) - 1,
        np.nan,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period", f"{first_year}-{latest_year}")
    c2.metric("Provinces", f"{latest['prov_id'].nunique():,}")
    c3.metric(f"Median GDPpc {latest_year}", f"{latest['gdppc'].median():,.0f}")
    c4.metric("Median CAGR", f"{merged['cagr'].median() * 100:,.2f}%")

    st.subheader("Growth leaders and laggards")
    top_n = st.slider("Number of provinces", 5, 30, 15)
    c1, c2 = st.columns(2)
    with c1:
        top_growth = merged.nlargest(top_n, "cagr")
        fig = px.bar(
            top_growth,
            x="cagr",
            y=merged_label_col,
            color="dep",
            orientation="h",
            title=f"Fastest annual growth, {first_year}-{latest_year}",
        )
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        bottom_growth = merged.nsmallest(top_n, "cagr")
        fig = px.bar(
            bottom_growth,
            x="cagr",
            y=merged_label_col,
            color="dep",
            orientation="h",
            title=f"Slowest annual growth, {first_year}-{latest_year}",
        )
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Ranking shifts")
    ranks = gdp.copy()
    ranks["rank"] = ranks.groupby("year")["gdppc"].rank(ascending=False, method="min")
    rank_wide = ranks[ranks["year"].isin([first_year, latest_year])].pivot_table(
        index=unique_existing(["prov_id", label_col, "prov", "dep"], ranks),
        columns="year",
        values="rank",
    ).reset_index()
    rank_wide["rank_change"] = rank_wide[first_year] - rank_wide[latest_year]
    movers = rank_wide.reindex(rank_wide["rank_change"].abs().sort_values(ascending=False).index).head(top_n)
    c1, c2 = st.columns(2)
    with c1:
        rank_label_col = province_label_column(movers)
        fig = px.bar(movers, x="rank_change", y=rank_label_col, color="dep", orientation="h", title="Largest rank moves")
        st.plotly_chart(fig, width="stretch")
    with c2:
        track_label_col = province_label_column(gdp)
        rank_options = sorted(gdp[track_label_col].dropna().unique())
        chosen = st.multiselect(
            "Track province ranks",
            rank_options,
            default=rank_options[:6],
        )
        rank_lines = ranks[ranks[track_label_col].isin(chosen)] if chosen else ranks.head(0)
        fig = px.line(rank_lines, x="year", y="rank", color=track_label_col, markers=True, title="GDPpc rank over time")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Distribution and inequality")
    yearly = gdp.groupby("year")["gdppc"].agg(
        mean="mean",
        median="median",
        p10=lambda s: s.quantile(0.10),
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
        gini=gini,
    ).reset_index()
    yearly["p90_p10_ratio"] = yearly["p90"] / yearly["p10"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["p90"], name="P90", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["p10"], name="P10", fill="tonexty", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["median"], name="Median", line=dict(width=3)))
        fig.update_layout(title="GDPpc distribution band", xaxis_title="Year", yaxis_title="GDP per capita")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(yearly, x="year", y=["gini", "p90_p10_ratio"], markers=True, title="Inequality over time")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Convergence")
    conv = first[[c for c in ["prov_id", "prov", "prov_label", "dep", "gdppc"] if c in first.columns]].merge(
        merged[["prov_id", "cagr"]],
        on="prov_id",
    )
    fig = px.scatter(
        conv,
        x="gdppc",
        y="cagr",
        color="dep",
        hover_name=province_label_column(conv),
        trendline=None,
        title=f"Initial GDPpc vs subsequent annual growth ({first_year}-{latest_year})",
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Department comparison")
    dep_yearly = gdp.groupby(["year", "dep"], as_index=False)["gdppc"].agg(["mean", "median"]).reset_index()
    fig = px.line(dep_yearly, x="year", y="median", color="dep", markers=True, title="Median GDPpc by department")
    st.plotly_chart(fig, width="stretch")

    with st.expander("GDP analysis table"):
        st.dataframe(
            merged.sort_values("cagr", ascending=False).assign(cagr=lambda d: d["cagr"] * 100),
            width="stretch",
        )
        download_frame(merged, "Download GDP growth table", "c3bolivia_gdppc_growth.csv")


def notebook_template(dataset_name: str, config: dict) -> bytes:
    text = f"""# Reproduce C3 Bolivia exploration locally
import pandas as pd
from pathlib import Path

dataset = {dataset_name!r}
config = {json.dumps(config, indent=2)}
data_dir = Path('..') / 'c3databolivia'
df = pd.read_csv(data_dir / dataset)
print(df.shape)
df.head()
"""
    return text.encode("utf-8")


with st.sidebar:
    st.title("Data Setup")
    st.subheader("Dataset")
    st.info(
        "Private/local mode: this app reads CSV files from your local repository only. "
        "Do not deploy or push private data to a public hosting service.",
    )
    dataset_label = st.selectbox("Dataset", list(DEFAULT_DATASETS), index=0)
    active_name = DEFAULT_DATASETS[dataset_label]
    raw = load_dataset(active_name)

    st.caption(f"Active local file: `{active_name}`")

    years = infer_years(raw)
    if "year" in raw.columns and years:
        st.subheader("Temporal Sample")
        st.slider("Period", min_value=min(years), max_value=max(years), value=(min(years), max(years)), key="year_range")
        st.caption("Drag the handles together for a single-year cross-section.")

    st.subheader("Geographic Filters")
    if "dep" in raw.columns:
        st.multiselect("Filter by category", sorted(raw["dep"].dropna().unique()), key="filter_dep", placeholder="Choose departments")
    province_filter_col = province_label_column(raw)
    if province_filter_col in raw.columns:
        st.multiselect("Filter by province", sorted(raw[province_filter_col].dropna().unique()), key="filter_prov", placeholder="Choose provinces")

    st.subheader("Preprocessing")
    nums_for_range = numeric_cols(raw)
    range_col = st.selectbox("Filter by range", ["None"] + nums_for_range, format_func=friendly)
    range_values = None
    if range_col != "None":
        min_v = float(raw[range_col].min())
        max_v = float(raw[range_col].max())
        if np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v:
            range_values = st.slider(f"{friendly(range_col)} range", min_v, max_v, (min_v, max_v))

    outlier_mode = st.selectbox("Outlier treatment", ["None", "Winsorize 1%-99%", "Winsorize 5%-95%"])

    with st.expander("Advanced options"):
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

st.title("C3 Bolivia")
st.caption("Research-data workbench for province-level SDG, satellite, population, night lights, and GDP per capita data.")

overview(df, raw, panel, active_name)

st.markdown("<div class='dashboard-rule'></div>", unsafe_allow_html=True)
st.header("Analysis")
st.markdown("<p class='section-caption'>Use the focused modules below after reviewing setup, overview, quality, and variables.</p>", unsafe_allow_html=True)
analysis_page = st.selectbox(
    "Analysis module",
    [
        "Data catalog",
        "Indicator sample",
        "Describe variables",
        "Within & between",
        "By group",
        "Composition",
        "Relationships",
        "Dynamics",
        "Kuznets-waves curve",
        "GDP per capita deep dive",
    ],
)

if analysis_page == "Data catalog":
    data_catalog()
elif analysis_page == "Indicator sample":
    indicator_sample()
elif analysis_page == "Describe variables":
    describe_variables(df)
elif analysis_page == "Within & between":
    within_between(df, panel)
elif analysis_page == "By group":
    by_group(df)
elif analysis_page == "Composition":
    composition(df)
elif analysis_page == "Relationships":
    relationships(df)
elif analysis_page == "Dynamics":
    dynamics(df, panel)
elif analysis_page == "Kuznets-waves curve":
    kuznets_waves()
elif analysis_page == "GDP per capita deep dive":
    gdp_deep_dive()
