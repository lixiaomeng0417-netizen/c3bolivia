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
        return df

    add_cols = [c for c in ID_COLS + ["capital", "n_mun"] if c in regions.columns and c not in df.columns]
    if not add_cols:
        return df
    return df.merge(regions[["prov_id"] + add_cols], on="prov_id", how="left")


def dataset_path(filename: str) -> Path:
    return DATA_DIR / filename


def geojson_path() -> Path:
    return DATA_DIR / GEOJSON_FILE


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
        return df[keep].copy()

    gdppc_cols = [c for c in df.columns if re.fullmatch(r"gdppc\d{4}", c)]
    if gdppc_cols:
        id_cols = [c for c in ["prov_id", "prov", "dep", "dep_id"] if c in df.columns]
        long_df = df.melt(id_vars=id_cols, value_vars=gdppc_cols, var_name="indicator", value_name="gdppc")
        long_df["year"] = long_df["indicator"].str.extract(r"(\d{4})").astype(int)
        long_df["log_gdppc"] = np.log(long_df["gdppc"].where(long_df["gdppc"] > 0))
        return long_df.drop(columns=["indicator"])

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
    return pd.concat(frames, ignore_index=True)


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
def load_gdp_panel() -> pd.DataFrame:
    path = dataset_path("gdp_perCapita_long.csv")
    if path.exists():
        return pd.read_csv(path)
    wide_path = dataset_path("gdp_perCapita_1990_2024.csv")
    if not wide_path.exists():
        return pd.DataFrame()
    panel = make_panel(pd.read_csv(wide_path))
    return panel if panel is not None else pd.DataFrame()


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
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows kept", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric variables", f"{len(numeric_cols(df)):,}")


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
        dtype_summary = original.dtypes.astype(str).value_counts().reset_index()
        dtype_summary.columns = ["dtype", "columns"]
        fig = px.bar(dtype_summary, x="dtype", y="columns", title="Column types")
        st.plotly_chart(fig, width="stretch")

    if panel is not None:
        st.subheader("Value heatmap")
        panel_for_heatmap = filter_panel_metric(panel, "overview_metric")
        value_options = panel_value_options(panel_for_heatmap)
        value = st.selectbox("Variable", value_options, format_func=friendly)
        index_col = "dep" if "dep" in panel_for_heatmap.columns else "prov"
        pivot = panel_for_heatmap.pivot_table(index=index_col, columns="year", values=value, aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", title=f"{friendly(value)} by {index_col} and year")
        st.plotly_chart(fig, width="stretch")

    if geojson_path().exists() and "prov_id" in df.columns and numeric_cols(df):
        with st.expander("Province map preview"):
            geo = json.loads(geojson_path().read_text())
            color_col = st.selectbox("Map color", numeric_cols(df), index=0, format_func=friendly)
            fig = px.choropleth(
                df,
                geojson=geo,
                locations="prov_id",
                featureidkey="properties.prov_id",
                color=color_col,
                hover_name="prov" if "prov" in df.columns else None,
                color_continuous_scale="Viridis",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(title=f"{friendly(color_col)} by province", margin=dict(l=0, r=0, t=40, b=0))
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


def trends(df: pd.DataFrame, panel: pd.DataFrame | None):
    st.header("Trends")
    if panel is None:
        st.info("Trend view is available for datasets with a year column or annual metric columns.")
        return
    panel_for_view = filter_panel_metric(panel, "trends_metric")
    value = st.selectbox("Trend variable", panel_value_options(panel_for_view), format_func=friendly)
    by = st.radio("Aggregate by", ["Department mean", "Selected provinces"], horizontal=True)
    if by == "Department mean" and "dep" in panel_for_view.columns:
        trend = panel_for_view.groupby(["year", "dep"], as_index=False)[value].mean()
        fig = px.line(trend, x="year", y=value, color="dep", markers=True, title=f"{friendly(value)} by department")
    else:
        provinces = sorted(panel_for_view["prov"].dropna().unique()) if "prov" in panel_for_view.columns else []
        default = provinces[:6]
        chosen = st.multiselect("Provinces", provinces, default=default)
        trend = panel_for_view[panel_for_view["prov"].isin(chosen)] if chosen else panel_for_view.head(0)
        fig = px.line(trend, x="year", y=value, color="prov", markers=True, title=f"{friendly(value)} for selected provinces")
    st.plotly_chart(fig, width="stretch")

    if {"prov", "year", value}.issubset(panel_for_view.columns):
        latest_year = int(panel_for_view["year"].max())
        latest = panel_for_view[panel_for_view["year"] == latest_year].nlargest(15, value)
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
    if panel is not None and {"year", "prov"}.issubset(panel.columns):
        panel_for_view = filter_panel_metric(panel, "dynamics_metric")
        value = st.selectbox("Dynamic variable", panel_value_options(panel_for_view), format_func=friendly)
        years = sorted(panel_for_view["year"].dropna().astype(int).unique())
        start, end = st.select_slider("Compare years", options=years, value=(years[0], years[-1]))
        wide = panel_for_view[panel_for_view["year"].isin([start, end])].pivot_table(index=["prov", "dep"], columns="year", values=value).reset_index()
        wide = wide.dropna(subset=[start, end])
        wide["change"] = wide[end] - wide[start]
        wide["pct_change"] = np.where(wide[start] != 0, wide["change"] / wide[start] * 100, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.scatter(wide, x=start, y=end, color="dep", hover_name="prov", title=f"{friendly(value)}: {start} vs {end}"), width="stretch")
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


def gini(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or np.isclose(x.sum(), 0):
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def gdp_deep_dive():
    st.header("GDP Per Capita Deep Dive")
    gdp = load_gdp_panel()
    if gdp.empty:
        st.info("GDP per capita files were not found in the local c3databolivia folder.")
        return

    if st.session_state.get("filter_dep") and "dep" in gdp.columns:
        gdp = gdp[gdp["dep"].isin(st.session_state["filter_dep"])]
    if st.session_state.get("filter_prov") and "prov" in gdp.columns:
        gdp = gdp[gdp["prov"].isin(st.session_state["filter_prov"])]

    years = sorted(gdp["year"].dropna().astype(int).unique())
    start, end = st.select_slider("GDP analysis period", options=years, value=(years[0], years[-1]))
    gdp = gdp[(gdp["year"] >= start) & (gdp["year"] <= end)].copy()
    gdp["gdppc"] = pd.to_numeric(gdp["gdppc"], errors="coerce")
    gdp["log_gdppc"] = np.log(gdp["gdppc"].where(gdp["gdppc"] > 0))

    latest_year = int(gdp["year"].max())
    first_year = int(gdp["year"].min())
    latest = gdp[gdp["year"] == latest_year].copy()
    first = gdp[gdp["year"] == first_year].copy()
    merged = first[["prov_id", "prov", "dep", "gdppc"]].merge(
        latest[["prov_id", "gdppc"]],
        on="prov_id",
        suffixes=("_start", "_end"),
    )
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
            y="prov",
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
            y="prov",
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
        index=["prov_id", "prov", "dep"], columns="year", values="rank"
    ).reset_index()
    rank_wide["rank_change"] = rank_wide[first_year] - rank_wide[latest_year]
    movers = rank_wide.reindex(rank_wide["rank_change"].abs().sort_values(ascending=False).index).head(top_n)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(movers, x="rank_change", y="prov", color="dep", orientation="h", title="Largest rank moves")
        st.plotly_chart(fig, width="stretch")
    with c2:
        chosen = st.multiselect(
            "Track province ranks",
            sorted(gdp["prov"].dropna().unique()),
            default=sorted(gdp["prov"].dropna().unique())[:6],
        )
        rank_lines = ranks[ranks["prov"].isin(chosen)] if chosen else ranks.head(0)
        fig = px.line(rank_lines, x="year", y="rank", color="prov", markers=True, title="GDPpc rank over time")
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
    conv = first[["prov_id", "prov", "dep", "gdppc"]].merge(
        merged[["prov_id", "cagr"]],
        on="prov_id",
    )
    fig = px.scatter(
        conv,
        x="gdppc",
        y="cagr",
        color="dep",
        hover_name="prov",
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
    st.title("ExPdPy")
    st.subheader("Data")
    st.info(
        "Private/local mode: this app reads CSV files from your local repository only. "
        "Do not deploy or push private data to a public hosting service.",
    )
    dataset_label = st.selectbox("Dataset", list(DEFAULT_DATASETS), index=0)
    active_name = DEFAULT_DATASETS[dataset_label]
    raw = load_dataset(active_name)

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

st.sidebar.subheader("Overview")
page = st.sidebar.radio(
    "Explore",
    [
        "Overview & Data",
        "Data catalog",
        "Describe variables",
        "Within & between",
        "Trends",
        "By group",
        "Composition",
        "Relationships",
        "Dynamics",
        "GDP per capita deep dive",
    ],
)

st.title("C3 Bolivia")
st.caption("Local-only exploration of province-level SDG, satellite, population, night lights, and GDP per capita data.")

if page == "Overview & Data":
    overview(df, raw, panel)
elif page == "Data catalog":
    data_catalog()
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
elif page == "GDP per capita deep dive":
    gdp_deep_dive()
