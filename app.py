

import streamlit as st
import pandas as pd
import numpy as np
import io
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import textwrap
import time
import re
import json
import math
from typing import Dict, Any, List, Optional

st.set_page_config(page_title="Auto Data Explorer & Dashboard", layout="wide")

# Global simple style
st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; }
    .dashboard-card { background: #0e1117; border: 1px solid #2a2a2a; padding: 16px; border-radius: 12px; }
    .kpi-card { background: linear-gradient(135deg, rgba(59,130,246,.15), rgba(16,185,129,.15)); border: 1px solid rgba(148,163,184,.25); padding: 16px; border-radius: 12px; }
    .kpi-title { font-size: .9rem; color: #94a3b8; margin-bottom: .25rem; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #e2e8f0; }
    .muted { color: #94a3b8; font-size: .85rem; }
    .section-title { font-weight: 700; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """Robust CSV loader: tries common encodings and fallback to utf-8 with errors='replace'."""
    try:
        return pd.read_csv(file)
    except Exception:
        # try utf-8 and latin1
        try:
            return pd.read_csv(file, encoding='latin1')
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding='utf-8', engine='python', on_bad_lines='skip')

@st.cache_data
def get_basic_profile(df: pd.DataFrame):
    desc = df.describe(include='all').T
    missing = df.isnull().sum()
    types = df.dtypes
    unique = df.nunique(dropna=False)
    profile = pd.concat([types.rename('dtype'), unique.rename('n_unique'), missing.rename('n_missing'), desc], axis=1)
    return profile

@st.cache_data
def impute_dataframe(df: pd.DataFrame, strategy_num='mean', strategy_cat='most_frequent') -> pd.DataFrame:
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df2.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy=strategy_num)
        df2[num_cols] = imp_num.fit_transform(df2[num_cols])
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy=strategy_cat, fill_value='')
        df2[cat_cols] = imp_cat.fit_transform(df2[cat_cols])
    return df2

@st.cache_data
def automatic_encoding(df: pd.DataFrame, max_onehot=10):
    """One-hot encode low-cardinality cat cols, ordinal-encode high-cardinality ones.
    Returns encoded dataframe and a mapping of which columns were transformed.
    """
    df2 = df.copy()
    cat_cols = df2.select_dtypes(exclude=[np.number]).columns.tolist()
    mapping = {}
    for c in cat_cols:
        nuniq = df2[c].nunique(dropna=False)
        if nuniq <= max_onehot:
            # safe one-hot
            dummies = pd.get_dummies(df2[c].astype(str), prefix=c)
            df2 = pd.concat([df2.drop(columns=[c]), dummies], axis=1)
            mapping[c] = {'type': 'onehot', 'new_cols': dummies.columns.tolist()}
        else:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            reshaped = df2[[c]].astype(str)
            df2[c] = enc.fit_transform(reshaped).astype(int)
            mapping[c] = {'type': 'ordinal'}
    return df2, mapping

@st.cache_data
def scale_numeric(df: pd.DataFrame, method='standard'):
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        return df2
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df2[num_cols] = scaler.fit_transform(df2[num_cols])
    return df2

@st.cache_data
def detect_task(df: pd.DataFrame, target_col: str):
    if target_col is None or target_col == '':
        return None
    s = df[target_col]
    # numeric
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique(dropna=True) > 20:
            return 'regression'
        else:
            return 'classification'
    else:
        return 'classification'

@st.cache_data
def quick_modeling(X: pd.DataFrame, y: pd.Series, task='classification'):
    results = {}
    # prepare X numeric only (models here expect numeric input)
    X = X.copy()
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return {'error': 'no numeric features available for modeling after encoding/scaling.'}

    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.25, random_state=42)

    if task == 'classification':
        # ensure label-encoded y
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train.astype(str))
        y_test_enc = le.transform(y_test.astype(str))
        # logistic baseline (works only for binary)
        try:
            if len(np.unique(y_train_enc)) == 2:
                lr = LogisticRegression(max_iter=2000)
                lr.fit(X_train, y_train_enc)
                preds = lr.predict(X_test)
                proba = lr.predict_proba(X_test)[:, 1]
                results['LogisticRegression'] = {
                    'accuracy': float(accuracy_score(y_test_enc, preds)),
                    'roc_auc': float(roc_auc_score(y_test_enc, proba))
                }
            else:
                lr = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
                lr.fit(X_train, y_train_enc)
                preds = lr.predict(X_test)
                results['LogisticRegression'] = {
                    'accuracy': float(accuracy_score(y_test_enc, preds)),
                    'roc_auc': None
                }
        except Exception as e:
            results['LogisticRegression'] = {'error': str(e)}

        # random forest
        try:
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_train_enc)
            preds = rf.predict(X_test)
            rf_report = {'accuracy': float(accuracy_score(y_test_enc, preds))}
            if len(np.unique(y_test_enc)) == 2:
                try:
                    proba = rf.predict_proba(X_test)[:, 1]
                    rf_report['roc_auc'] = float(roc_auc_score(y_test_enc, proba))
                except Exception:
                    rf_report['roc_auc'] = None
            else:
                rf_report['roc_auc'] = None
            # feature importances
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            rf_report['top_features'] = importances.head(10).to_dict()
            results['RandomForest'] = rf_report
        except Exception as e:
            results['RandomForest'] = {'error': str(e)}

    else:
        # regression baseline - use numeric y
        try:
            y_train_num = pd.to_numeric(y_train, errors='coerce')
            y_test_num = pd.to_numeric(y_test, errors='coerce')
            lr = LinearRegression()
            lr.fit(X_train, y_train_num)
            preds = lr.predict(X_test)
            results['LinearRegression'] = {
                'rmse': float(mean_squared_error(y_test_num, preds, squared=False)),
                'r2': float(r2_score(y_test_num, preds))
            }
        except Exception as e:
            results['LinearRegression'] = {'error': str(e)}

        try:
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train_num)
            preds = rf.predict(X_test)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            results['RandomForest'] = {
                'rmse': float(mean_squared_error(y_test_num, preds, squared=False)),
                'r2': float(r2_score(y_test_num, preds)),
                'top_features': importances.head(10).to_dict()
            }
        except Exception as e:
            results['RandomForest'] = {'error': str(e)}

    return results

# ---- Dashboard helpers ----
AGG_FUNCS: Dict[str, Any] = {
    'SUM': lambda s: pd.to_numeric(s, errors='coerce').sum(skipna=True),
    'AVERAGE': lambda s: pd.to_numeric(s, errors='coerce').mean(skipna=True),
    'AVG': lambda s: pd.to_numeric(s, errors='coerce').mean(skipna=True),
    'COUNT': lambda s: s.shape[0],
    'DISTINCTCOUNT': lambda s: s.nunique(dropna=True),
    'MIN': lambda s: pd.to_numeric(s, errors='coerce').min(skipna=True),
    'MAX': lambda s: pd.to_numeric(s, errors='coerce').max(skipna=True),
    'MEDIAN': lambda s: pd.to_numeric(s, errors='coerce').median(skipna=True),
    'STD': lambda s: pd.to_numeric(s, errors='coerce').std(skipna=True),
    'VAR': lambda s: pd.to_numeric(s, errors='coerce').var(skipna=True),
}

def evaluate_aggregation(df: pd.DataFrame, func_name: str, column: str) -> float:
    fn = func_name.upper()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in dataframe")
    if fn not in AGG_FUNCS:
        raise ValueError(f"Aggregation '{func_name}' not supported")
    return float(AGG_FUNCS[fn](df[column]))

def evaluate_measure_formula(formula: str, df: pd.DataFrame) -> float:
    """Evaluate a very small DAX-like expression such as: SUM(Sales) / DISTINCTCOUNT(OrderID)"""
    if not isinstance(formula, str) or formula.strip() == '':
        raise ValueError("Empty measure expression")
    expr = formula
    # Replace function calls with their numeric values
    pattern = re.compile(r"([A-Za-z_]+)\(\s*([^\)]+?)\s*\)")
    idx = 0
    while True:
        match = pattern.search(expr, idx)
        if not match:
            break
        func = match.group(1)
        col = match.group(2)
        value = evaluate_aggregation(df, func, col)
        expr = expr[:match.start()] + str(value) + expr[match.end():]
        idx = 0  # restart to handle cascading replacements
    # Ensure only numbers and safe operators remain
    if re.search(r"[^0-9eE+\-*/(). ]", expr):
        raise ValueError("Unsupported tokens in expression after expansion")
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")

def build_basic_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str],
    y_col: Optional[str],
    agg: str = 'SUM',
    color: Optional[str] = None,
    size: Optional[str] = None,
    facet_col: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_desc: bool = True,
    cumulative: bool = False,
    trendline: Optional[str] = None,
):
    chart_type = chart_type.lower()
    dd = df.copy()

    # If aggregation needed per x_col
    if x_col and y_col and chart_type in {'bar', 'column', 'line', 'area'}:
        if x_col not in dd.columns or y_col not in dd.columns:
            raise ValueError("Selected columns not in data")
        grouped = dd.groupby(x_col)[y_col].apply(lambda s: AGG_FUNCS[agg.upper()](s)).reset_index(name=y_col)
        if top_n:
            grouped = grouped.sort_values(by=y_col, ascending=not sort_desc).head(top_n)
        if cumulative:
            grouped[y_col] = grouped[y_col].cumsum()
        dd = grouped

    fig = None
    if chart_type in {'bar', 'column'}:
        orientation = 'h' if chart_type == 'bar' else 'v'
        fig = px.bar(dd, x=x_col if orientation=='v' else y_col, y=y_col if orientation=='v' else x_col, color=color)
    elif chart_type == 'line':
        fig = px.line(dd, x=x_col, y=y_col, color=color)
    elif chart_type == 'area':
        fig = px.area(dd, x=x_col, y=y_col, color=color)
    elif chart_type == 'scatter':
        fig = px.scatter(dd, x=x_col, y=y_col, color=color, size=size, trendline=trendline)
    elif chart_type == 'pie':
        if not x_col or not y_col:
            raise ValueError("Pie requires label (x) and value (y)")
        fig = px.pie(dd, names=x_col, values=y_col, color=color, hole=0.0)
    elif chart_type == 'donut':
        if not x_col or not y_col:
            raise ValueError("Donut requires label (x) and value (y)")
        fig = px.pie(dd, names=x_col, values=y_col, color=color, hole=0.5)
    elif chart_type == 'box':
        fig = px.box(dd, x=x_col, y=y_col, color=color)
    elif chart_type == 'violin':
        fig = px.violin(dd, x=x_col, y=y_col, color=color, box=True, points='outliers')
    elif chart_type == 'histogram':
        fig = px.histogram(dd, x=x_col or y_col, color=color)
    elif chart_type == 'treemap':
        if not x_col or not y_col:
            raise ValueError("Treemap requires path (x) and value (y)")
        fig = px.treemap(dd, path=[x_col], values=y_col)
    elif chart_type == 'sunburst':
        if not x_col or not y_col:
            raise ValueError("Sunburst requires path (x) and value (y)")
        fig = px.sunburst(dd, path=[x_col], values=y_col)
    else:
        raise ValueError("Unsupported chart type")

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ------------------ UI ------------------
st.title("Auto Data Explorer & Dashboard")
st.markdown("Upload a CSV and get automatic preprocessing, EDA, visualizations, a modeling sandbox, and a customizable dashboard.\n\n")

with st.sidebar:
    st.header("Settings")
    show_profile = st.checkbox("Show data profile (summary)", value=True)
    impute_num = st.selectbox("Numeric imputation", options=['mean','median','most_frequent'], index=0)
    impute_cat = st.selectbox("Categorical imputation", options=['most_frequent','constant'], index=0)
    scale_method = st.selectbox("Scale numeric columns", options=['none','standard','minmax'], index=0)
    onehot_max = st.slider("Max unique values to one-hot encode", 2, 200, 10)
    var_threshold = st.slider("Remove low-variance features (threshold)", 0.0, 0.5, 0.0, 0.01)
    sample_n = st.number_input("Max rows to display in tables/plots (sampling)", min_value=100, max_value=1000000, value=5000, step=100)
    st.markdown("---")
    st.write("App tips:")
    st.write(textwrap.dedent("""  
    • For very large files (>50 MB) prefer preprocessing offline or increase server resources.
    • This app provides a quick automated flow — tune and validate each step before production.
    """))

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Prepare pipelines up-front so all tabs can access
    df_proc = impute_dataframe(df, strategy_num=impute_num, strategy_cat=impute_cat)
    df_enc, enc_map = automatic_encoding(df_proc, max_onehot=onehot_max)
    if scale_method != 'none':
        df_scaled = scale_numeric(df_enc, method=scale_method)
    else:
        df_scaled = df_enc

    # Variance threshold (numeric only)
    if var_threshold > 0:
        sel = VarianceThreshold(threshold=var_threshold)
        num_cols_v = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_v) > 0:
            try:
                arr = sel.fit_transform(df_scaled[num_cols_v])
                kept = [c for (c, keep) in zip(num_cols_v, sel.get_support()) if keep]
                dropped = [c for c in num_cols_v if c not in kept]
                df_scaled = df_scaled.drop(columns=dropped)
            except Exception:
                pass

    st.session_state['data_variants'] = {
        'Raw': df,
        'Imputed': df_proc,
        'Encoded': df_enc,
        'Scaled': df_scaled,
    }

    tab_overview, tab_preprocess, tab_explore, tab_model, tab_dashboard, tab_export = st.tabs([
        "Overview", "Preprocess", "Explore", "Modeling", "Dashboard", "Export"
    ])

    with tab_overview:
        st.subheader("File information")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        try:
            uploaded_file.seek(0)
            size_kb = len(uploaded_file.getvalue())/1024
            st.write(f"File size: {size_kb:.1f} KB")
        except Exception:
            pass

        st.subheader("Raw data preview (sample)")
        st.dataframe(df.head(100))

        if show_profile:
            st.subheader("Data profile")
            profile = get_basic_profile(df)
            st.dataframe(profile)

        st.subheader("Quick cleaning actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Drop duplicate rows"):
                before = len(df)
                df = df.drop_duplicates()
                st.success(f"Dropped {before - len(df)} duplicate rows")
        with c2:
            if st.button("Drop rows with >50% missing"):
                thresh = int(df.shape[1] * 0.5)
                before = len(df)
                df = df.dropna(thresh=thresh)
                st.success(f"Dropped {before - len(df)} rows")
        with c3:
            if st.button("Reset to uploaded"):
                uploaded_file.seek(0)
                df = load_csv(uploaded_file)
                st.success("Reset done")

    with tab_preprocess:
        st.subheader("Automated preprocessing pipeline")
        with st.expander("1) Impute missing values", expanded=True):
            st.write("Numeric strategy:", impute_num)
            st.write("Categorical strategy:", impute_cat)
            st.write("Missing values after imputation:")
            st.dataframe(df_proc.isnull().sum())
        with st.expander("2) Encoding categorical variables", expanded=True):
            st.write(f"One-hot when unique values <= {onehot_max}, else ordinal encoding")
            st.write("Columns after encoding (sample):")
            st.dataframe(pd.DataFrame({'columns': df_enc.columns}).head(200))
        with st.expander("3) Scaling numeric features", expanded=False):
            if scale_method != 'none':
                st.write("Scaled numeric preview")
                st.dataframe(df_scaled.select_dtypes(include=[np.number]).head())
            else:
                st.write("No scaling applied")
        with st.expander("4) Remove low-variance features", expanded=False):
            if var_threshold > 0:
                st.write("Low-variance features removed where applicable.")
            else:
                st.write("No variance threshold applied")

    with tab_explore:
        st.subheader("Exploratory Data Analysis (auto)")
        left, right = st.columns((2,1))
        sample_n_eff = int(min(sample_n, len(df_scaled)))
        df_for_vis = df_scaled.sample(sample_n_eff, random_state=42) if len(df_scaled) > sample_n_eff else df_scaled

        with left:
            st.write("### Distributions")
            numeric_cols = df_for_vis.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df_for_vis.columns if c not in numeric_cols]
            if len(numeric_cols) > 0:
                sel_num = st.selectbox("Choose numeric column for histogram", options=numeric_cols, key='hist')
                fig = px.histogram(df_for_vis, x=sel_num, nbins=40, marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            if len(cat_cols) > 0:
                sel_cat = st.selectbox("Choose categorical column for bar chart", options=cat_cols, key='bar')
                vc = df_for_vis[sel_cat].astype(str).value_counts().reset_index()
                vc.columns = [sel_cat, 'count']
                fig2 = px.bar(vc, x=sel_cat, y='count')
                st.plotly_chart(fig2, use_container_width=True)

        with right:
            st.write("### Correlation (numeric)")
            if len(df_for_vis.select_dtypes(include=[np.number]).columns) > 1:
                corr = df_for_vis.select_dtypes(include=[np.number]).corr()
                fig3 = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write("Not enough numeric columns for correlation")

        st.subheader("Interactive row-level exploration")
        max_show = min(500, len(df_scaled))
        st.dataframe(df_scaled.sample(max_show, random_state=42))

    with tab_model:
        st.subheader("Modeling sandbox (optional)")
        target = st.selectbox("Select target column (optional)", options=[''] + df.columns.tolist())
        if target:
            task = detect_task(df_proc, target)
            st.write("Detected task:", task)
            try:
                encoded_target_cols = [c for c in df_scaled.columns if c.startswith(f"{target}_")]
                if len(encoded_target_cols) > 0:
                    st.write("Detected multiple encoded target columns (one-hot). Please pick one target column from the encoded options.")
                    chosen = st.selectbox("Choose encoded target", options=encoded_target_cols)
                    y = df_scaled[chosen]
                    X = df_scaled.drop(columns=[chosen])
                else:
                    if target in df_scaled.columns:
                        y = df_scaled[target]
                        X = df_scaled.drop(columns=[target])
                    else:
                        y = df_proc[target]
                        drop_cols = [c for c in df_scaled.columns if c.startswith(target + '_')]
                        if drop_cols:
                            X = df_scaled.drop(columns=drop_cols)
                        else:
                            X = df_scaled.drop(columns=[c for c in df_scaled.columns if c == target], errors='ignore')

                st.write(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

                if st.button("Run quick baseline models"):
                    with st.spinner("Training models..."):
                        try:
                            results = quick_modeling(X, y, task=task)
                            st.write(results)
                            if 'RandomForest' in results and isinstance(results['RandomForest'], dict):
                                rf = results['RandomForest']
                                if 'top_features' in rf and isinstance(rf['top_features'], dict):
                                    feat = pd.Series(rf['top_features']).sort_values(ascending=True)
                                    fig = go.Figure(go.Bar(x=feat.values, y=feat.index, orientation='h'))
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Modeling failed: {e}")
            except Exception as e:
                st.error(f"Could not prepare features/target for modeling: {e}")

    with tab_dashboard:
        st.subheader("Dashboard Builder")
        # Choose data variant
        data_variant = st.selectbox("Choose dataset for dashboard", options=list(st.session_state['data_variants'].keys()), index=0)
        df_dash = st.session_state['data_variants'][data_variant]

        # Filters
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        if 'dashboard_filters' not in st.session_state:
            st.session_state['dashboard_filters'] = []
        add_filter = st.button("Add filter")
        if add_filter:
            st.session_state['dashboard_filters'].append({"column": None, "op": "==", "values": None})
        new_filters = []
        for idx_f, fcfg in enumerate(st.session_state['dashboard_filters']):
            st.write(f"Filter {idx_f+1}")
            cols = st.columns((2,2,6,1))
            with cols[0]:
                col = st.selectbox("Column", options=[None] + df_dash.columns.tolist(), key=f"fcol_{idx_f}")
            with cols[1]:
                op = st.selectbox("Op", options=['==','!=','in','not in','>','>=','<','<='], index=0, key=f"fop_{idx_f}")
            with cols[2]:
                val_widget = None
                if col is not None:
                    if pd.api.types.is_numeric_dtype(df_dash[col]):
                        if op in ['in','not in']:
                            unique_vals = sorted(df_dash[col].dropna().unique().tolist())[:2000]
                            val_widget = st.multiselect("Values", options=unique_vals, key=f"fvals_{idx_f}")
                        else:
                            rng = st.slider("Value", float(np.nanmin(pd.to_numeric(df_dash[col], errors='coerce'))), float(np.nanmax(pd.to_numeric(df_dash[col], errors='coerce'))), value=(float(np.nanmin(pd.to_numeric(df_dash[col], errors='coerce'))), float(np.nanmax(pd.to_numeric(df_dash[col], errors='coerce')))), key=f"frng_{idx_f}")
                            val_widget = rng
                    else:
                        unique_vals = sorted(df_dash[col].astype(str).dropna().unique().tolist())[:2000]
                        if op in ['in','not in']:
                            val_widget = st.multiselect("Values", options=unique_vals, key=f"fvals_{idx_f}")
                        else:
                            val_widget = st.selectbox("Value", options=[None] + unique_vals, key=f"fval_{idx_f}")
                else:
                    st.write("Select a column")
            with cols[3]:
                remove = st.button("✖", key=f"frem_{idx_f}")
            if remove:
                continue
            new_filters.append({"column": col, "op": op, "values": val_widget})
        st.session_state['dashboard_filters'] = new_filters

        # Apply filters
        dff = df_dash.copy()
        try:
            for f in st.session_state['dashboard_filters']:
                col = f.get('column')
                op = f.get('op')
                val = f.get('values')
                if not col:
                    continue
                if op == '==':
                    dff = dff[dff[col].astype(str) == str(val)]
                elif op == '!=':
                    dff = dff[dff[col].astype(str) != str(val)]
                elif op == 'in' and isinstance(val, list):
                    dff = dff[dff[col].isin(val)]
                elif op == 'not in' and isinstance(val, list):
                    dff = dff[~dff[col].isin(val)]
                elif op in ['>','>=','<','<='] and isinstance(val, tuple):
                    lo, hi = val
                    if op in ['>','>=']:
                        dff = dff[dff[col] >= lo] if op == '>=' else dff[dff[col] > lo]
                    if op in ['<','<=']:
                        dff = dff[dff[col] <= hi] if op == '<=' else dff[dff[col] < hi]
        except Exception:
            st.info("Some filters could not be applied due to data types.")

        # KPI Cards
        st.markdown("<div class='section-title'>KPI / Measures (DAX-like)</div>", unsafe_allow_html=True)
        if 'kpi_cards' not in st.session_state:
            st.session_state['kpi_cards'] = []
        kpi_cols = st.columns((3,3,2,2,2))
        with kpi_cols[0]:
            kpi_title = st.text_input("Card title", value="New KPI")
        with kpi_cols[1]:
            measure_expr = st.text_input("Measure expression (e.g., SUM(Sales)/DISTINCTCOUNT(Orders))", value="")
        with kpi_cols[2]:
            quick_func = st.selectbox("Quick agg", options=['','SUM','AVERAGE','COUNT','DISTINCTCOUNT','MIN','MAX','MEDIAN'], index=0)
        with kpi_cols[3]:
            quick_col = st.selectbox("Column", options=[''] + dff.columns.tolist(), index=0)
        with kpi_cols[4]:
            add_kpi = st.button("Add KPI")
        if add_kpi:
            try:
                val = None
                if measure_expr:
                    val = evaluate_measure_formula(measure_expr, dff)
                    expr_used = measure_expr
                elif quick_func and quick_col:
                    val = evaluate_aggregation(dff, quick_func, quick_col)
                    expr_used = f"{quick_func}({quick_col})"
                else:
                    st.warning("Provide a measure expression or quick aggregation + column.")
                    val = None
                if val is not None:
                    st.session_state['kpi_cards'].append({"title": kpi_title, "expr": expr_used, "value": float(val)})
                    st.success("KPI added")
            except Exception as e:
                st.error(f"KPI error: {e}")

        # Render KPI cards
        if st.session_state['kpi_cards']:
            cols = st.columns(4)
            for i, card in enumerate(st.session_state['kpi_cards']):
                with cols[i % 4]:
                    st.markdown("<div class='kpi-card'>" \
                        + f"<div class='kpi-title'>{card['title']}</div>" \
                        + f"<div class='kpi-value'>{card['value']:.4g}</div>" \
                        + f"<div class='muted'>{card['expr']}</div>" \
                        + "</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='section-title'>Chart Builder</div>", unsafe_allow_html=True)
        if 'charts' not in st.session_state:
            st.session_state['charts'] = []
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_type = st.selectbox("Type", options=['Column','Bar','Line','Area','Scatter','Pie','Donut','Box','Violin','Histogram','Treemap','Sunburst'])
            chart_type_key = chart_type.lower()
        with c2:
            x_col = st.selectbox("X", options=[None] + dff.columns.tolist())
            color = st.selectbox("Color", options=[None] + dff.columns.tolist())
        with c3:
            y_col = st.selectbox("Y (value)", options=[None] + dff.columns.tolist())
            agg = st.selectbox("Aggregate", options=['SUM','AVERAGE','COUNT','DISTINCTCOUNT','MIN','MAX','MEDIAN'])
        c4, c5, c6 = st.columns(3)
        with c4:
            top_n = st.number_input("Top N (optional)", min_value=0, max_value=1000, value=0, step=1)
        with c5:
            sort_desc = st.checkbox("Sort desc", value=True)
        with c6:
            cumulative = st.checkbox("Cumulative", value=False)
        add_chart = st.button("Add chart")
        if add_chart:
            try:
                fig = build_basic_chart(dff, chart_type_key, x_col, y_col, agg=agg, color=color, top_n=int(top_n) if top_n else None, sort_desc=sort_desc, cumulative=cumulative)
                st.session_state['charts'].append({
                    'type': chart_type_key,
                    'x': x_col,
                    'y': y_col,
                    'agg': agg,
                    'color': color,
                    'top_n': int(top_n) if top_n else None,
                    'sort_desc': sort_desc,
                    'cumulative': cumulative,
                })
                st.success("Chart added")
            except Exception as e:
                st.error(f"Chart error: {e}")

        # Render charts grid
        if st.session_state['charts']:
            for cfg in st.session_state['charts']:
                try:
                    fig = build_basic_chart(dff, cfg['type'], cfg['x'], cfg['y'], agg=cfg['agg'], color=cfg['color'], top_n=cfg['top_n'], sort_desc=cfg['sort_desc'], cumulative=cfg['cumulative'])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not render chart: {e}")

        st.markdown("---")
        st.write("Save/Load dashboard configuration")
        if st.button("Clear dashboard"):
            st.session_state['charts'] = []
            st.session_state['kpi_cards'] = []
            st.session_state['dashboard_filters'] = []
        cfg = {
            'dataset': data_variant,
            'filters': st.session_state['dashboard_filters'],
            'kpi_cards': st.session_state['kpi_cards'],
            'charts': st.session_state['charts'],
        }
        st.download_button("Download config JSON", data=json.dumps(cfg, indent=2), file_name='dashboard_config.json', mime='application/json')
        uploaded_cfg = st.file_uploader("Upload config JSON", type=['json'], key='cfg_upload')
        if uploaded_cfg is not None:
            try:
                loaded = json.load(uploaded_cfg)
                st.session_state['dashboard_filters'] = loaded.get('filters', [])
                st.session_state['kpi_cards'] = loaded.get('kpi_cards', [])
                st.session_state['charts'] = loaded.get('charts', [])
                st.success("Config loaded")
            except Exception as e:
                st.error(f"Failed to load config: {e}")

    with tab_export:
        st.subheader("Export cleaned dataset")
        to_export = df_scaled.copy()
        csv = to_export.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download cleaned CSV", data=csv, file_name='cleaned_data.csv', mime='text/csv')
        try:
            parquet = to_export.to_parquet(index=False)
            st.download_button(label="Download cleaned Parquet", data=parquet, file_name='cleaned_data.parquet', mime='application/octet-stream')
        except Exception:
            pass
        try:
            to_excel = BytesIO()
            to_export.to_excel(to_excel, index=False)
            st.download_button(label="Download cleaned Excel", data=to_excel.getvalue(), file_name='cleaned_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except Exception:
            pass
        st.markdown("---")
        st.write("Further tips: 1) For extremely large datasets consider using sampling + chunked processing. 2) Check encodings and categorical mappings before trusting automated encodings.")

else:
    st.info("Upload a CSV to get started. The app will preview data, run an automatic preprocessing pipeline and build visualizations, plus a custom dashboard builder.")

# END