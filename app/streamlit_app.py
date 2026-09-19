import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from analytics.anomaly_detector import detect_all_anomalies
from analytics.executive_insights import generate_executive_insights

from analytics.marketing_ai import (
    is_marketing_question,
    generate_marketing_answer,
)

from analytics.marketing_analytics import (
    load_marketing_data,
    calculate_marketing_kpis,
    channel_performance,
    performance_type_summary,
    top_campaigns_by_roi,
    lowest_campaigns_by_roi,
)

from rag.rag_pipeline import (
    load_vector_store,
    load_embedding_model,
    ask_question,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightRAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1.2rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(49, 51, 63, 0.68);
        margin-top: -0.4rem;
        margin-bottom: 0.5rem;
    }

    .section-note {
        color: rgba(49, 51, 63, 0.62);
        font-size: 0.92rem;
        margin-top: -0.35rem;
    }

    .analysis-scope {
        padding: 0.65rem 0.9rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.16);
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
    }

    .answer-card {
        padding: 1.15rem 1.25rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .question-label {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .answer-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 1rem;
    }

    .source-card {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        margin-bottom: 0.5rem;
    }

    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_business_data():

    data_dir = BASE_DIR / "data"

    datasets = {}

    if not data_dir.exists():
        return datasets

    csv_files = list(data_dir.rglob("*.csv"))

    for file_path in csv_files:

        filename = file_path.stem.lower()

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if "master_analytics" in filename:
            datasets["master"] = df

        elif "order_item" in filename or "order-items" in filename:
            datasets["order_items"] = df

        elif "orders" in filename or filename == "order":
            datasets["orders"] = df

        elif "returns" in filename or "return" in filename:
            datasets["returns"] = df

        elif "products" in filename or "product" in filename:
            datasets["products"] = df

        elif "customers" in filename or "customer" in filename:
            datasets["customers"] = df

        elif "marketing" in filename:
            datasets["marketing"] = df

    return datasets


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(df, possible_names):

    if df is None:
        return None

    normalized = {
        str(col).lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }

    for name in possible_names:

        normalized_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized_name in normalized:
            return normalized[normalized_name]

    return None


# ============================================================
# KPI CALCULATION
# ============================================================

def calculate_kpis(datasets):

    sales = 0
    profit = 0
    orders = 0
    returns = 0

    order_items = datasets.get("order_items")

    if order_items is not None and not order_items.empty:

        sales_col = find_column(
            order_items,
            [
                "sales_amount",
                "sales",
                "revenue",
                "total_sales",
                "selling_price",
                "amount",
            ],
        )

        profit_col = find_column(
            order_items,
            [
                "profit",
                "total_profit",
                "net_profit",
            ],
        )

        if sales_col:

            sales = pd.to_numeric(
                order_items[sales_col],
                errors="coerce",
            ).fillna(0).sum()

        if profit_col:

            profit = pd.to_numeric(
                order_items[profit_col],
                errors="coerce",
            ).fillna(0).sum()

    orders_df = datasets.get("orders")

    if orders_df is not None and not orders_df.empty:

        order_id_col = find_column(
            orders_df,
            [
                "order_id",
                "orderid",
                "id",
            ],
        )

        if order_id_col:
            orders = orders_df[order_id_col].nunique()
        else:
            orders = len(orders_df)

    elif order_items is not None and not order_items.empty:

        order_id_col = find_column(
            order_items,
            [
                "order_id",
                "orderid",
            ],
        )

        if order_id_col:
            orders = order_items[order_id_col].nunique()

    returns_df = datasets.get("returns")

    if returns_df is not None and not returns_df.empty:

        return_id_col = find_column(
            returns_df,
            [
                "return_id",
                "returnid",
                "id",
            ],
        )

        if return_id_col:
            returns = returns_df[return_id_col].nunique()
        else:
            returns = len(returns_df)

    if sales > 0:
        margin = (profit / sales) * 100
    else:
        margin = 0

    if orders > 0:
        return_rate = (returns / orders) * 100
    else:
        return_rate = 0

    return {
        "sales": sales,
        "profit": profit,
        "margin": margin,
        "orders": orders,
        "returns": returns,
        "return_rate": return_rate,
    }


# ============================================================
# FILTERED KPI CALCULATION
# ============================================================

def calculate_filtered_kpis(master_df):
    if master_df is None or master_df.empty:
        return {
            "sales": 0,
            "profit": 0,
            "margin": 0,
            "orders": 0,
            "returns": 0,
            "return_rate": 0,
        }

    sales = pd.to_numeric(
        master_df.get("sales_amount", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()

    profit = pd.to_numeric(
        master_df.get("profit", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()

    orders = (
        master_df["order_id"].nunique()
        if "order_id" in master_df.columns
        else len(master_df)
    )

    if "is_returned" in master_df.columns and "order_id" in master_df.columns:
        return_flags = (
            master_df.assign(
                is_returned=pd.to_numeric(
                    master_df["is_returned"], errors="coerce"
                ).fillna(0)
            )
            .groupby("order_id")["is_returned"]
            .max()
        )
        returns = int((return_flags > 0).sum())
    else:
        returns = 0

    margin = (profit / sales) * 100 if sales > 0 else 0
    return_rate = (returns / orders) * 100 if orders > 0 else 0

    return {
        "sales": sales,
        "profit": profit,
        "margin": margin,
        "orders": orders,
        "returns": returns,
        "return_rate": return_rate,
    }


# ============================================================
# BUSINESS ALERTS
# ============================================================

def display_business_alerts():

    try:

        anomalies = detect_all_anomalies()

    except Exception as error:

        st.warning(
            f"Unable to load business alerts: {error}"
        )

        return []

    if not anomalies:

        st.success(
            "No significant business anomalies detected."
        )

        return []

    st.markdown("## 🚨 Business Alerts")

    st.caption(
        f"{len(anomalies)} potential business issue(s) "
        "detected automatically."
    )

    for anomaly in anomalies:

        severity = anomaly.get(
            "severity",
            "LOW"
        )

        anomaly_type = anomaly.get(
            "type",
            "Business Anomaly"
        )

        message = anomaly.get(
            "message",
            ""
        )

        action = anomaly.get(
            "action",
            ""
        )

        period = anomaly.get(
            "period",
            "Overall"
        )

        metric = anomaly.get(
            "metric",
            ""
        )

        value = anomaly.get(
            "value",
            None
        )

        # ====================================================
        # HIGH PRIORITY
        # ====================================================

        if severity == "HIGH":

            with st.container(border=True):

                st.markdown(
                    f"### 🔴 HIGH PRIORITY — {anomaly_type}"
                )

                st.write(
                    f"**Period:** {period}"
                )

                if metric and value is not None:

                    st.write(
                        f"**{metric}:** {value}"
                    )

                st.write(message)

                st.info(
                    f"**Recommended investigation:** "
                    f"{action}"
                )

        # ====================================================
        # MEDIUM PRIORITY
        # ====================================================

        elif severity == "MEDIUM":

            with st.container(border=True):

                st.markdown(
                    f"### 🟠 MEDIUM PRIORITY — {anomaly_type}"
                )

                st.write(
                    f"**Period:** {period}"
                )

                if metric and value is not None:

                    st.write(
                        f"**{metric}:** {value}"
                    )

                st.write(message)

                st.info(
                    f"**Recommended investigation:** "
                    f"{action}"
                )

        # ====================================================
        # LOW PRIORITY
        # ====================================================

        else:

            with st.container(border=True):

                st.markdown(
                    f"### 🟡 LOW PRIORITY — {anomaly_type}"
                )

                st.write(
                    f"**Period:** {period}"
                )

                if metric and value is not None:

                    st.write(
                        f"**{metric}:** {value}"
                    )

                st.write(message)

                st.info(
                    f"**Recommended investigation:** "
                    f"{action}"
                )

    # IMPORTANT:
    # Return AFTER the entire loop finishes.
    return anomalies


# ============================================================
# VISUAL INSIGHT ROUTER
# ============================================================

def show_visual_insight(question, master_df, marketing_df=None):

    question_lower = question.lower()

    # ========================================================
    # MARKETING QUESTIONS
    # ========================================================

    if (
        marketing_df is not None
        and not marketing_df.empty
        and (
            "marketing" in question_lower
            or "campaign" in question_lower
            or "roi" in question_lower
            or "advertising" in question_lower
        )
    ):

        st.markdown("## 📣 Marketing Visual Insight")

        st.caption(
            "A relevant visualization selected from the marketing analytics dataset."
        )

        # ----------------------------------------------------
        # POOR / LOW-PERFORMING CAMPAIGNS
        # ----------------------------------------------------

        if (
            "poor" in question_lower
            or "poorly" in question_lower
            or "lowest" in question_lower
            or "underperform" in question_lower
            or "attention" in question_lower
        ):

            if "performance_type" not in marketing_df.columns:
                return

            poor_campaigns = marketing_df[
                marketing_df["performance_type"].astype(str).str.lower() == "poor"
            ].copy()

            if poor_campaigns.empty:
                st.info(
                    "No campaigns are currently classified as Poor."
                )
                return

            poor_campaigns["roi_percentage"] = pd.to_numeric(
                poor_campaigns["roi_percentage"],
                errors="coerce",
            )

            poor_campaigns = (
                poor_campaigns
                .dropna(subset=["roi_percentage"])
                .sort_values("roi_percentage", ascending=True)
            )

            st.markdown(
                "### ⚠️ Lowest Performing Marketing Campaigns"
            )

            # Horizontal chart keeps long campaign names readable.
            chart_data = poor_campaigns[
                [
                    "campaign_name",
                    "roi_percentage",
                    "channel",
                    "campaign_cost",
                    "revenue_generated",
                ]
            ].copy()

            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "roi_percentage:Q",
                        title="ROI (%)",
                    ),
                    y=alt.Y(
                        "campaign_name:N",
                        sort="-x",
                        title="Campaign",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "campaign_name:N",
                            title="Campaign",
                        ),
                        alt.Tooltip(
                            "channel:N",
                            title="Channel",
                        ),
                        alt.Tooltip(
                            "roi_percentage:Q",
                            title="ROI",
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "campaign_cost:Q",
                            title="Campaign Cost",
                            format=",.2f",
                        ),
                        alt.Tooltip(
                            "revenue_generated:Q",
                            title="Revenue Generated",
                            format=",.2f",
                        ),
                    ],
                )
                .properties(
                    height=max(320, len(chart_data) * 38),
                )
            )

            st.altair_chart(
                chart,
                width="stretch",
            )

            st.caption(
                f"{len(poor_campaigns)} campaigns are classified as Poor. "
                "The chart shows all of them, ordered from lowest to highest ROI."
            )

            return

        # ----------------------------------------------------
        # HIGHEST ROI CHANNEL
        # ----------------------------------------------------

        if (
            "channel" in question_lower
            or "channels" in question_lower
        ):

            channel_data = channel_performance(marketing_df)

            if channel_data.empty:
                return

            channel_chart_data = channel_data[
                [
                    "channel",
                    "overall_roi",
                    "total_cost",
                    "total_revenue",
                ]
            ].copy()

            st.markdown(
                "### 📈 ROI by Marketing Channel"
            )

            chart = (
                alt.Chart(channel_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "overall_roi:Q",
                        title="Aggregated ROI (%)",
                    ),
                    y=alt.Y(
                        "channel:N",
                        sort="-x",
                        title="Marketing Channel",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "channel:N",
                            title="Channel",
                        ),
                        alt.Tooltip(
                            "overall_roi:Q",
                            title="Aggregated ROI",
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "total_cost:Q",
                            title="Total Cost",
                            format=",.2f",
                        ),
                        alt.Tooltip(
                            "total_revenue:Q",
                            title="Total Revenue",
                            format=",.2f",
                        ),
                    ],
                )
                .properties(
                    height=260,
                )
            )

            st.altair_chart(
                chart,
                width="stretch",
            )

            return

        # ----------------------------------------------------
        # DEFAULT MARKETING VISUAL
        # ----------------------------------------------------

        channel_data = channel_performance(marketing_df)

        if channel_data.empty:
            return

        chart_data = channel_data[
            [
                "channel",
                "overall_roi",
                "total_revenue",
            ]
        ].copy()

        st.markdown(
            "### 📣 Marketing Channel Performance"
        )

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "overall_roi:Q",
                    title="Aggregated ROI (%)",
                ),
                y=alt.Y(
                    "channel:N",
                    sort="-x",
                    title="Marketing Channel",
                ),
                tooltip=[
                    alt.Tooltip(
                        "channel:N",
                        title="Channel",
                    ),
                    alt.Tooltip(
                        "overall_roi:Q",
                        title="Aggregated ROI",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "total_revenue:Q",
                        title="Total Revenue",
                        format=",.2f",
                    ),
                ],
            )
            .properties(
                height=260,
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

        return

    # ========================================================
    # NORMAL BUSINESS VISUALS
    # ========================================================

    if master_df is None or master_df.empty:
        return

    st.markdown("## 📊 Visual Insight")

    st.caption(
        "A relevant visualization selected from your business data."
    )

    # ========================================================
    # RETURN QUESTIONS
    # ========================================================

    if (
        "return" in question_lower
        or "returned" in question_lower
    ):

        if "product_name" not in master_df.columns:
            return

        if "is_returned" not in master_df.columns:
            return

        return_data = (
            master_df
            .groupby("product_name")
            .agg(
                Return_Rate=("is_returned", "mean"),
                Orders=("order_item_id", "count"),
            )
        )

        return_data["Return_Rate"] = (
            return_data["Return_Rate"] * 100
        )

        return_data = (
            return_data
            .sort_values("Return_Rate", ascending=False)
            .head(10)
        )

        st.markdown(
            "### 🔄 Products with Highest Return Rates"
        )

        st.bar_chart(
            return_data["Return_Rate"],
            width="stretch",
        )

        return

    # ========================================================
    # CUSTOMER QUESTIONS
    # ========================================================

    if (
        "customer" in question_lower
        or "segment" in question_lower
    ):

        if "customer_segment" not in master_df.columns:
            return

        segment_data = (
            master_df
            .groupby("customer_segment")
            .agg(
                Sales=("sales_amount", "sum"),
                Profit=("profit", "sum"),
            )
            .sort_values("Profit", ascending=False)
        )

        st.markdown(
            "### 👥 Profit by Customer Segment"
        )

        st.bar_chart(
            segment_data["Profit"],
            width="stretch",
        )

        return

    # ========================================================
    # MONTHLY / TREND QUESTIONS
    # ========================================================

    if (
        "month" in question_lower
        or "monthly" in question_lower
        or "trend" in question_lower
        or "over time" in question_lower
        or "quarter" in question_lower
    ):

        if "order_date" not in master_df.columns:
            return

        df = master_df.copy()

        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce",
        )

        monthly_data = (
            df
            .dropna(subset=["order_date"])
            .groupby(
                df["order_date"].dt.to_period("M")
            )
            .agg(
                Sales=("sales_amount", "sum"),
                Profit=("profit", "sum"),
            )
        )

        monthly_data.index = monthly_data.index.astype(str)

        if "profit" in question_lower:

            st.markdown(
                "### 📈 Monthly Profit Trend"
            )

            st.line_chart(
                monthly_data["Profit"],
                width="stretch",
            )

        else:

            st.markdown(
                "### 📈 Monthly Sales Trend"
            )

            st.line_chart(
                monthly_data["Sales"],
                width="stretch",
            )

        return

    # ========================================================
    # PRODUCT QUESTIONS
    # ========================================================

    if "product" in question_lower:

        if "product_name" not in master_df.columns:
            return

        product_data = (
            master_df
            .groupby("product_name")
            .agg(
                Sales=("sales_amount", "sum"),
                Profit=("profit", "sum"),
            )
        )

        if "sales" in question_lower:

            top_products = (
                product_data
                .sort_values("Sales", ascending=False)
                .head(10)
            )

            st.markdown(
                "### 🏆 Top Products by Sales"
            )

            st.bar_chart(
                top_products["Sales"],
                width="stretch",
            )

        else:

            top_products = (
                product_data
                .sort_values("Profit", ascending=False)
                .head(10)
            )

            st.markdown(
                "### 🏆 Top Products by Profit"
            )

            st.bar_chart(
                top_products["Profit"],
                width="stretch",
            )

        return

    # ========================================================
    # CATEGORY QUESTIONS
    # ========================================================

    if "category" in question_lower:

        if "category" not in master_df.columns:
            return

        category_data = (
            master_df
            .groupby("category")
            .agg(
                Sales=("sales_amount", "sum"),
                Profit=("profit", "sum"),
            )
            .sort_values("Sales", ascending=False)
        )

        if (
            "profit" in question_lower
            or "profitable" in question_lower
            or "profitability" in question_lower
        ):

            st.markdown(
                "### 💰 Profit by Category"
            )

            st.bar_chart(
                category_data["Profit"],
                width="stretch",
            )

        else:

            st.markdown(
                "### 💰 Sales by Category"
            )

            st.bar_chart(
                category_data["Sales"],
                width="stretch",
            )

        return

    # ========================================================
    # DEFAULT BUSINESS VISUAL
    # ========================================================

    if "category" not in master_df.columns:
        return

    category_data = (
        master_df
        .groupby("category")
        .agg(
            Sales=("sales_amount", "sum"),
            Profit=("profit", "sum"),
        )
        .sort_values("Profit", ascending=False)
    )

    st.markdown(
        "### 💰 Profit by Category"
    )

    st.bar_chart(
        category_data["Profit"],
        width="stretch",
    )


# ============================================================
# LOAD BUSINESS DATA
# ============================================================

business_data = load_business_data()

kpis = calculate_kpis(business_data)

master_df = business_data.get("master")


# ============================================================
# PREPARE MASTER DATA
# ============================================================

if master_df is not None and not master_df.empty:

    numeric_columns = [
        "sales_amount",
        "profit",
        "total_cost",
        "quantity",
        "discount_percentage",
        "profit_margin_percentage",
    ]

    for column in numeric_columns:

        if column in master_df.columns:

            master_df[column] = pd.to_numeric(
                master_df[column],
                errors="coerce",
            ).fillna(0)


# ============================================================
# DASHBOARD FILTERS
# ============================================================

filtered_master_df = master_df.copy() if master_df is not None else pd.DataFrame()

with st.sidebar:
    st.divider()
    st.markdown("### 🎛️ Dashboard Filters")
    st.caption("Filters apply to the Business Overview, Business Analytics, and Visual Insights sections.")

    if filtered_master_df is not None and not filtered_master_df.empty:
        if "order_date" in filtered_master_df.columns:
            filtered_master_df["order_date"] = pd.to_datetime(
                filtered_master_df["order_date"], errors="coerce"
            )

        if st.button("🔄 Reset Filters", width="stretch"):
            for key in [
                "filter_date_range",
                "filter_category",
                "filter_segment",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

        if "order_date" in filtered_master_df.columns:
            valid_dates = filtered_master_df["order_date"].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="filter_date_range",
                )

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_master_df = filtered_master_df[
                        (filtered_master_df["order_date"].dt.date >= start_date)
                        & (filtered_master_df["order_date"].dt.date <= end_date)
                    ]

        if "category" in master_df.columns:
            category_options = sorted(
                master_df["category"].dropna().astype(str).unique().tolist()
            )
            selected_categories = st.multiselect(
                "🏷️ Category",
                options=category_options,
                default=category_options,
                key="filter_category",
            )
            filtered_master_df = filtered_master_df[
                filtered_master_df["category"].astype(str).isin(selected_categories)
            ]

        if "customer_segment" in master_df.columns:
            segment_options = sorted(
                master_df["customer_segment"].dropna().astype(str).unique().tolist()
            )
            selected_segments = st.multiselect(
                "👥 Customer Segment",
                options=segment_options,
                default=segment_options,
                key="filter_segment",
            )
            filtered_master_df = filtered_master_df[
                filtered_master_df["customer_segment"].astype(str).isin(selected_segments)
            ]

        st.caption(
            f"Showing {len(filtered_master_df):,} of {len(master_df):,} records"
        )
    else:
        st.info("Dashboard filters are unavailable because the master dataset was not found.")

filtered_kpis = calculate_filtered_kpis(filtered_master_df)


# ============================================================
# LOAD MARKETING DATA
# ============================================================

try:

    marketing_df = load_marketing_data()

except Exception:

    marketing_df = pd.DataFrame()


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    index, metadata = load_vector_store()

    model = load_embedding_model()

    return index, metadata, model


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📊 InsightRAG</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### AI Business Analyst")

    st.write(
        "Ask questions about your business data "
        "and get evidence-based insights using "
        "Retrieval-Augmented Generation."
    )

    st.divider()

    st.markdown("### 💡 What can I ask?")

    st.markdown(
        """
        - 📈 Sales performance
        - 💰 Profitability
        - 📦 Product performance
        - 👥 Customer segments
        - 🔄 Returns
        - 🏷️ Categories
        - 📅 Monthly trends
        - 📣 Marketing performance
        - 🎯 Business recommendations
        """
    )

    st.divider()

    st.markdown("### 🧠 How it works")

    st.markdown(
        """
        **1. Question**

        You ask a business question.

        **2. Retrieval**

        FAISS finds the most relevant business data.

        **3. Generation**

        The retrieved context is sent to the AI model.

        **4. Insight**

        InsightRAG generates a business-focused answer.
        """
    )

    st.divider()

    st.caption(
        "InsightRAG • AI-Powered Business Intelligence"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.title("📊 InsightRAG")

st.markdown(
    '<div class="hero-subtitle">AI-Powered Business Intelligence Assistant</div>',
    unsafe_allow_html=True,
)

st.write(
    "Ask questions about sales, profitability, "
    "products, customers, returns, categories, and marketing."
)


# ============================================================
# BUSINESS ALERTS
# ============================================================

anomalies = display_business_alerts()


# ============================================================
# EXECUTIVE BUSINESS INSIGHTS
# ============================================================

st.divider()

st.header("🧠 Executive Business Insights")

st.caption(
    "AI-generated business reasoning based on KPIs, analytics, "
    "and automatically detected anomalies."
)


if "executive_insights" not in st.session_state:

    with st.spinner(
        "AI is analyzing the business data..."
    ):

        st.session_state.executive_insights = (
            generate_executive_insights(
                master_df,
                anomalies
            )
        )


with st.container(border=True):

    st.markdown(
        st.session_state.executive_insights
    )


# ============================================================
# INITIALIZE RAG
# ============================================================

try:

    with st.spinner("Initializing InsightRAG..."):

        index, metadata, model = initialize_rag()

except Exception as error:

    st.error(
        f"Unable to initialize InsightRAG:\n\n{error}"
    )

    st.stop()


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.markdown("## 📊 Business Overview")

st.markdown(
    '<div class="section-note">Key performance indicators from your business dataset.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# KPI CARDS
# ============================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


with kpi1:

    st.metric(
        "💰 Total Sales",
        f"₹{filtered_kpis['sales']:,.0f}",
    )


with kpi2:

    st.metric(
        "📈 Total Profit",
        f"₹{filtered_kpis['profit']:,.0f}",
    )


with kpi3:

    st.metric(
        "📊 Profit Margin",
        f"{filtered_kpis['margin']:.2f}%",
    )


with kpi4:

    st.metric(
        "🛒 Total Orders",
        f"{filtered_kpis['orders']:,}",
    )


with kpi5:

    st.metric(
        "🔄 Return Rate",
        f"{filtered_kpis['return_rate']:.2f}%",
    )


st.divider()


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

st.markdown("## 📈 Business Analytics")

st.markdown(
    '<div class="section-note">Explore sales, profitability, product performance, and return trends.</div>',
    unsafe_allow_html=True,
)


if filtered_master_df is not None and not filtered_master_df.empty:

    # ========================================================
    # CATEGORY PERFORMANCE
    # ========================================================

    st.markdown("### 🏷️ Category Performance")

    category_data = (
        filtered_master_df
        .groupby("category", as_index=True)
        .agg(
            Sales=("sales_amount", "sum"),
            Profit=("profit", "sum"),
        )
        .sort_values("Sales", ascending=False)
    )

    category_col1, category_col2 = st.columns(2)

    with category_col1:

        st.markdown("**Sales by Category**")

        st.bar_chart(
            category_data["Sales"],
            width="stretch",
        )

    with category_col2:

        st.markdown("**Profit by Category**")

        st.bar_chart(
            category_data["Profit"],
            width="stretch",
        )

    st.divider()

    # ========================================================
    # MONTHLY PERFORMANCE
    # ========================================================

    st.markdown("### 📅 Monthly Performance")

    if "order_date" in filtered_master_df.columns:

        filtered_master_df["order_date"] = pd.to_datetime(
            filtered_master_df["order_date"],
            errors="coerce",
        )

        monthly_data = (
            filtered_master_df
            .dropna(subset=["order_date"])
            .groupby(
                filtered_master_df["order_date"].dt.to_period("M")
            )
            .agg(
                Sales=("sales_amount", "sum"),
                Profit=("profit", "sum"),
            )
        )

        monthly_data.index = monthly_data.index.astype(str)

        monthly_col1, monthly_col2 = st.columns(2)

        with monthly_col1:

            st.markdown("**Monthly Sales Trend**")

            st.line_chart(
                monthly_data["Sales"],
                width="stretch",
            )

        with monthly_col2:

            st.markdown("**Monthly Profit Trend**")

            st.line_chart(
                monthly_data["Profit"],
                width="stretch",
            )

    st.divider()

    # ========================================================
    # PRODUCT PERFORMANCE
    # ========================================================

    st.markdown("### 🏆 Product Performance")

    product_data = (
        filtered_master_df
        .groupby("product_name", as_index=True)
        .agg(
            Sales=("sales_amount", "sum"),
            Profit=("profit", "sum"),
        )
    )

    top_products = (
        product_data
        .sort_values("Profit", ascending=False)
        .head(10)
    )

    product_col1, product_col2 = st.columns(2)

    with product_col1:

        st.markdown("**Top 10 Products by Profit**")

        st.bar_chart(
            top_products["Profit"],
            width="stretch",
        )

    if "is_returned" in filtered_master_df.columns:

        return_analysis = (
            filtered_master_df
            .groupby("product_name")
            .agg(
                Return_Rate=("is_returned", "mean"),
            )
        )

        return_analysis["Return_Rate"] = (
            return_analysis["Return_Rate"] * 100
        )

        top_return_products = (
            return_analysis
            .sort_values(
                "Return_Rate",
                ascending=False,
            )
            .head(10)
        )

        with product_col2:

            st.markdown(
                "**Top 10 Products by Return Rate**"
            )

            st.bar_chart(
                top_return_products["Return_Rate"],
                width="stretch",
            )

    st.divider()

    # ========================================================
    # ANALYTICS SUMMARY
    # ========================================================

    with st.expander("🔎 View Analytics Summary"):

        st.dataframe(
            category_data.style.format(
                {
                    "Sales": "₹{:,.0f}",
                    "Profit": "₹{:,.0f}",
                }
            ),
            width="stretch",
        )


else:

    st.info(
        "Master analytics dataset was not found."
    )


# ============================================================
# MARKETING ANALYTICS
# ============================================================

st.divider()

st.markdown("## 📣 Marketing Analytics")

st.caption(
    "Analyze campaign spending, revenue generation, ROI, "
    "channel performance, and campaign effectiveness."
)


if marketing_df is not None and not marketing_df.empty:

    # ========================================================
    # MARKETING KPIs
    # ========================================================

    marketing_kpis = calculate_marketing_kpis(
        marketing_df
    )

    marketing_kpi1, marketing_kpi2, marketing_kpi3, marketing_kpi4 = (
        st.columns(4)
    )

    with marketing_kpi1:

        st.metric(
            "🎯 Total Campaigns",
            f"{marketing_kpis['total_campaigns']:,}",
        )

    with marketing_kpi2:

        st.metric(
            "💸 Marketing Spend",
            f"₹{marketing_kpis['total_cost']:,.0f}",
        )

    with marketing_kpi3:

        st.metric(
            "💰 Revenue Generated",
            f"₹{marketing_kpis['total_revenue']:,.0f}",
        )

    with marketing_kpi4:

        st.metric(
            "📈 Overall ROI",
            f"{marketing_kpis['overall_roi']:.2f}%",
        )

    st.divider()

    # ========================================================
    # CHANNEL PERFORMANCE
    # ========================================================

    st.markdown("### 📣 Channel Performance")

    channel_data = channel_performance(
        marketing_df
    )

    if not channel_data.empty:

        channel_col1, channel_col2 = st.columns(2)

        with channel_col1:

            st.markdown(
                "**Overall ROI by Marketing Channel**"
            )

            channel_chart_data = (
                channel_data
                .set_index("channel")["overall_roi"]
                .sort_values(ascending=False)
            )

            st.bar_chart(
                channel_chart_data,
                width="stretch",
            )

        with channel_col2:

            st.markdown(
                "**Revenue Generated by Channel**"
            )

            channel_revenue_data = (
                channel_data
                .set_index("channel")["total_revenue"]
                .sort_values(ascending=False)
            )

            st.bar_chart(
                channel_revenue_data,
                width="stretch",
            )

        st.dataframe(
            channel_data.style.format(
                {
                    "total_cost": "₹{:,.0f}",
                    "total_revenue": "₹{:,.0f}",
                    "average_roi": "{:.2f}%",
                    "overall_roi": "{:.2f}%",
                }
            ),
            width="stretch",
        )

    st.divider()

    # ========================================================
    # PERFORMANCE TYPE
    # ========================================================

    st.markdown("### 🏅 Campaign Performance Classification")

    performance_data = performance_type_summary(
        marketing_df
    )

    if not performance_data.empty:

        performance_col1, performance_col2 = st.columns(2)

        with performance_col1:

            st.markdown(
                "**Campaigns by Performance Type**"
            )

            performance_count = (
                performance_data
                .set_index("performance_type")["campaigns"]
                .sort_values(ascending=False)
            )

            st.bar_chart(
                performance_count,
                width="stretch",
            )

        with performance_col2:

            st.markdown(
                "**ROI by Performance Type**"
            )

            performance_roi = (
                performance_data
                .set_index("performance_type")["overall_roi"]
                .sort_values(ascending=False)
            )

            st.bar_chart(
                performance_roi,
                width="stretch",
            )

        st.dataframe(
            performance_data.style.format(
                {
                    "total_cost": "₹{:,.0f}",
                    "total_revenue": "₹{:,.0f}",
                    "average_roi": "{:.2f}%",
                    "overall_roi": "{:.2f}%",
                }
            ),
            width="stretch",
        )

    st.divider()

    # ========================================================
    # TOP CAMPAIGNS
    # ========================================================

    st.markdown("### 🚀 Top Campaigns by ROI")

    top_campaign_data = top_campaigns_by_roi(
        marketing_df,
        10,
    )

    if not top_campaign_data.empty:

        st.dataframe(
            top_campaign_data.style.format(
                {
                    "campaign_cost": "₹{:,.0f}",
                    "revenue_generated": "₹{:,.0f}",
                    "roi_percentage": "{:.2f}%",
                }
            ),
            width="stretch",
        )

    st.divider()

    # ========================================================
    # LOWEST CAMPAIGNS
    # ========================================================

    st.markdown("### ⚠️ Campaigns Requiring Attention")

    low_campaign_data = lowest_campaigns_by_roi(
        marketing_df,
        10,
    )

    if not low_campaign_data.empty:

        st.dataframe(
            low_campaign_data.style.format(
                {
                    "campaign_cost": "₹{:,.0f}",
                    "revenue_generated": "₹{:,.0f}",
                    "roi_percentage": "{:.2f}%",
                }
            ),
            width="stretch",
        )

    st.divider()

    # ========================================================
    # MARKETING COST VS REVENUE
    # ========================================================

    st.markdown("### 💸 Campaign Cost vs Revenue")

    cost_revenue_data = marketing_df[
        [
            "campaign_name",
            "campaign_cost",
            "revenue_generated",
        ]
    ].copy()

    cost_revenue_data = (
        cost_revenue_data
        .set_index("campaign_name")
        .sort_values(
            "revenue_generated",
            ascending=False,
        )
        .head(10)
    )

    cost_revenue_chart = cost_revenue_data[
        [
            "campaign_cost",
            "revenue_generated",
        ]
    ]

    st.bar_chart(
        cost_revenue_chart,
        width="stretch",
    )

    # ========================================================
    # MARKETING SUMMARY
    # ========================================================

    with st.expander("🔎 View Marketing Analytics Summary"):

        st.write(
            f"**Average Campaign ROI:** "
            f"{marketing_kpis['average_roi']:.2f}%"
        )

        st.write(
            f"**Average Campaign Duration:** "
            f"{marketing_kpis['average_duration']:.2f} days"
        )

        if not channel_data.empty:

            best_channel = channel_data.iloc[0]

            st.write(
                f"**Highest Aggregated ROI Channel:** "
                f"{best_channel['channel']} "
                f"({best_channel['overall_roi']:.2f}%)"
            )

        if not top_campaign_data.empty:

            best_campaign = top_campaign_data.iloc[0]

            st.write(
                f"**Highest ROI Campaign:** "
                f"{best_campaign['campaign_name']} "
                f"({best_campaign['roi_percentage']:.2f}%)"
            )

        if not low_campaign_data.empty:

            lowest_campaign = low_campaign_data.iloc[0]

            st.write(
                f"**Lowest ROI Campaign:** "
                f"{lowest_campaign['campaign_name']} "
                f"({lowest_campaign['roi_percentage']:.2f}%)"
            )

else:

    st.info(
        "Marketing campaigns dataset was not found."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("## 🧠 InsightRAG System")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Knowledge Chunks",
        len(metadata),
    )

with col2:

    st.metric(
        "Vector Records",
        index.ntotal,
    )

with col3:

    st.metric(
        "Embedding Model",
        "MiniLM",
    )


st.divider()


# ============================================================
# AI BUSINESS ANALYST
# ============================================================

st.markdown("## 🤖 AI Business Analyst")

st.caption(
    "Ask questions and get evidence-based business insights."
)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown("### 💬 Try a business question")

suggested_questions = [
    "Which category has the highest sales and profit?",
    "What caused the Electronics profitability issue in Q3 2025?",
    "Which customer segment generates the highest profit?",
    "Which products have the highest return rates?",
    "What are the biggest business problems?",
    "Give me recommendations to improve business profitability.",
    "Which marketing channel has the highest ROI?",
    "Which marketing campaigns performed poorly?",
]


# ============================================================
# SESSION STATE
# ============================================================

if "question_input" not in st.session_state:

    st.session_state.question_input = ""


# ============================================================
# QUESTION CALLBACK
# ============================================================

def select_question(question_text):

    st.session_state.question_input = question_text


# ============================================================
# SUGGESTED QUESTION BUTTONS
# ============================================================

cols = st.columns(2)

for i, suggestion in enumerate(suggested_questions):

    with cols[i % 2]:

        st.button(
            suggestion,
            width="stretch",
            key=f"suggestion_{i}",
            on_click=select_question,
            args=(suggestion,),
        )


# ============================================================
# QUESTION INPUT
# ============================================================

scope_label = (
    f"Filtered analysis • {len(filtered_master_df):,} records"
    if filtered_master_df is not None and len(filtered_master_df) != len(master_df)
    else f"Full business dataset • {len(filtered_master_df):,} records"
)

st.markdown(
    f'<div class="analysis-scope">📌 <strong>Analysis scope:</strong> {scope_label}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="question-label">Ask a business question</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Business question",
    key="question_input",
    placeholder="Example: Which category has the highest profit?",
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Analyze Business Data",
    type="primary",
    width="stretch",
)


# ============================================================
# CONVERSATION MEMORY + FOLLOW-UP REFERENCE RESOLUTION
# ============================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def _find_known_referent(text, master_df=None, marketing_df=None):
    """
    Find a known business entity explicitly mentioned in text.

    Matching uses whole words so values such as the marketing performance
    type "High" do not accidentally match words such as "highest".
    """
    if not text:
        return None

    text_lower = str(text).lower()

    def find_from_df(df, column_names):
        if df is None or df.empty:
            return None

        candidates = []

        for column in column_names:
            if column not in df.columns:
                continue

            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

            for value in values:
                if value and len(value) > 1:
                    candidates.append(value)

        # Prefer the longest exact entity name, but never allow substring
        # matches such as "High" inside "highest".
        for candidate in sorted(set(candidates), key=len, reverse=True):
            pattern = rf"(?<!\w){re.escape(candidate.lower())}(?!\w)"
            if re.search(pattern, text_lower):
                return candidate

        return None

    # Business entities first. This prevents marketing labels such as
    # "High" from becoming the referent of a normal business question.
    referent = find_from_df(
        master_df,
        ["category", "product_name", "customer_segment"],
    )
    if referent:
        return referent

    return find_from_df(
        marketing_df,
        ["channel", "campaign_name", "performance_type"],
    )


def _infer_referent_from_turn(item, master_df=None, marketing_df=None):
    """
    Infer the actual entity discussed in a conversation turn.

    For ranking questions, derive the referent directly from the filtered
    business dataframe instead of scanning arbitrary entity names in the
    question. This prevents generic words such as "category" or unrelated
    entities in an AI response from becoming the referent.
    """
    if not item:
        return None

    user_question = str(item.get("question", ""))
    answer = str(item.get("answer", ""))
    q = user_question.lower()

    # ------------------------------------------------------------
    # Deterministic ranking resolution for the business dataset
    # ------------------------------------------------------------
    if master_df is not None and not master_df.empty:
        entity_column = None
        if "category" in q:
            entity_column = "category"
        elif "product" in q:
            entity_column = "product_name"
        elif "customer segment" in q or "segment" in q:
            entity_column = "customer_segment"

        ranking_metric = None
        if "return rate" in q or "most returned" in q or "highest returns" in q:
            ranking_metric = "return_rate"
        elif "profit margin" in q or "highest margin" in q:
            ranking_metric = "profit_margin"
        elif "profit" in q:
            ranking_metric = "profit"
        elif "sales" in q or "revenue" in q:
            ranking_metric = "sales"

        is_highest = any(
            phrase in q
            for phrase in ["highest", "most", "top", "best"]
        )
        is_lowest = any(
            phrase in q
            for phrase in ["lowest", "least", "bottom", "worst"]
        )

        if entity_column and ranking_metric and (is_highest or is_lowest):
            try:
                work = master_df.copy()
                if entity_column not in work.columns:
                    raise ValueError("Entity column unavailable")

                if ranking_metric == "profit" and "profit" in work.columns:
                    summary = work.groupby(entity_column)["profit"].sum()
                elif ranking_metric == "sales" and "sales_amount" in work.columns:
                    summary = work.groupby(entity_column)["sales_amount"].sum()
                elif ranking_metric == "profit_margin" and {"profit", "sales_amount"}.issubset(work.columns):
                    grouped = work.groupby(entity_column).agg(
                        profit=("profit", "sum"),
                        sales=("sales_amount", "sum"),
                    )
                    grouped = grouped[grouped["sales"] != 0]
                    summary = grouped["profit"] / grouped["sales"]
                elif ranking_metric == "return_rate" and "is_returned" in work.columns:
                    summary = work.groupby(entity_column)["is_returned"].mean()
                else:
                    summary = None

                if summary is not None and not summary.empty:
                    target = summary.idxmax() if is_highest else summary.idxmin()
                    if pd.notna(target):
                        return str(target)
            except Exception:
                pass

    # ------------------------------------------------------------
    # Explicit entities in the user's question
    # ------------------------------------------------------------
    referent = _find_known_referent(
        user_question,
        master_df,
        marketing_df,
    )
    if referent:
        return referent

    # ------------------------------------------------------------
    # Fallback: entity explicitly named by the assistant
    # ------------------------------------------------------------
    return _find_known_referent(
        answer,
        master_df,
        marketing_df,
    )


def resolve_follow_up_question(
    question,
    conversation_history,
    master_df=None,
    marketing_df=None,
):
    """
    Resolve conversational references such as:
    "its profit margin", "why is it different?", and
    "what about that category?".
    """
    if not conversation_history:
        return question, _find_known_referent(
            question,
            master_df,
            marketing_df,
        )

    question_lower = question.lower()

    reference_patterns = [
        r"\bits\b", r"\bit\b", r"\bthey\b", r"\btheir\b",
        r"\bthem\b", r"\bthis category\b", r"\bthat category\b",
        r"\bthis product\b", r"\bthat product\b",
        r"\bthis channel\b", r"\bthat channel\b",
        r"\bthis campaign\b", r"\bthat campaign\b",
        r"\bthe same category\b", r"\bthe same product\b",
        r"\bthe same channel\b", r"\bthe same campaign\b",
    ]

    has_reference = any(
        re.search(pattern, question_lower)
        for pattern in reference_patterns
    )

    short_follow_up_patterns = [
        r"^what caused (the )?(issue|problem|decline|drop|change)",
        r"^why (is|was|did|does) ",
        r"^what about ",
        r"^how can (we|i) ",
        r"^what happened ",
    ]

    is_short_follow_up = any(
        re.search(pattern, question_lower)
        for pattern in short_follow_up_patterns
    )

    if not has_reference and not is_short_follow_up:
        return question, _find_known_referent(
            question,
            master_df,
            marketing_df,
        )

    # Always infer from the most recent turn first. Do not blindly reuse an
    # older stored referent: that can leak an unrelated marketing entity
    # into a new business conversation.
    referent = None
    for item in reversed(conversation_history):
        referent = _infer_referent_from_turn(
            item,
            master_df,
            marketing_df,
        )
        if referent:
            break

    if referent is None:
        return question, None

    resolved = question

    replacements = [
        (r"\bthe same category\b", f"the {referent} category"),
        (r"\bthe same product\b", referent),
        (r"\bthe same channel\b", referent),
        (r"\bthe same campaign\b", referent),
        (r"\bthis category\b", f"the {referent} category"),
        (r"\bthat category\b", f"the {referent} category"),
        (r"\bthis product\b", referent),
        (r"\bthat product\b", referent),
        (r"\bthis channel\b", referent),
        (r"\bthat channel\b", referent),
        (r"\bthis campaign\b", referent),
        (r"\bthat campaign\b", referent),
        (r"\bits\b", referent),
        (r"\btheir\b", referent),
        (r"\bit\b", referent),
        (r"\bthey\b", referent),
        (r"\bthem\b", referent),
    ]

    for pattern, replacement in replacements:
        resolved = re.sub(
            pattern,
            replacement,
            resolved,
            count=1,
            flags=re.IGNORECASE,
        )

    if resolved == question and is_short_follow_up:
        resolved = f"Regarding {referent}: {question}"

    return resolved, referent


def _is_marketing_referent(referent, marketing_df=None):
    """Return True only when the resolved entity belongs to marketing data."""
    if not referent or marketing_df is None or marketing_df.empty:
        return False

    for column in ["channel", "campaign_name", "performance_type"]:
        if column not in marketing_df.columns:
            continue
        values = (
            marketing_df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        if str(referent).strip().lower() in values:
            return True

    return False


def _is_business_referent(referent, master_df=None):
    """Return True when the resolved entity belongs to the business dataset."""
    if not referent or master_df is None or master_df.empty:
        return False

    for column in ["category", "product_name", "customer_segment"]:
        if column not in master_df.columns:
            continue
        values = (
            master_df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        if str(referent).strip().lower() in values:
            return True

    return False


def build_deterministic_follow_up_answer(question, referent, df):
    """
    Answer a category-vs-overall comparison directly from the current
    dashboard dataframe. This prevents unrelated retrieved chunks from
    hijacking a conversational follow-up such as "Why is it different
    from the overall business?".
    """
    if not referent or df is None or df.empty:
        return None

    q = str(question).lower()
    if not re.search(r"\b(why|how)\b", q):
        return None
    if not re.search(r"\b(different|difference|compare|compared)\b", q):
        return None
    if not re.search(r"\boverall\s+(business|company|performance|margin)\b", q):
        return None

    required = {"category", "sales_amount", "profit"}
    if not required.issubset(df.columns):
        return None

    work = df.copy()
    work["sales_amount"] = pd.to_numeric(work["sales_amount"], errors="coerce")
    work["profit"] = pd.to_numeric(work["profit"], errors="coerce")
    work = work.dropna(subset=["sales_amount", "profit"])

    if work.empty:
        return None

    match = work[
        work["category"].astype(str).str.strip().str.lower()
        == str(referent).strip().lower()
    ]
    if match.empty:
        return None

    category_sales = match["sales_amount"].sum()
    category_profit = match["profit"].sum()
    overall_sales = work["sales_amount"].sum()
    overall_profit = work["profit"].sum()

    if category_sales == 0 or overall_sales == 0:
        return None

    category_margin = category_profit / category_sales * 100
    overall_margin = overall_profit / overall_sales * 100
    difference = category_margin - overall_margin

    direction = "higher" if difference > 0 else "lower" if difference < 0 else "the same as"

    if direction == "the same as":
        comparison = (
            f"The {referent} category margin is the same as the overall business margin "
            f"at {overall_margin:.2f}%."
        )
    else:
        comparison = (
            f"The {referent} category has a profit margin of {category_margin:.2f}%, "
            f"which is {abs(difference):.2f} percentage points {direction} than the "
            f"overall business margin of {overall_margin:.2f}%."
        )

    return f"""
**ANSWER**

{comparison}

**WHAT THE DATA SHOWS**

- {referent} sales: ₹{category_sales:,.2f}
- {referent} profit: ₹{category_profit:,.2f}
- {referent} profit margin: {category_margin:.2f}%
- Overall business sales: ₹{overall_sales:,.2f}
- Overall business profit: ₹{overall_profit:,.2f}
- Overall business profit margin: {overall_margin:.2f}%
- Margin difference: {difference:+.2f} percentage points

**WHAT THE DATA DOES NOT ESTABLISH**

- The comparison shows the difference in observed margins, but it does not by itself establish a specific causal driver such as pricing, discounts, or costs.

**BUSINESS IMPLICATION**

- The {referent} category contributes to the overall business result with a margin that is {direction} than the aggregate business margin. The specific reasons for the difference require further category-level investigation.
""".strip()



def build_deterministic_filtered_kpi_answer(question, df):
    """
    Answer direct business KPI questions from the current dashboard dataframe.
    This is authoritative when filters are active and prevents unrelated RAG
    chunks from returning a different aggregate.
    """
    if df is None or df.empty:
        return None

    q = str(question).lower().strip()

    # Avoid taking over analytical questions such as "why did profit decline".
    direct_patterns = {
        "profit": [r"\btotal\s+profit\b", r"\boverall\s+profit\b", r"^profit\??$"],
        "sales": [r"\btotal\s+sales\b", r"\boverall\s+sales\b", r"^sales\??$"],
        "margin": [r"\bprofit\s+margin\b", r"\boverall\s+margin\b", r"^margin\??$"],
        "orders": [r"\btotal\s+orders\b", r"\boverall\s+orders\b", r"^orders\??$"],
        "return_rate": [r"\breturn\s+rate\b", r"\boverall\s+return\s+rate\b"],
    }

    metric = None
    for name, patterns in direct_patterns.items():
        if any(re.search(pattern, q) for pattern in patterns):
            metric = name
            break

    if metric is None:
        return None

    # Do not treat "profit margin" as a profit query.
    if metric == "profit" and "margin" in q:
        return None

    work = df.copy()

    sales = pd.to_numeric(
        work.get("sales_amount", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()

    profit = pd.to_numeric(
        work.get("profit", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()

    orders = (
        work["order_id"].nunique()
        if "order_id" in work.columns
        else len(work)
    )

    if "is_returned" in work.columns and "order_id" in work.columns:
        flags = pd.to_numeric(work["is_returned"], errors="coerce").fillna(0)
        returns = int(
            pd.DataFrame({"order_id": work["order_id"], "returned": flags})
            .groupby("order_id")["returned"]
            .max()
            .gt(0)
            .sum()
        )
    else:
        returns = 0

    margin = (profit / sales * 100) if sales else 0
    return_rate = (returns / orders * 100) if orders else 0

    values = {
        "profit": f"₹{profit:,.2f}",
        "sales": f"₹{sales:,.2f}",
        "margin": f"{margin:.2f}%",
        "orders": f"{orders:,}",
        "return_rate": f"{return_rate:.2f}%",
    }

    labels = {
        "profit": "total profit",
        "sales": "total sales",
        "margin": "profit margin",
        "orders": "total orders",
        "return_rate": "return rate",
    }

    return f"""**ANSWER**

The {labels[metric]} for the current analysis scope is **{values[metric]}**.

**WHAT THE DATA SHOWS**

- Analysis scope: **{len(work):,} records**
- {labels[metric].title()}: **{values[metric]}**
- Total Sales: **₹{sales:,.2f}**
- Total Profit: **₹{profit:,.2f}**
- Profit Margin: **{margin:.2f}%**
- Total Orders: **{orders:,}**
- Return Rate: **{return_rate:.2f}%**

**WHAT THE DATA DOES NOT ESTABLISH**

- This KPI summarizes the currently selected dashboard data. It does not by itself explain the causes behind the metric.

**BUSINESS IMPLICATION**

- The reported KPI reflects the current dashboard filters and can be compared with the full business dataset or other filtered scopes.
""".strip()

def build_conversation_context(history, limit=4):
    """
    Build a compact recent-conversation block for the generation model.
    """
    recent = history[-limit:]

    if not recent:
        return ""

    lines = ["RECENT CONVERSATION CONTEXT"]

    for item in recent:
        lines.append(f"User: {item.get('question', '')}")
        lines.append(f"Assistant: {item.get('answer', '')}")

    return "\n".join(lines)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a business question first."
        )

    else:

        try:

            # =================================================
            # CONVERSATION CONTEXT
            # =================================================

            contextual_question, referent = resolve_follow_up_question(
                question,
                st.session_state.conversation_history,
                filtered_master_df,
                marketing_df,
            )

            conversation_context = build_conversation_context(
                st.session_state.conversation_history
            )

            if conversation_context:
                contextual_question = (
                    f"{contextual_question}\n\n"
                    f"{conversation_context}"
                )

            # =================================================
            # DETERMINISTIC FOLLOW-UP ANALYSIS
            # =================================================
            # For a category-vs-overall comparison, use the current filtered
            # dataframe directly. This is more reliable than allowing a
            # retrieved but unrelated chunk (for example discount analysis)
            # to answer the follow-up.

            # For comparison follow-ups, independently recover the latest
            # business category from conversation memory. This makes the
            # comparison path deterministic even if the general referent
            # resolver picked a non-category entity or the previous answer
            # contained unrelated RAG terms.
            comparison_referent = referent

            comparison_follow_up = (
                re.search(r"\b(why|how)\b", question.lower()) is not None
                and re.search(r"\b(different|difference|compare|compared)\b", question.lower()) is not None
                and re.search(r"\boverall\s+(business|company|performance|margin)\b", question.lower()) is not None
            )

            if comparison_follow_up and st.session_state.conversation_history:
                for previous_item in reversed(st.session_state.conversation_history):
                    candidate = _infer_referent_from_turn(
                        previous_item,
                        filtered_master_df,
                        marketing_df,
                    )
                    if candidate and _is_business_referent(candidate, filtered_master_df):
                        # The comparison function currently operates on
                        # categories, so prefer a category explicitly present
                        # in the filtered business data.
                        if "category" in filtered_master_df.columns:
                            category_values = (
                                filtered_master_df["category"]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .tolist()
                            )
                            if any(candidate.lower() == value.lower() for value in category_values):
                                comparison_referent = candidate
                                break

            direct_follow_up_answer = build_deterministic_follow_up_answer(
                question,
                comparison_referent,
                filtered_master_df,
            )

            # Direct KPI questions must use the current dashboard scope.
            # Otherwise FAISS can retrieve an unrelated historical chunk and
            # return a different aggregate than the filtered KPI cards.
            direct_filtered_kpi_answer = build_deterministic_filtered_kpi_answer(
                question,
                filtered_master_df,
            )

            # =================================================
            # QUESTION ROUTING
            # =================================================
            # Never classify the entire conversation context as a marketing
            # question. A previous marketing answer can contain words such as
            # "campaign", "ROI", or "marketing" even when the current
            # follow-up is a normal business question.
            #
            # For follow-ups, the resolved referent is authoritative: if it
            # belongs to the business dataset (for example, Electronics),
            # route to the business/RAG path; if it belongs to marketing data,
            # route to the marketing path.

            marketing_route = is_marketing_question(question)

            if referent:
                if _is_business_referent(referent, filtered_master_df):
                    marketing_route = False
                elif _is_marketing_referent(referent, marketing_df):
                    marketing_route = True

            # =================================================
            # DETERMINISTIC BUSINESS FOLLOW-UP
            # =================================================

            if direct_follow_up_answer is not None:

                answer = direct_follow_up_answer
                results = []

            elif direct_filtered_kpi_answer is not None and not marketing_route:

                answer = direct_filtered_kpi_answer
                results = []

            # =================================================
            # MARKETING QUESTION
            # =================================================

            elif marketing_route:

                with st.spinner(
                    "Analyzing marketing performance..."
                ):

                    answer = generate_marketing_answer(
                        contextual_question,
                        marketing_df,
                    )

                    results = []

            # =================================================
            # NORMAL RAG QUESTION
            # =================================================

            else:

                with st.spinner(
                    "Searching business knowledge base and generating insight..."
                ):

                    answer, results = ask_question(
                        question=contextual_question,
                        model=model,
                        index=index,
                        metadata=metadata,
                        top_k=5,
                    )

            # =================================================
            # BUSINESS ANALYSIS
            # =================================================

            st.divider()

            st.markdown(
                "## 🤖 Business Analysis"
            )

            # =================================================
            # QUESTION
            # =================================================

            st.markdown(
                "### 👨‍💼 Business Question"
            )

            st.write(question)

            # =================================================
            # ANSWER
            # =================================================

            st.markdown(
                "### 💬 Analysis Result"
            )

            # The AI already structures its response with section headings.
            # Remove only a redundant leading "ANSWER" heading to keep the UI clean.
            display_answer = answer.strip() if isinstance(answer, str) else str(answer)
            display_lines = display_answer.splitlines()
            while display_lines and not display_lines[0].strip():
                display_lines.pop(0)
            if display_lines and display_lines[0].strip().lower() in {
                "answer",
                "## answer",
                "### answer",
                "**answer**",
            }:
                display_lines.pop(0)
            display_answer = "\n".join(display_lines).strip()

            with st.container(border=True):
                st.markdown(display_answer)

            # =================================================
            # VISUAL INSIGHT
            # =================================================

            show_visual_insight(
                question,
                filtered_master_df,
                marketing_df,
            )

            # =================================================
            # SOURCES
            # =================================================

            if results:

                st.markdown(
                    "## 📚 Sources Used"
                )

                unique_sources = []

                for result in results:

                    source = result.get(
                        "source",
                        "Unknown source",
                    )

                    if source not in unique_sources:

                        unique_sources.append(source)

                for source in unique_sources:

                    st.markdown(
                        f"""
                        <div class="source-card">
                            📄 {source}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # =================================================
                # RETRIEVED BUSINESS DATA
                # =================================================

                with st.expander(
                    "🔎 View Retrieved Business Data"
                ):

                    for result in results:

                        st.markdown(
                            f"### Rank {result.get('rank', 'N/A')}"
                        )

                        st.caption(
                            f"Source: {result.get('source', 'Unknown')} "
                            f"• Chunk: {result.get('chunk_id', 'N/A')} "
                            f"• Distance: "
                            f"{result.get('distance', 0):.4f}"
                        )

                        st.text(
                            result.get(
                                "content",
                                "No retrieved content available.",
                            )
                        )

                        st.divider()

            # =================================================
            # DATA SOURCE / NO SOURCES
            # =================================================

            else:

                if marketing_route:

                    st.info(
                        "This answer was generated from the "
                        "marketing analytics dataset."
                    )

                else:

                    st.info(
                        "No supporting sources were returned."
                    )

            # =================================================
            # SAVE CONVERSATION MEMORY
            # =================================================

            st.session_state.conversation_history.append(
                {
                    "question": question,
                    "resolved_question": contextual_question,
                    "answer": display_answer,
                    "referent": referent,
                    "is_marketing": marketing_route,
                }
            )

            st.session_state.conversation_history = (
                st.session_state.conversation_history[-10:]
            )


        except Exception as error:

            st.error(
                f"Something went wrong:\n\n{error}"
            )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation_history:

    st.divider()

    st.markdown("## 🧠 Conversation History")

    st.caption(
        "Recent questions are retained during this Streamlit session "
        "so follow-up questions can use previous context."
    )

    for item in reversed(st.session_state.conversation_history):

        with st.expander(item.get("question", "Previous question")):

            st.markdown(item.get("answer", ""))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightRAG • AI-Powered Business Intelligence • FAISS • Sentence Transformers • Groq"
)