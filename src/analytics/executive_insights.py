import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATH / ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# GROQ CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# SAFE NUMBER FORMATTERS
# ============================================================

def format_currency(value):
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value):
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


# ============================================================
# BUILD BUSINESS EVIDENCE
# ============================================================

def build_business_evidence(df, anomalies):
    """
    Build a compact, data-grounded evidence package
    for the AI Executive Business Reasoning layer.
    """

    evidence = []

    # --------------------------------------------------------
    # OVERALL KPIs
    # --------------------------------------------------------

    total_sales = (
        df["sales_amount"].sum()
        if "sales_amount" in df.columns
        else 0
    )

    total_profit = (
        df["profit"].sum()
        if "profit" in df.columns
        else 0
    )

    total_orders = (
        df["order_id"].nunique()
        if "order_id" in df.columns
        else 0
    )

    returned_items = 0

    if "is_returned" in df.columns:
        returned_items = df["is_returned"].sum()

    total_items = len(df)

    return_rate = (
        (returned_items / total_items) * 100
        if total_items > 0
        else 0
    )

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales != 0
        else 0
    )

    evidence.append("OVERALL BUSINESS KPIs")
    evidence.append(f"Total Sales: {format_currency(total_sales)}")
    evidence.append(f"Total Profit: {format_currency(total_profit)}")
    evidence.append(
        f"Overall Profit Margin: {format_percentage(profit_margin)}"
    )
    evidence.append(f"Total Orders: {total_orders:,}")
    evidence.append(
        f"Return Rate: {format_percentage(return_rate)}"
    )

    # --------------------------------------------------------
    # CATEGORY PERFORMANCE
    # --------------------------------------------------------

    if {
        "category",
        "sales_amount",
        "profit"
    }.issubset(df.columns):

        category_summary = (
            df.groupby("category")
            .agg(
                sales=("sales_amount", "sum"),
                profit=("profit", "sum")
            )
            .sort_values("profit", ascending=False)
        )

        evidence.append("")
        evidence.append("CATEGORY PERFORMANCE")

        for category, row in category_summary.iterrows():
            margin = (
                (row["profit"] / row["sales"]) * 100
                if row["sales"] != 0
                else 0
            )

            evidence.append(
                f"{category}: "
                f"Sales={format_currency(row['sales'])}, "
                f"Profit={format_currency(row['profit'])}, "
                f"Margin={format_percentage(margin)}"
            )

    # --------------------------------------------------------
    # MONTHLY PERFORMANCE
    # --------------------------------------------------------

    if {
        "year_month",
        "sales_amount",
        "profit"
    }.issubset(df.columns):

        monthly_summary = (
            df.groupby("year_month")
            .agg(
                sales=("sales_amount", "sum"),
                profit=("profit", "sum")
            )
            .sort_index()
        )

        evidence.append("")
        evidence.append("MONTHLY PERFORMANCE")

        for month, row in monthly_summary.iterrows():
            evidence.append(
                f"{month}: "
                f"Sales={format_currency(row['sales'])}, "
                f"Profit={format_currency(row['profit'])}"
            )

    # --------------------------------------------------------
    # TOP PRODUCTS BY PROFIT
    # --------------------------------------------------------

    if {
        "product_name",
        "profit"
    }.issubset(df.columns):

        top_profit_products = (
            df.groupby("product_name")["profit"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        evidence.append("")
        evidence.append("TOP 5 PRODUCTS BY PROFIT")

        for product, profit in top_profit_products.items():
            evidence.append(
                f"{product}: Profit={format_currency(profit)}"
            )

    # --------------------------------------------------------
    # HIGH RETURN PRODUCTS
    # --------------------------------------------------------

    if {
        "product_name",
        "is_returned"
    }.issubset(df.columns):

        product_returns = (
            df.groupby("product_name")
            .agg(
                total_items=("product_name", "size"),
                returned_items=("is_returned", "sum")
            )
        )

        product_returns["return_rate"] = (
            product_returns["returned_items"]
            / product_returns["total_items"]
            * 100
        )

        top_returns = (
            product_returns
            .sort_values("return_rate", ascending=False)
            .head(5)
        )

        evidence.append("")
        evidence.append("TOP 5 PRODUCTS BY RETURN RATE")

        for product, row in top_returns.iterrows():
            evidence.append(
                f"{product}: "
                f"Return Rate={format_percentage(row['return_rate'])}"
            )

    # --------------------------------------------------------
    # DISCOUNT ANALYSIS
    # --------------------------------------------------------

    if {
        "discount_percentage",
        "profit_margin_percentage"
    }.issubset(df.columns):

        discount_summary = (
            df.groupby("discount_percentage")
            ["profit_margin_percentage"]
            .mean()
            .sort_index()
        )

        if not discount_summary.empty:

            lowest_discount = discount_summary.index.min()
            highest_discount = discount_summary.index.max()

            lowest_margin = discount_summary.loc[
                lowest_discount
            ]

            highest_margin = discount_summary.loc[
                highest_discount
            ]

            evidence.append("")
            evidence.append("DISCOUNT / PROFITABILITY")

            evidence.append(
                f"Lowest discount level: "
                f"{lowest_discount}% → "
                f"Average margin={format_percentage(lowest_margin)}"
            )

            evidence.append(
                f"Highest discount level: "
                f"{highest_discount}% → "
                f"Average margin={format_percentage(highest_margin)}"
            )

    # --------------------------------------------------------
    # AUTOMATIC ANOMALIES
    # --------------------------------------------------------

    evidence.append("")
    evidence.append("AUTOMATICALLY DETECTED BUSINESS ANOMALIES")

    if anomalies:

        for anomaly in anomalies:

            priority = anomaly.get(
                "priority",
                anomaly.get("severity", "UNKNOWN")
            )

            anomaly_type = anomaly.get(
                "type",
                anomaly.get("title", "Business Anomaly")
            )

            description = anomaly.get(
                "description",
                anomaly.get("message", "")
            )

            action = anomaly.get(
                "action",
                anomaly.get("recommended_action", "")
            )

            evidence.append(
                f"[{priority}] {anomaly_type}: {description}"
            )

            if action:
                evidence.append(
                    f"Recommended investigation: {action}"
                )

    return "\n".join(evidence)


# ============================================================
# AI EXECUTIVE REASONING
# ============================================================

def generate_executive_insights(df, anomalies):
    """
    Generate an executive-level business interpretation
    using only the supplied analytics and anomaly evidence.
    """

    evidence = build_business_evidence(
        df,
        anomalies
    )

    system_prompt = """
You are the Executive Business Analyst for InsightRAG.

Your job is to analyze ONLY the business evidence provided
to you.

STRICT GROUNDING RULES:

1. Never invent facts, numbers, causes, trends, or explanations.
2. Do not claim that one factor caused another unless the evidence
   explicitly establishes that relationship.
3. Clearly distinguish between:
   - What the data shows
   - What should be investigated
4. If the evidence is insufficient to determine a cause, explicitly
   say that further investigation is required.
5. Recommendations must be directly connected to the evidence.
6. Do not introduce external business knowledge as if it were
   part of the dataset.
7. Do not exaggerate.
8. Keep the analysis useful for a business decision-maker.

Return the answer using EXACTLY this structure:

EXECUTIVE SUMMARY

2-4 concise sentences describing the most important business
situation supported by the data.

KEY BUSINESS FINDINGS

• Finding 1
• Finding 2
• Finding 3
• Finding 4

TOP PRIORITIES

1. Priority
   Why it matters: ...

2. Priority
   Why it matters: ...

3. Priority
   Why it matters: ...

INVESTIGATION AREAS

• Investigation area
• Investigation area
• Investigation area

IMPORTANT:
Do not say that an investigation area is the confirmed cause
of an issue unless the evidence explicitly proves it.
"""


    user_prompt = f"""
Analyze the following InsightRAG business evidence.

================ BUSINESS EVIDENCE ================

{evidence}

=====================================================

Generate the executive business reasoning report.
Use only the evidence above.
"""


    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "Unable to generate executive insights.\n\n"
            f"Error: {str(e)}"
        )