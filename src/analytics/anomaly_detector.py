import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_analytics_dataset.csv"
)


# ============================================================
# LOAD BUSINESS DATA
# ============================================================

def load_business_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Business dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Convert date
    if "order_date" in df.columns:

        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

    # Numeric columns
    numeric_columns = [
        "sales_amount",
        "total_cost",
        "profit",
        "profit_margin_percentage",
        "quantity",
        "discount_percentage",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# CATEGORY QUARTERLY ANOMALIES
# ============================================================

def detect_category_margin_anomalies(df):

    required_columns = [
        "category",
        "order_year",
        "order_quarter",
        "sales_amount",
        "total_cost",
        "profit"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        return []

    grouped = (
        df
        .groupby(
            [
                "category",
                "order_year",
                "order_quarter"
            ]
        )
        .agg(
            sales=("sales_amount", "sum"),
            cost=("total_cost", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
    )

    grouped["margin"] = (
        grouped["profit"]
        / grouped["sales"]
        * 100
    )

    anomalies = []

    for category in grouped["category"].dropna().unique():

        category_data = (
            grouped[
                grouped["category"] == category
            ]
            .sort_values(
                [
                    "order_year",
                    "order_quarter"
                ]
            )
            .reset_index(drop=True)
        )

        if len(category_data) < 3:
            continue

        for i in range(1, len(category_data) - 1):

            previous = category_data.iloc[i - 1]
            current = category_data.iloc[i]
            following = category_data.iloc[i + 1]

            surrounding_margin = (
                previous["margin"]
                + following["margin"]
            ) / 2

            margin_difference = (
                current["margin"]
                - surrounding_margin
            )

            # Detect significant margin deterioration
            if margin_difference <= -8:

                anomalies.append({

                    "type": "Profit Margin Anomaly",

                    "severity": "HIGH",

                    "category": category,

                    "period": (
                        f"{current['order_quarter']} "
                        f"{int(current['order_year'])}"
                    ),

                    "metric": "Profit Margin",

                    "value": round(
                        current["margin"],
                        2
                    ),

                    "comparison": round(
                        surrounding_margin,
                        2
                    ),

                    "change": round(
                        margin_difference,
                        2
                    ),

                    "message": (
                        f"{category} profit margin dropped "
                        f"to {current['margin']:.2f}% in "
                        f"{current['order_quarter']} "
                        f"{int(current['order_year'])}, "
                        f"compared with approximately "
                        f"{surrounding_margin:.2f}% in the "
                        f"surrounding quarters."
                    ),

                    "action": (
                        f"Investigate the drivers of the "
                        f"{category} margin decline during "
                        f"{current['order_quarter']} "
                        f"{int(current['order_year'])}."
                    )
                })

    return anomalies


# ============================================================
# MONTHLY PROFIT ANOMALIES
# ============================================================

def detect_monthly_profit_anomalies(df):

    required_columns = [
        "order_date",
        "sales_amount",
        "profit"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        return []

    monthly = (
        df
        .dropna(subset=["order_date"])
        .groupby(
            df["order_date"].dt.to_period("M")
        )
        .agg(
            sales=("sales_amount", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
    )

    monthly["profit_margin"] = (
        monthly["profit"]
        / monthly["sales"]
        * 100
    )

    anomalies = []

    for i in range(1, len(monthly)):

        previous = monthly.iloc[i - 1]
        current = monthly.iloc[i]

        if previous["profit"] == 0:
            continue

        profit_change = (
            (
                current["profit"]
                - previous["profit"]
            )
            / abs(previous["profit"])
        ) * 100

        # Significant monthly profit decline
        if profit_change <= -25:

            period = str(
                current["order_date"]
            )

            anomalies.append({

                "type": "Monthly Profit Drop",

                "severity": "MEDIUM",

                "category": "Overall Business",

                "period": period,

                "metric": "Profit",

                "value": round(
                    current["profit"],
                    2
                ),

                "comparison": round(
                    previous["profit"],
                    2
                ),

                "change": round(
                    profit_change,
                    2
                ),

                "message": (
                    f"Overall profit declined by "
                    f"{abs(profit_change):.1f}% "
                    f"from {previous['order_date']} "
                    f"to {current['order_date']}."
                ),

                "action": (
                    "Investigate the factors behind "
                    "the monthly profit decline."
                )
            })

    return anomalies


# ============================================================
# PRODUCT RETURN ANOMALIES
# ============================================================

def detect_return_anomalies(df):

    required_columns = [
        "product_name",
        "is_returned"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        return []

    product_returns = (
        df
        .groupby("product_name")
        .agg(
            total_items=("product_name", "count"),
            returned_items=("is_returned", "sum")
        )
        .reset_index()
    )

    product_returns["return_rate"] = (
        product_returns["returned_items"]
        / product_returns["total_items"]
        * 100
    )

    anomalies = []

    # High-return products
    high_return_products = (
        product_returns[
            product_returns["return_rate"] >= 25
        ]
        .sort_values(
            "return_rate",
            ascending=False
        )
        .head(5)
    )

    for _, row in high_return_products.iterrows():

        anomalies.append({

            "type": "High Return Rate",

            "severity": "HIGH",

            "category": "Product Returns",

            "period": "Overall",

            "metric": "Return Rate",

            "value": round(
                row["return_rate"],
                2
            ),

            "comparison": 25.0,

            "change": round(
                row["return_rate"] - 25,
                2
            ),

            "message": (
                f"{row['product_name']} has a return "
                f"rate of {row['return_rate']:.2f}%."
            ),

            "action": (
                f"Investigate the return reasons and "
                f"customer issues associated with "
                f"{row['product_name']}."
            )
        })

    return anomalies


# ============================================================
# DISCOUNT / PROFIT ANOMALIES
# ============================================================

def detect_discount_profit_anomalies(df):

    required_columns = [
        "discount_percentage",
        "profit_margin_percentage"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        return []

    discount_data = (
        df
        .groupby("discount_percentage")
        .agg(
            average_margin=(
                "profit_margin_percentage",
                "mean"
            ),
            total_profit=("profit", "sum"),
            sales=("sales_amount", "sum")
        )
        .reset_index()
        .sort_values(
            "discount_percentage"
        )
    )

    anomalies = []

    if len(discount_data) < 2:
        return anomalies

    highest_discount = (
        discount_data
        .sort_values(
            "discount_percentage",
            ascending=False
        )
        .iloc[0]
    )

    lowest_discount = (
        discount_data
        .sort_values(
            "discount_percentage"
        )
        .iloc[0]
    )

    margin_difference = (
        highest_discount["average_margin"]
        - lowest_discount["average_margin"]
    )

    if margin_difference <= -5:

        anomalies.append({

            "type": "Discount Profit Risk",

            "severity": "MEDIUM",

            "category": "Pricing",

            "period": "Overall",

            "metric": "Average Profit Margin",

            "value": round(
                highest_discount["average_margin"],
                2
            ),

            "comparison": round(
                lowest_discount["average_margin"],
                2
            ),

            "change": round(
                margin_difference,
                2
            ),

            "message": (
                f"Higher discount levels are associated "
                f"with lower average profit margins. "
                f"The difference between the highest and "
                f"lowest discount levels is "
                f"{abs(margin_difference):.2f} percentage "
                f"points."
            ),

            "action": (
                "Review discount levels and their impact "
                "on profitability."
            )
        })

    return anomalies


# ============================================================
# RUN ALL ANOMALY DETECTION
# ============================================================

def detect_all_anomalies():

    df = load_business_data()

    df = prepare_data(df)

    anomalies = []

    anomalies.extend(
        detect_category_margin_anomalies(df)
    )

    anomalies.extend(
        detect_monthly_profit_anomalies(df)
    )

    anomalies.extend(
        detect_return_anomalies(df)
    )

    anomalies.extend(
        detect_discount_profit_anomalies(df)
    )

    # Severity ordering
    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    anomalies.sort(
        key=lambda item:
        severity_order.get(
            item.get("severity", "LOW"),
            2
        )
    )

    return anomalies


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "INSIGHTRAG - AUTOMATIC ANOMALY DETECTION"
    )

    print("=" * 70)

    anomalies = detect_all_anomalies()

    print(
        f"\nTotal anomalies detected: "
        f"{len(anomalies)}"
    )

    for i, anomaly in enumerate(
        anomalies,
        start=1
    ):

        print(
            f"\n{i}. "
            f"[{anomaly['severity']}] "
            f"{anomaly['type']}"
        )

        print(
            f"   {anomaly['message']}"
        )

        print(
            f"   Action: "
            f"{anomaly['action']}"
        )