# ============================================================
# INSIGHTRAG - MARKETING ANALYTICS
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MARKETING_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "marketing_campaigns_cleaned.csv"
)


# ============================================================
# LOAD MARKETING DATA
# ============================================================

def load_marketing_data():
    """
    Load the cleaned marketing campaigns dataset.
    """

    if not MARKETING_FILE.exists():
        raise FileNotFoundError(
            f"Marketing dataset not found: {MARKETING_FILE}"
        )

    df = pd.read_csv(MARKETING_FILE)

    # --------------------------------------------------------
    # Convert date columns
    # --------------------------------------------------------

    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(
            df["start_date"],
            errors="coerce"
        )

    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(
            df["end_date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Ensure numeric columns are numeric
    # --------------------------------------------------------

    numeric_columns = [
        "campaign_cost",
        "revenue_generated",
        "roi_percentage",
        "campaign_duration_days",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# MARKETING KPIs
# ============================================================

def calculate_marketing_kpis(df):
    """
    Calculate overall marketing performance KPIs.
    """

    total_campaigns = len(df)

    total_cost = (
        df["campaign_cost"].sum()
        if "campaign_cost" in df.columns
        else 0
    )

    total_revenue = (
        df["revenue_generated"].sum()
        if "revenue_generated" in df.columns
        else 0
    )

    # Overall ROI based on total cost and total revenue.
    #
    # ROI = ((Revenue - Cost) / Cost) * 100
    #
    # This is calculated from aggregated values rather than
    # averaging individual campaign ROI percentages.

    overall_roi = (
        ((total_revenue - total_cost) / total_cost) * 100
        if total_cost != 0
        else 0
    )

    average_roi = (
        df["roi_percentage"].mean()
        if "roi_percentage" in df.columns
        else 0
    )

    average_duration = (
        df["campaign_duration_days"].mean()
        if "campaign_duration_days" in df.columns
        else 0
    )

    return {
        "total_campaigns": total_campaigns,
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "overall_roi": overall_roi,
        "average_roi": average_roi,
        "average_duration": average_duration,
    }


# ============================================================
# CHANNEL PERFORMANCE
# ============================================================

def channel_performance(df):
    """
    Calculate marketing performance by channel.
    """

    required_columns = {
        "channel",
        "campaign_cost",
        "revenue_generated",
    }

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    summary = (
        df.groupby("channel")
        .agg(
            campaigns=("campaign_id", "count"),
            total_cost=("campaign_cost", "sum"),
            total_revenue=("revenue_generated", "sum"),
            average_roi=("roi_percentage", "mean"),
        )
        .reset_index()
    )

    summary["overall_roi"] = (
        (
            summary["total_revenue"]
            - summary["total_cost"]
        )
        / summary["total_cost"]
        * 100
    )

    summary = summary.sort_values(
        "overall_roi",
        ascending=False
    )

    return summary


# ============================================================
# PERFORMANCE TYPE SUMMARY
# ============================================================

def performance_type_summary(df):
    """
    Summarize campaigns by their performance classification.
    """

    required_columns = {
        "performance_type",
        "campaign_cost",
        "revenue_generated",
    }

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    summary = (
        df.groupby("performance_type")
        .agg(
            campaigns=("campaign_id", "count"),
            total_cost=("campaign_cost", "sum"),
            total_revenue=("revenue_generated", "sum"),
            average_roi=("roi_percentage", "mean"),
        )
        .reset_index()
    )

    summary["overall_roi"] = (
        (
            summary["total_revenue"]
            - summary["total_cost"]
        )
        / summary["total_cost"]
        * 100
    )

    return summary.sort_values(
        "overall_roi",
        ascending=False
    )


# ============================================================
# TOP CAMPAIGNS BY ROI
# ============================================================

def top_campaigns_by_roi(df, n=10):
    """
    Return campaigns with the highest ROI.
    """

    if "roi_percentage" not in df.columns:
        return pd.DataFrame()

    columns = [
        "campaign_name",
        "channel",
        "campaign_cost",
        "revenue_generated",
        "roi_percentage",
        "performance_type",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return (
        df[available_columns]
        .sort_values(
            "roi_percentage",
            ascending=False
        )
        .head(n)
    )


# ============================================================
# LOWEST PERFORMING CAMPAIGNS
# ============================================================

def lowest_campaigns_by_roi(df, n=10):
    """
    Return campaigns with the lowest ROI.
    """

    if "roi_percentage" not in df.columns:
        return pd.DataFrame()

    columns = [
        "campaign_name",
        "channel",
        "campaign_cost",
        "revenue_generated",
        "roi_percentage",
        "performance_type",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return (
        df[available_columns]
        .sort_values(
            "roi_percentage",
            ascending=True
        )
        .head(n)
    )


# ============================================================
# COST VS REVENUE DATA
# ============================================================

def campaign_cost_revenue(df):
    """
    Return campaign-level cost and revenue data
    for comparison charts.
    """

    columns = [
        "campaign_name",
        "campaign_cost",
        "revenue_generated",
        "roi_percentage",
        "channel",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[available_columns].copy()


# ============================================================
# CHANNEL ROI RANKING
# ============================================================

def best_marketing_channel(df):
    """
    Return the channel with the highest aggregated ROI.
    """

    summary = channel_performance(df)

    if summary.empty:
        return None

    return summary.iloc[0]


# ============================================================
# MARKETING INSIGHT SUMMARY
# ============================================================

def generate_marketing_summary(df):
    """
    Generate simple data-backed marketing observations.
    """

    kpis = calculate_marketing_kpis(df)
    channels = channel_performance(df)
    top_campaigns = top_campaigns_by_roi(df, 3)
    low_campaigns = lowest_campaigns_by_roi(df, 3)

    summary = []

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    summary.append(
        f"Total marketing spend: "
        f"₹{kpis['total_cost']:,.2f}"
    )

    summary.append(
        f"Total revenue generated: "
        f"₹{kpis['total_revenue']:,.2f}"
    )

    summary.append(
        f"Overall marketing ROI: "
        f"{kpis['overall_roi']:.2f}%"
    )

    # --------------------------------------------------------
    # Best channel
    # --------------------------------------------------------

    if not channels.empty:

        best_channel = channels.iloc[0]

        summary.append(
            f"Highest aggregated ROI channel: "
            f"{best_channel['channel']} "
            f"({best_channel['overall_roi']:.2f}%)"
        )

    # --------------------------------------------------------
    # Top campaigns
    # --------------------------------------------------------

    if not top_campaigns.empty:

        summary.append("")
        summary.append("Top campaigns by ROI:")

        for _, row in top_campaigns.iterrows():

            summary.append(
                f"- {row['campaign_name']}: "
                f"{row['roi_percentage']:.2f}% ROI"
            )

    # --------------------------------------------------------
    # Lowest campaigns
    # --------------------------------------------------------

    if not low_campaigns.empty:

        summary.append("")
        summary.append("Lowest campaigns by ROI:")

        for _, row in low_campaigns.iterrows():

            summary.append(
                f"- {row['campaign_name']}: "
                f"{row['roi_percentage']:.2f}% ROI"
            )

    return "\n".join(summary)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INSIGHTRAG - MARKETING ANALYTICS")
    print("=" * 70)

    marketing_df = load_marketing_data()

    print(
        f"\nTotal campaigns: {len(marketing_df)}"
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpis = calculate_marketing_kpis(
        marketing_df
    )

    print("\nMARKETING KPIs")
    print("-" * 70)

    print(
        f"Total Campaigns: "
        f"{kpis['total_campaigns']}"
    )

    print(
        f"Total Cost: "
        f"₹{kpis['total_cost']:,.2f}"
    )

    print(
        f"Total Revenue: "
        f"₹{kpis['total_revenue']:,.2f}"
    )

    print(
        f"Overall ROI: "
        f"{kpis['overall_roi']:.2f}%"
    )

    print(
        f"Average Campaign ROI: "
        f"{kpis['average_roi']:.2f}%"
    )

    print(
        f"Average Campaign Duration: "
        f"{kpis['average_duration']:.2f} days"
    )

    # --------------------------------------------------------
    # CHANNELS
    # --------------------------------------------------------

    print("\nCHANNEL PERFORMANCE")
    print("-" * 70)

    print(
        channel_performance(
            marketing_df
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # PERFORMANCE TYPES
    # --------------------------------------------------------

    print("\nPERFORMANCE TYPE SUMMARY")
    print("-" * 70)

    print(
        performance_type_summary(
            marketing_df
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # TOP CAMPAIGNS
    # --------------------------------------------------------

    print("\nTOP CAMPAIGNS BY ROI")
    print("-" * 70)

    print(
        top_campaigns_by_roi(
            marketing_df,
            10
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # LOWEST CAMPAIGNS
    # --------------------------------------------------------

    print("\nLOWEST CAMPAIGNS BY ROI")
    print("-" * 70)

    print(
        lowest_campaigns_by_roi(
            marketing_df,
            10
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nMARKETING INSIGHT SUMMARY")
    print("-" * 70)

    print(
        generate_marketing_summary(
            marketing_df
        )
    )