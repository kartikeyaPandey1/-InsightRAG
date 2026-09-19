import os
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

KNOWLEDGE_BASE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge_base"
)


# Create knowledge base folder if it doesn't exist
os.makedirs(
    KNOWLEDGE_BASE_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading processed datasets...\n")


master_df = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "master_analytics_dataset.csv"
    )
)

products_df = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "products_cleaned.csv"
    )
)

returns_df = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "returns_cleaned.csv"
    )
)

campaigns_df = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "marketing_campaigns_cleaned.csv"
    )
)


print("Datasets loaded successfully!")

print(f"Master Dataset: {master_df.shape}")
print(f"Products Dataset: {products_df.shape}")
print(f"Returns Dataset: {returns_df.shape}")
print(f"Campaigns Dataset: {campaigns_df.shape}")


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_document(filename, content):

    file_path = os.path.join(
        KNOWLEDGE_BASE_DIR,
        filename
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)

    print(
        f"Created: {filename}"
    )


# ============================================================
# 1. OVERALL BUSINESS SUMMARY
# ============================================================

print("\nGenerating overall business summary...")


total_sales = master_df[
    "sales_amount"
].sum()

total_cost = master_df[
    "total_cost"
].sum()

total_profit = master_df[
    "profit"
].sum()

total_quantity = master_df[
    "quantity"
].sum()

total_orders = master_df[
    "order_id"
].nunique()

total_customers = master_df[
    "customer_id"
].nunique()


profit_margin = (
    total_profit
    / total_sales
    * 100
)


overall_summary = f"""
INSIGHTRAG OVERALL BUSINESS SUMMARY

The InsightRAG business dataset contains transactional,
customer, product, return, and marketing information.

Overall Business Performance:

Total Sales: ₹{total_sales:,.2f}

Total Cost: ₹{total_cost:,.2f}

Total Profit: ₹{total_profit:,.2f}

Total Quantity Sold: {total_quantity:,}

Total Orders: {total_orders:,}

Total Unique Customers: {total_customers:,}

Overall Profit Margin: {profit_margin:.2f}%

The business generated approximately ₹{total_sales:,.2f}
in sales and ₹{total_profit:,.2f} in profit.

The overall profit margin was {profit_margin:.2f}%.
"""


save_document(
    "overall_business_summary.txt",
    overall_summary
)


# ============================================================
# 2. CATEGORY PERFORMANCE
# ============================================================

print("Generating category performance analysis...")


category_performance = (
    master_df
    .groupby("category")
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        ),
        total_quantity=(
            "quantity",
            "sum"
        )
    )
)


category_performance[
    "profit_margin"
] = (
    category_performance[
        "total_profit"
    ]
    /
    category_performance[
        "total_sales"
    ]
    * 100
)


category_document = """
CATEGORY PERFORMANCE ANALYSIS

"""


for category, row in category_performance.iterrows():

    category_document += f"""
Category: {category}

Total Sales: ₹{row['total_sales']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

Total Quantity Sold: {row['total_quantity']:,.0f}

Profit Margin: {row['profit_margin']:.2f}%

"""


best_sales_category = (
    category_performance[
        "total_sales"
    ].idxmax()
)

best_profit_category = (
    category_performance[
        "total_profit"
    ].idxmax()
)


category_document += f"""
KEY CATEGORY INSIGHTS

The category with the highest sales is
{best_sales_category}.

The category generating the highest total profit is
{best_profit_category}.
"""


save_document(
    "category_performance.txt",
    category_document
)


# ============================================================
# 3. PRODUCT PERFORMANCE
# ============================================================

print("Generating product performance analysis...")


product_performance = (
    master_df
    .groupby(
        [
            "product_id",
            "product_name",
            "category"
        ]
    )
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        ),
        total_quantity=(
            "quantity",
            "sum"
        )
    )
    .reset_index()
)


top_products = (
    product_performance
    .sort_values(
        "total_sales",
        ascending=False
    )
    .head(10)
)


product_document = """
TOP PRODUCT PERFORMANCE ANALYSIS

The following products generated the highest sales.

"""


for index, row in top_products.iterrows():

    product_document += f"""
Product: {row['product_name']}

Product ID: {row['product_id']}

Category: {row['category']}

Total Sales: ₹{row['total_sales']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

Quantity Sold: {row['total_quantity']:,.0f}

"""


save_document(
    "product_performance.txt",
    product_document
)


# ============================================================
# 4. MONTHLY SALES AND PROFIT TRENDS
# ============================================================

print("Generating monthly trends...")


master_df[
    "order_date"
] = pd.to_datetime(
    master_df[
        "order_date"
    ]
)


master_df[
    "year_month"
] = master_df[
    "order_date"
].dt.to_period(
    "M"
)


monthly_performance = (
    master_df
    .groupby(
        "year_month"
    )
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        )
    )
)


monthly_document = """
MONTHLY SALES AND PROFIT ANALYSIS

"""


for month, row in monthly_performance.iterrows():

    monthly_document += f"""
Month: {month}

Total Sales: ₹{row['total_sales']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

"""


best_sales_month = (
    monthly_performance[
        "total_sales"
    ].idxmax()
)


best_profit_month = (
    monthly_performance[
        "total_profit"
    ].idxmax()
)


monthly_document += f"""
KEY MONTHLY INSIGHTS

The month with the highest sales was
{best_sales_month}.

The month with the highest profit was
{best_profit_month}.
"""


save_document(
    "monthly_trends.txt",
    monthly_document
)


# ============================================================
# 5. ELECTRONICS PROFITABILITY ISSUE
# ============================================================

print(
    "Generating Electronics profitability analysis..."
)


master_df[
    "quarter"
] = master_df[
    "order_date"
].dt.to_period(
    "Q"
)


electronics_df = master_df[
    master_df[
        "category"
    ]
    == "Electronics"
]


electronics_quarterly = (
    electronics_df
    .groupby(
        "quarter"
    )
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_cost=(
            "total_cost",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        )
    )
)


electronics_quarterly[
    "profit_margin"
] = (
    electronics_quarterly[
        "total_profit"
    ]
    /
    electronics_quarterly[
        "total_sales"
    ]
    * 100
)


electronics_document = """
ELECTRONICS PROFITABILITY ANALYSIS

"""


for quarter, row in electronics_quarterly.iterrows():

    electronics_document += f"""
Quarter: {quarter}

Total Sales: ₹{row['total_sales']:,.2f}

Total Cost: ₹{row['total_cost']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

Profit Margin: {row['profit_margin']:.2f}%

"""


q3_2025 = electronics_quarterly.loc[
    "2025Q3"
]


electronics_document += f"""
CRITICAL BUSINESS INSIGHT

During Q3 2025, the Electronics category experienced
a significant decline in profitability.

Electronics generated sales of approximately
₹{q3_2025['total_sales']:,.2f} during Q3 2025.

However, the profit margin declined to only
{q3_2025['profit_margin']:.2f}%.

Compared with the typical Electronics profit margin
of approximately 25% to 28% during other quarters,
Q3 2025 represents a significant profitability issue.

This indicates that revenue remained relatively strong,
but increased costs or reduced margins significantly
affected profitability.
"""


save_document(
    "electronics_profitability_issue.txt",
    electronics_document
)


# ============================================================
# 6. CLOTHING DISCOUNT TRAP
# ============================================================

print(
    "Generating Clothing discount analysis..."
)


clothing_df = master_df[
    master_df[
        "category"
    ]
    == "Clothing"
].copy()


clothing_df[
    "year_month"
] = clothing_df[
    "order_date"
].dt.to_period(
    "M"
)


clothing_monthly = (
    clothing_df
    .groupby(
        "year_month"
    )
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        ),
        average_discount=(
            "discount_percentage",
            "mean"
        )
    )
)


clothing_monthly[
    "profit_margin"
] = (
    clothing_monthly[
        "total_profit"
    ]
    /
    clothing_monthly[
        "total_sales"
    ]
    * 100
)


clothing_document = """
CLOTHING DISCOUNT AND PROFITABILITY ANALYSIS

"""


for month, row in clothing_monthly.iterrows():

    clothing_document += f"""
Month: {month}

Total Sales: ₹{row['total_sales']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

Average Discount: {row['average_discount']:.2f}%

Profit Margin: {row['profit_margin']:.2f}%

"""


november_2025 = clothing_monthly.loc[
    "2025-11"
]

december_2025 = clothing_monthly.loc[
    "2025-12"
]


clothing_document += f"""
DISCOUNT TRAP BUSINESS INSIGHT

During November and December 2025,
the Clothing category experienced a significant increase
in average discounts.

November 2025:

Average Discount:
{november_2025['average_discount']:.2f}%

Profit Margin:
{november_2025['profit_margin']:.2f}%

December 2025:

Average Discount:
{december_2025['average_discount']:.2f}%

Profit Margin:
{december_2025['profit_margin']:.2f}%

The higher discount levels were associated with
a significant decline in profit margins.

This demonstrates a discount trap scenario where
increasing discounts can support sales activity but
significantly reduce business profitability.
"""


save_document(
    "clothing_discount_trap.txt",
    clothing_document
)


# ============================================================
# 7. RETURNS ANALYSIS
# ============================================================

print(
    "Generating returns analysis..."
)


items_sold = (
    master_df
    .groupby(
        "product_id"
    )
    .size()
    .reset_index(
        name="items_sold"
    )
)


product_returns = (
    returns_df
    .groupby(
        "product_id"
    )
    .size()
    .reset_index(
        name="returns"
    )
)


return_analysis = (
    items_sold
    .merge(
        product_returns,
        on="product_id",
        how="left"
    )
)


return_analysis[
    "returns"
] = return_analysis[
    "returns"
].fillna(0)


return_analysis[
    "return_rate"
] = (
    return_analysis[
        "returns"
    ]
    /
    return_analysis[
        "items_sold"
    ]
    * 100
)


return_analysis = (
    return_analysis
    .merge(
        products_df[
            [
                "product_id",
                "product_name",
                "category"
            ]
        ],
        on="product_id",
        how="left"
    )
)


top_returns = (
    return_analysis
    .sort_values(
        "return_rate",
        ascending=False
    )
    .head(10)
)


returns_document = """
RETURNS ANALYSIS

The following products have the highest return rates.

"""


for index, row in top_returns.iterrows():

    returns_document += f"""
Product: {row['product_name']}

Product ID: {row['product_id']}

Category: {row['category']}

Items Sold: {row['items_sold']:,.0f}

Returns: {row['returns']:,.0f}

Return Rate:
{row['return_rate']:.2f}%

"""


save_document(
    "returns_analysis.txt",
    returns_document
)


# ============================================================
# 8. CUSTOMER SEGMENT ANALYSIS
# ============================================================

print(
    "Generating customer analysis..."
)


customer_analysis = (
    master_df
    .groupby(
        "customer_segment"
    )
    .agg(
        total_sales=(
            "sales_amount",
            "sum"
        ),
        total_profit=(
            "profit",
            "sum"
        ),
        total_quantity=(
            "quantity",
            "sum"
        ),
        unique_customers=(
            "customer_id",
            "nunique"
        )
    )
)


customer_analysis[
    "profit_margin"
] = (
    customer_analysis[
        "total_profit"
    ]
    /
    customer_analysis[
        "total_sales"
    ]
    * 100
)


customer_document = """
CUSTOMER SEGMENT ANALYSIS

"""


for segment, row in customer_analysis.iterrows():

    customer_document += f"""
Customer Segment: {segment}

Total Sales: ₹{row['total_sales']:,.2f}

Total Profit: ₹{row['total_profit']:,.2f}

Quantity Purchased: {row['total_quantity']:,.0f}

Unique Customers: {row['unique_customers']:,.0f}

Profit Margin: {row['profit_margin']:.2f}%

"""


save_document(
    "customer_analysis.txt",
    customer_document
)


# ============================================================
# 9. MARKETING CAMPAIGN ANALYSIS
# ============================================================

print(
    "Generating marketing campaign analysis..."
)


campaign_analysis = (
    campaigns_df
    .groupby(
        "channel"
    )
    .agg(
        total_campaign_cost=(
            "campaign_cost",
            "sum"
        ),
        total_revenue=(
            "revenue_generated",
            "sum"
        ),
        average_roi=(
            "roi_percentage",
            "mean"
        ),
        campaign_count=(
            "campaign_id",
            "count"
        )
    )
)


marketing_document = """
MARKETING CAMPAIGN PERFORMANCE ANALYSIS

"""


for channel, row in campaign_analysis.iterrows():

    marketing_document += f"""
Marketing Channel: {channel}

Total Campaign Cost:
₹{row['total_campaign_cost']:,.2f}

Total Revenue Generated:
₹{row['total_revenue']:,.2f}

Average ROI:
{row['average_roi']:.2f}%

Number of Campaigns:
{row['campaign_count']:.0f}

"""


best_channel = (
    campaign_analysis[
        "average_roi"
    ].idxmax()
)


marketing_document += f"""
KEY MARKETING INSIGHT

The marketing channel with the highest average ROI is
{best_channel}.
"""


save_document(
    "marketing_analysis.txt",
    marketing_document
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "KNOWLEDGE BASE GENERATION COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print(
    f"\nKnowledge Base Location:\n"
    f"{KNOWLEDGE_BASE_DIR}"
)

print("\nGenerated Documents:")


for file_name in sorted(
    os.listdir(
        KNOWLEDGE_BASE_DIR
    )
):

    print(
        f"- {file_name}"
    )