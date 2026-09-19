import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():

    print("\nLoading raw datasets...\n")

    customers_df = pd.read_csv(
        RAW_DATA_DIR / "customers.csv"
    )

    products_df = pd.read_csv(
        RAW_DATA_DIR / "products.csv"
    )

    orders_df = pd.read_csv(
        RAW_DATA_DIR / "orders.csv"
    )

    order_items_df = pd.read_csv(
        RAW_DATA_DIR / "order_items.csv"
    )

    returns_df = pd.read_csv(
        RAW_DATA_DIR / "returns.csv"
    )

    marketing_campaigns_df = pd.read_csv(
        RAW_DATA_DIR / "marketing_campaigns.csv"
    )

    print("Raw datasets loaded successfully!")

    return (
        customers_df,
        products_df,
        orders_df,
        order_items_df,
        returns_df,
        marketing_campaigns_df
    )


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def check_data_quality(dataset_name, dataframe):

    print("\n" + "=" * 60)
    print(f"DATA QUALITY CHECK: {dataset_name}")
    print("=" * 60)

    print(f"\nRows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    # Missing values
    missing_values = dataframe.isnull().sum()

    print("\nMissing Values:")

    missing_columns = missing_values[
        missing_values > 0
    ]

    if missing_columns.empty:

        print("No missing values found.")

    else:

        print(missing_columns)

    # Duplicate rows
    duplicate_rows = dataframe.duplicated().sum()

    print(
        f"\nDuplicate Rows: {duplicate_rows}"
    )

    # Data types
    print("\nData Types:")

    print(
        dataframe.dtypes
    )


# ============================================================
# CLEAN CUSTOMERS DATA
# ============================================================

def clean_customers(customers_df):

    print("\nCleaning Customers dataset...")

    df = customers_df.copy()

    # Convert date
    df["signup_date"] = pd.to_datetime(
        df["signup_date"],
        errors="coerce"
    )

    # Clean text columns
    text_columns = [
        "customer_id",
        "customer_name",
        "gender",
        "city",
        "state",
        "customer_segment"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    print(
        "Customers dataset cleaned successfully!"
    )

    return df


# ============================================================
# CLEAN PRODUCTS DATA
# ============================================================

def clean_products(products_df):

    print("\nCleaning Products dataset...")

    df = products_df.copy()

    # Clean text columns
    text_columns = [
        "product_id",
        "product_name",
        "category",
        "sub_category",
        "supplier"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Ensure numeric columns
    numeric_columns = [
        "cost_price",
        "selling_price"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Round monetary values
    df["cost_price"] = (
        df["cost_price"]
        .round(2)
    )

    df["selling_price"] = (
        df["selling_price"]
        .round(2)
    )

    print(
        "Products dataset cleaned successfully!"
    )

    return df


# ============================================================
# CLEAN ORDERS DATA
# ============================================================

def clean_orders(orders_df):

    print("\nCleaning Orders dataset...")

    df = orders_df.copy()

    # Convert order date
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    # Clean text columns
    text_columns = [
        "order_id",
        "customer_id",
        "payment_method",
        "order_status"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # CREATE TIME FEATURES
    # --------------------------------------------------------

    df["order_year"] = (
        df["order_date"].dt.year
    )

    df["order_month"] = (
        df["order_date"].dt.month
    )

    df["order_month_name"] = (
        df["order_date"].dt.month_name()
    )

    df["order_quarter"] = (
        "Q" +
        df["order_date"]
        .dt.quarter
        .astype(str)
    )

    df["year_month"] = (
        df["order_date"]
        .dt.to_period("M")
        .astype(str)
    )

    df["year_quarter"] = (
        df["order_date"]
        .dt.to_period("Q")
        .astype(str)
    )

    print(
        "Orders dataset cleaned successfully!"
    )

    return df


# ============================================================
# CLEAN ORDER ITEMS DATA
# ============================================================

def clean_order_items(order_items_df):

    print("\nCleaning Order Items dataset...")

    df = order_items_df.copy()

    # Clean ID columns
    id_columns = [
        "order_item_id",
        "order_id",
        "product_id"
    ]

    for column in id_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Convert numeric columns
    numeric_columns = [
        "quantity",
        "discount_percentage",
        "sales_amount",
        "total_cost",
        "profit"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # PROFIT MARGIN
    # --------------------------------------------------------

    df["profit_margin_percentage"] = (
        df["profit"]
        /
        df["sales_amount"]
        * 100
    )

    df["profit_margin_percentage"] = (
        df["profit_margin_percentage"]
        .round(2)
    )

    # --------------------------------------------------------
    # DISCOUNT CATEGORY
    # --------------------------------------------------------

    def get_discount_category(discount):

        if discount == 0:

            return "No Discount"

        elif discount <= 10:

            return "Low Discount"

        elif discount <= 20:

            return "Medium Discount"

        else:

            return "High Discount"

    df["discount_category"] = (
        df["discount_percentage"]
        .apply(get_discount_category)
    )

    # Round monetary values
    monetary_columns = [
        "sales_amount",
        "total_cost",
        "profit"
    ]

    for column in monetary_columns:

        df[column] = (
            df[column]
            .round(2)
        )

    print(
        "Order Items dataset cleaned successfully!"
    )

    return df


# ============================================================
# CLEAN RETURNS DATA
# ============================================================

def clean_returns(returns_df):

    print("\nCleaning Returns dataset...")

    df = returns_df.copy()

    # Convert return date
    df["return_date"] = pd.to_datetime(
        df["return_date"],
        errors="coerce"
    )

    # Clean text columns
    text_columns = [
        "return_id",
        "order_id",
        "product_id",
        "return_reason"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    print(
        "Returns dataset cleaned successfully!"
    )

    return df


# ============================================================
# CLEAN MARKETING CAMPAIGNS DATA
# ============================================================

def clean_marketing_campaigns(marketing_campaigns_df):

    print("\nCleaning Marketing Campaigns dataset...")

    df = marketing_campaigns_df.copy()

    # Convert dates
    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce"
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce"
    )

    # Clean text columns
    text_columns = [
        "campaign_id",
        "campaign_name",
        "channel",
        "performance_type"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Numeric columns
    numeric_columns = [
        "campaign_cost",
        "revenue_generated",
        "roi_percentage"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # CAMPAIGN DURATION
    # --------------------------------------------------------

    df["campaign_duration_days"] = (
        df["end_date"]
        -
        df["start_date"]
    ).dt.days

    # Round monetary values
    monetary_columns = [
        "campaign_cost",
        "revenue_generated",
        "roi_percentage"
    ]

    for column in monetary_columns:

        df[column] = (
            df[column]
            .round(2)
        )

    print(
        "Marketing Campaigns dataset cleaned successfully!"
    )

    return df


# ============================================================
# VALIDATE RELATIONSHIPS
# ============================================================

def validate_relationships(
    customers_df,
    products_df,
    orders_df,
    order_items_df,
    returns_df
):

    print("\n" + "=" * 60)
    print("RELATIONSHIP VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # ORDERS -> CUSTOMERS
    # --------------------------------------------------------

    invalid_customer_ids = (
        ~orders_df["customer_id"]
        .isin(customers_df["customer_id"])
    ).sum()

    print(
        f"\nInvalid Customer IDs in Orders: "
        f"{invalid_customer_ids}"
    )

    # --------------------------------------------------------
    # ORDER ITEMS -> ORDERS
    # --------------------------------------------------------

    invalid_order_ids = (
        ~order_items_df["order_id"]
        .isin(orders_df["order_id"])
    ).sum()

    print(
        f"Invalid Order IDs in Order Items: "
        f"{invalid_order_ids}"
    )

    # --------------------------------------------------------
    # ORDER ITEMS -> PRODUCTS
    # --------------------------------------------------------

    invalid_product_ids = (
        ~order_items_df["product_id"]
        .isin(products_df["product_id"])
    ).sum()

    print(
        f"Invalid Product IDs in Order Items: "
        f"{invalid_product_ids}"
    )

    # --------------------------------------------------------
    # RETURNS -> ORDERS
    # --------------------------------------------------------

    invalid_return_order_ids = (
        ~returns_df["order_id"]
        .isin(orders_df["order_id"])
    ).sum()

    print(
        f"Invalid Order IDs in Returns: "
        f"{invalid_return_order_ids}"
    )

    # --------------------------------------------------------
    # RETURNS -> PRODUCTS
    # --------------------------------------------------------

    invalid_return_product_ids = (
        ~returns_df["product_id"]
        .isin(products_df["product_id"])
    ).sum()

    print(
        f"Invalid Product IDs in Returns: "
        f"{invalid_return_product_ids}"
    )

    # --------------------------------------------------------
    # RETURNS -> ORDER PRODUCT COMBINATION
    # --------------------------------------------------------

    valid_combinations = set(
        zip(
            order_items_df["order_id"],
            order_items_df["product_id"]
        )
    )

    invalid_combinations = sum(

        (order_id, product_id)
        not in valid_combinations

        for order_id, product_id in zip(

            returns_df["order_id"],

            returns_df["product_id"]

        )

    )

    print(
        f"Invalid Order-Product Combinations "
        f"in Returns: {invalid_combinations}"
    )

    print("\nRelationship validation completed!")


# ============================================================
# CREATE MASTER ANALYTICAL DATASET
# ============================================================

def create_master_dataset(
    customers_df,
    products_df,
    orders_df,
    order_items_df,
    returns_df
):

    print(
        "\nCreating Master Analytical Dataset..."
    )

    # --------------------------------------------------------
    # STEP 1: START WITH ORDER ITEMS
    # --------------------------------------------------------

    master_df = order_items_df.copy()

    # --------------------------------------------------------
    # STEP 2: ADD ORDER INFORMATION
    # --------------------------------------------------------

    master_df = master_df.merge(

        orders_df,

        on="order_id",

        how="left"

    )

    # --------------------------------------------------------
    # STEP 3: ADD CUSTOMER INFORMATION
    # --------------------------------------------------------

    master_df = master_df.merge(

        customers_df,

        on="customer_id",

        how="left"

    )

    # --------------------------------------------------------
    # STEP 4: ADD PRODUCT INFORMATION
    # --------------------------------------------------------

    master_df = master_df.merge(

        products_df,

        on="product_id",

        how="left"

    )

    # --------------------------------------------------------
    # STEP 5: CREATE RETURN FLAG
    # --------------------------------------------------------
    #
    # We aggregate returns by order-product combination.
    # This prevents duplicate rows during the merge.
    # --------------------------------------------------------

    returns_summary = (

        returns_df

        .groupby(

            ["order_id", "product_id"],

            as_index=False

        )

        .agg(

            is_returned=(

                "return_id",

                lambda x: 1

            ),

            return_reason=(

                "return_reason",

                "first"

            ),

            return_date=(

                "return_date",

                "min"

            )

        )

    )

    # --------------------------------------------------------
    # STEP 6: MERGE RETURN INFORMATION
    # --------------------------------------------------------

    master_df = master_df.merge(

        returns_summary,

        on=[

            "order_id",

            "product_id"

        ],

        how="left"

    )

    # Fill missing return values

    master_df["is_returned"] = (

        master_df["is_returned"]

        .fillna(0)

        .astype(int)

    )

    # --------------------------------------------------------
    # RETURN RATE FLAGS
    # --------------------------------------------------------

    master_df["return_status"] = (

        master_df["is_returned"]

        .map({

            1: "Returned",

            0: "Not Returned"

        })

    )

    # --------------------------------------------------------
    # REORDER COLUMNS
    # --------------------------------------------------------

    important_columns = [

        # Order Item Information
        "order_item_id",
        "order_id",
        "product_id",

        # Customer Information
        "customer_id",
        "customer_name",
        "age",
        "gender",
        "city",
        "state",
        "customer_segment",
        "signup_date",

        # Product Information
        "product_name",
        "category",
        "sub_category",
        "supplier",
        "cost_price",
        "selling_price",

        # Order Information
        "order_date",
        "order_year",
        "order_month",
        "order_month_name",
        "order_quarter",
        "year_month",
        "year_quarter",
        "payment_method",
        "order_status",

        # Sales Metrics
        "quantity",
        "discount_percentage",
        "discount_category",
        "sales_amount",
        "total_cost",
        "profit",
        "profit_margin_percentage",

        # Return Information
        "is_returned",
        "return_status",
        "return_reason",
        "return_date"

    ]

    master_df = master_df[

        important_columns

    ]

    print(
        "Master Analytical Dataset created successfully!"
    )

    print(
        f"Master Dataset Shape: "
        f"{master_df.shape}"
    )

    return master_df


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    customers_df,
    products_df,
    orders_df,
    order_items_df,
    returns_df,
    marketing_campaigns_df,
    master_df
):

    print(
        "\nSaving processed datasets..."
    )

    # Create processed directory if it does not exist

    PROCESSED_DATA_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    # --------------------------------------------------------
    # SAVE DATASETS
    # --------------------------------------------------------

    customers_df.to_csv(

        PROCESSED_DATA_DIR /

        "customers_cleaned.csv",

        index=False

    )

    products_df.to_csv(

        PROCESSED_DATA_DIR /

        "products_cleaned.csv",

        index=False

    )

    orders_df.to_csv(

        PROCESSED_DATA_DIR /

        "orders_cleaned.csv",

        index=False

    )

    order_items_df.to_csv(

        PROCESSED_DATA_DIR /

        "order_items_cleaned.csv",

        index=False

    )

    returns_df.to_csv(

        PROCESSED_DATA_DIR /

        "returns_cleaned.csv",

        index=False

    )

    marketing_campaigns_df.to_csv(

        PROCESSED_DATA_DIR /

        "marketing_campaigns_cleaned.csv",

        index=False

    )

    master_df.to_csv(

        PROCESSED_DATA_DIR /

        "master_analytics_dataset.csv",

        index=False

    )

    print(
        "\nAll processed datasets saved successfully!"
    )

    print(
        f"\nSaved Location:\n"
        f"{PROCESSED_DATA_DIR}"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------------

    (

        customers_df,

        products_df,

        orders_df,

        order_items_df,

        returns_df,

        marketing_campaigns_df

    ) = load_raw_data()


    # --------------------------------------------------------
    # INITIAL DATA QUALITY CHECKS
    # --------------------------------------------------------

    datasets = [

        customers_df,

        products_df,

        orders_df,

        order_items_df,

        returns_df,

        marketing_campaigns_df

    ]

    dataset_names = [

        "Customers",

        "Products",

        "Orders",

        "Order Items",

        "Returns",

        "Marketing Campaigns"

    ]

    for name, dataframe in zip(

        dataset_names,

        datasets

    ):

        check_data_quality(

            name,

            dataframe

        )


    # --------------------------------------------------------
    # CLEAN DATASETS
    # --------------------------------------------------------

    customers_df = clean_customers(

        customers_df

    )

    products_df = clean_products(

        products_df

    )

    orders_df = clean_orders(

        orders_df

    )

    order_items_df = clean_order_items(

        order_items_df

    )

    returns_df = clean_returns(

        returns_df

    )

    marketing_campaigns_df = (

        clean_marketing_campaigns(

            marketing_campaigns_df

        )

    )


    # --------------------------------------------------------
    # VALIDATE RELATIONSHIPS
    # --------------------------------------------------------

    validate_relationships(

        customers_df,

        products_df,

        orders_df,

        order_items_df,

        returns_df

    )


    # --------------------------------------------------------
    # CREATE MASTER ANALYTICAL DATASET
    # --------------------------------------------------------

    master_df = create_master_dataset(

        customers_df,

        products_df,

        orders_df,

        order_items_df,

        returns_df

    )


    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    save_processed_data(

        customers_df,

        products_df,

        orders_df,

        order_items_df,

        returns_df,

        marketing_campaigns_df,

        master_df

    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")

    print("=" * 60)

    print(
        f"\nCustomers: {customers_df.shape}"
    )

    print(
        f"Products: {products_df.shape}"
    )

    print(
        f"Orders: {orders_df.shape}"
    )

    print(
        f"Order Items: {order_items_df.shape}"
    )

    print(
        f"Returns: {returns_df.shape}"
    )

    print(
        f"Marketing Campaigns: "
        f"{marketing_campaigns_df.shape}"
    )

    print(
        f"Master Dataset: "
        f"{master_df.shape}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()