import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import random
from datetime import date

# ============================================================
# CONFIGURATION
# ============================================================

NUM_CUSTOMERS = 2000
NUM_PRODUCTS = 250
NUM_ORDERS = 10000

MIN_ITEMS_PER_ORDER = 1
MAX_ITEMS_PER_ORDER = 4

HIGH_RETURN_PRODUCTS_COUNT = 12

NUM_CAMPAIGNS = 40

# ============================================================
# CONTROLLED BUSINESS SCENARIO CONFIGURATION
# ============================================================

ELECTRONICS_DECLINE_START = date(2025, 7, 1)
ELECTRONICS_DECLINE_END = date(2025, 9, 30)

ELECTRONICS_COST_INCREASE = 0.12
ELECTRONICS_EXTRA_DISCOUNT = 10

CLOTHING_DISCOUNT_START = date(2025, 11, 1)
CLOTHING_DISCOUNT_END = date(2025, 12, 31)

CLOTHING_EXTRA_DISCOUNT = 15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERATE CUSTOMERS
# ============================================================

def generate_customers(num_customers):

    print("Generating customers...")

    customers = []

    cities = [
        ("Delhi", "Delhi"),
        ("Mumbai", "Maharashtra"),
        ("Bengaluru", "Karnataka"),
        ("Hyderabad", "Telangana"),
        ("Chennai", "Tamil Nadu"),
        ("Kolkata", "West Bengal"),
        ("Pune", "Maharashtra"),
        ("Ahmedabad", "Gujarat"),
        ("Jaipur", "Rajasthan"),
        ("Lucknow", "Uttar Pradesh")
    ]

    segments = ["Premium", "Regular", "Budget"]

    for i in range(1, num_customers + 1):

        city, state = random.choice(cities)

        customer = {
            "customer_id": f"C{i:05d}",
            "customer_name": fake.name(),
            "age": random.randint(18, 70),
            "gender": random.choice(["Male", "Female"]),
            "city": city,
            "state": state,
            "signup_date": fake.date_between(
                start_date="-4y",
                end_date="today"
            ),
            "customer_segment": random.choices(
                segments,
                weights=[0.2, 0.55, 0.25]
            )[0]
        }

        customers.append(customer)

    df = pd.DataFrame(customers)

    return df



# ============================================================
# GENERATE PRODUCTS
# ============================================================

def generate_products(num_products):

    print("Generating products...")

    product_catalog = {

        "Electronics": {
            "Smartphones": [
                "Nova X1",
                "Nova X2",
                "Galaxy Pro",
                "Pixel Max",
                "Edge Ultra"
            ],
            "Laptops": [
                "TechBook Air",
                "TechBook Pro",
                "UltraBook X",
                "PowerBook 15"
            ],
            "Headphones": [
                "SoundBeat Pro",
                "BassMax",
                "AirSound",
                "NoiseFree X"
            ],
            "Smartwatches": [
                "FitWatch Pro",
                "ActiveWatch",
                "Pulse X"
            ]
        },

        "Clothing": {
            "T-Shirts": [
                "Classic Cotton Tee",
                "Urban Fit T-Shirt",
                "Premium Polo"
            ],
            "Jeans": [
                "Slim Fit Jeans",
                "Regular Fit Jeans",
                "Stretch Denim"
            ],
            "Jackets": [
                "Winter Jacket",
                "Denim Jacket",
                "Bomber Jacket"
            ],
            "Shoes": [
                "Running Shoes",
                "Casual Sneakers",
                "Training Shoes"
            ]
        },

        "Home & Kitchen": {
            "Furniture": [
                "Modern Office Chair",
                "Study Table",
                "Wooden Bookshelf"
            ],
            "Kitchen Appliances": [
                "Air Fryer",
                "Mixer Grinder",
                "Coffee Maker"
            ],
            "Home Decor": [
                "Wall Clock",
                "Table Lamp",
                "Decorative Vase"
            ]
        },

        "Sports": {
            "Fitness Equipment": [
                "Yoga Mat",
                "Adjustable Dumbbells",
                "Resistance Bands"
            ],
            "Outdoor Equipment": [
                "Camping Tent",
                "Cycling Helmet",
                "Sports Backpack"
            ]
        },

        "Beauty": {
            "Skincare": [
                "Vitamin C Serum",
                "Daily Moisturizer",
                "Sunscreen SPF 50"
            ],
            "Haircare": [
                "Shampoo",
                "Hair Conditioner",
                "Hair Serum"
            ],
            "Cosmetics": [
                "Matte Lipstick",
                "Foundation",
                "Face Palette"
            ]
        }
    }


    suppliers = [
        "TechSource India",
        "Global Retail Supply",
        "Prime Distributors",
        "Urban Goods",
        "Elite Wholesale",
        "Nova Supply Chain"
    ]


    # Base price ranges for each category
    price_ranges = {

        "Electronics": (1500, 80000),

        "Clothing": (500, 6000),

        "Home & Kitchen": (800, 25000),

        "Sports": (500, 15000),

        "Beauty": (200, 5000)
    }


    products = []

    product_id = 1


    while len(products) < num_products:

        category = random.choice(
            list(product_catalog.keys())
        )

        sub_category = random.choice(
            list(product_catalog[category].keys())
        )

        product_base_name = random.choice(
            product_catalog[category][sub_category]
        )


        # Add a model/version number to make products unique
        version = random.randint(100, 999)

        product_name = f"{product_base_name} {version}"


        min_price, max_price = price_ranges[category]

        selling_price = round(
            random.uniform(min_price, max_price),
            2
        )


        # Cost is between 55% and 80% of selling price
        cost_price = round(
            selling_price * random.uniform(0.55, 0.80),
            2
        )


        product = {

            "product_id": f"P{product_id:04d}",

            "product_name": product_name,

            "category": category,

            "sub_category": sub_category,

            "cost_price": cost_price,

            "selling_price": selling_price,

            "supplier": random.choice(suppliers)
        }


        products.append(product)

        product_id += 1


    df = pd.DataFrame(products)

    return df


# ============================================================
# GENERATE ORDERS
# ============================================================

def generate_orders(num_orders, customers_df):

    print("Generating orders...")

    orders = []

    customer_ids = customers_df["customer_id"].tolist()

    payment_methods = [
        "UPI",
        "Credit Card",
        "Debit Card",
        "Net Banking",
        "Cash on Delivery"
    ]

    order_statuses = [
        "Delivered",
        "Delivered",
        "Delivered",
        "Delivered",
        "Cancelled"
    ]


    for i in range(1, num_orders + 1):

        customer_id = random.choice(
            customer_ids
        )

        customer_segment = customers_df.loc[
            customers_df["customer_id"] == customer_id,
            "customer_segment"
        ].iloc[0]


        # Payment preferences based on customer segment
        if customer_segment == "Premium":

            payment_method = random.choices(
                payment_methods,
                weights=[0.20, 0.40, 0.20, 0.15, 0.05]
            )[0]

        elif customer_segment == "Budget":

            payment_method = random.choices(
                payment_methods,
                weights=[0.35, 0.10, 0.15, 0.10, 0.30]
            )[0]

        else:

            payment_method = random.choice(
                payment_methods
            )


        order = {

            "order_id": f"O{i:06d}",

            "customer_id": customer_id,

            "order_date": fake.date_between(
    start_date=date(2024, 1, 1),
    end_date=date(2025, 12, 31)
),

            "payment_method": payment_method,

            "order_status": random.choice(
                order_statuses
            )
        }


        orders.append(order)


    df = pd.DataFrame(orders)

    return df


# ============================================================
# GENERATE ORDER ITEMS
# ============================================================

def generate_order_items(
    orders_df,
    products_df
):

    print("Generating order items...")

    order_items = []

    order_item_id = 1

    product_lookup = (
        products_df
        .set_index("product_id")
        .to_dict("index")
    )

    product_ids = (
        products_df["product_id"]
        .tolist()
    )


    # Process each order
    for _, order in orders_df.iterrows():

        # Cancelled orders will not contain completed order items
        if order["order_status"] == "Cancelled":
            continue


        # Number of different products in this order
        num_items = random.randint(
            MIN_ITEMS_PER_ORDER,
            MAX_ITEMS_PER_ORDER
        )


        # Select unique products for the order
        selected_products = random.sample(
            product_ids,
            num_items
        )


        # Create one row for every product
        for product_id in selected_products:

            product = product_lookup[
                product_id
            ]


            quantity = random.randint(
                1,
                5
            )


            discount_percentage = random.choices(
                [0, 5, 10, 15, 20, 25],
                weights=[30, 25, 20, 12, 8, 5]
            )[0]


            selling_price = (
                product["selling_price"]
            )

            cost_price = (
                product["cost_price"]
            )


            # Revenue after discount
            sales_amount = round(
                quantity
                * selling_price
                * (1 - discount_percentage / 100),
                2
            )


            # Total cost for the quantity sold
            total_cost = round(
                quantity
                * cost_price,
                2
            )


            # Profit
            profit = round(
                sales_amount - total_cost,
                2
            )


            order_item = {

                "order_item_id": (
                    f"OI{order_item_id:07d}"
                ),

                "order_id": (
                    order["order_id"]
                ),

                "product_id": product_id,

                "quantity": quantity,

                "discount_percentage": (
                    discount_percentage
                ),

                "sales_amount": (
                    sales_amount
                ),

                "total_cost": (
                    total_cost
                ),

                "profit": profit
            }


            order_items.append(
                order_item
            )

            order_item_id += 1


    df = pd.DataFrame(
        order_items
    )

    return df


# ============================================================
# APPLY CONTROLLED BUSINESS SCENARIOS
# ============================================================

def apply_controlled_business_scenarios(
    orders_df,
    order_items_df,
    products_df
):

    print("Applying controlled business scenarios...")


    # --------------------------------------------------------
    # CREATE PRODUCT LOOKUP
    # --------------------------------------------------------

    product_category_lookup = (
        products_df
        .set_index("product_id")["category"]
        .to_dict()
    )


    # --------------------------------------------------------
    # CREATE ORDER DATE LOOKUP
    # --------------------------------------------------------

    order_date_lookup = (
        orders_df
        .set_index("order_id")["order_date"]
        .to_dict()
    )


    # --------------------------------------------------------
    # IDENTIFY ELECTRONICS ITEMS
    # --------------------------------------------------------

    order_items_df = order_items_df.copy()


    order_items_df["category"] = (
        order_items_df["product_id"]
        .map(product_category_lookup)
    )


    order_items_df["order_date"] = (
        order_items_df["order_id"]
        .map(order_date_lookup)
    )


    order_items_df["order_date"] = pd.to_datetime(
        order_items_df["order_date"]
    )


    # --------------------------------------------------------
    # CREATE Q3 2025 ELECTRONICS MASK
    # --------------------------------------------------------

    scenario_mask = (

        (order_items_df["category"] == "Electronics")

        &

        (
            order_items_df["order_date"]
            >= pd.Timestamp(ELECTRONICS_DECLINE_START)
        )

        &

        (
            order_items_df["order_date"]
            <= pd.Timestamp(ELECTRONICS_DECLINE_END)
        )

    )


    affected_rows = scenario_mask.sum()


    print(
        f"Electronics items affected in Q3 2025: "
        f"{affected_rows}"
    )


    # --------------------------------------------------------
    # INCREASE COST
    # --------------------------------------------------------

    order_items_df.loc[
        scenario_mask,
        "total_cost"
    ] = (

        order_items_df.loc[
            scenario_mask,
            "total_cost"
        ]

        * (1 + ELECTRONICS_COST_INCREASE)

    ).round(2)


    # --------------------------------------------------------
    # INCREASE DISCOUNT
    # --------------------------------------------------------

    order_items_df.loc[
        scenario_mask,
        "discount_percentage"
    ] = (

        order_items_df.loc[
            scenario_mask,
            "discount_percentage"
        ]

        + ELECTRONICS_EXTRA_DISCOUNT

    ).clip(
        upper=40
    )


    # --------------------------------------------------------
    # RECALCULATE SALES AMOUNT
    # --------------------------------------------------------

    # To correctly recalculate discounted sales,
    # first reconstruct the original gross sales.

    original_discount = (

        order_items_df.loc[
            scenario_mask,
            "discount_percentage"
        ]

        - ELECTRONICS_EXTRA_DISCOUNT

    )


    original_sales_amount = (

        order_items_df.loc[
            scenario_mask,
            "sales_amount"
        ]

        /

        (
            1
            - original_discount / 100
        )

    )


    new_discount = (

        order_items_df.loc[
            scenario_mask,
            "discount_percentage"
        ]

    )


    new_sales_amount = (

        original_sales_amount

        * (

            1
            - new_discount / 100

        )

    ).round(2)


    order_items_df.loc[
        scenario_mask,
        "sales_amount"
    ] = new_sales_amount


    # --------------------------------------------------------
    # RECALCULATE PROFIT
    # --------------------------------------------------------

    order_items_df.loc[
        scenario_mask,
        "profit"
    ] = (

        order_items_df.loc[
            scenario_mask,
            "sales_amount"
        ]

        -

        order_items_df.loc[
            scenario_mask,
            "total_cost"
        ]

    ).round(2)

        # ========================================================
    # CLOTHING DISCOUNT TRAP SCENARIO
    # ========================================================

    clothing_mask = (
        (order_items_df["category"] == "Clothing")
        &
        (
            order_items_df["order_date"]
            >= pd.Timestamp(CLOTHING_DISCOUNT_START)
        )
        &
        (
            order_items_df["order_date"]
            <= pd.Timestamp(CLOTHING_DISCOUNT_END)
        )
    )

    clothing_affected_rows = clothing_mask.sum()

    print(
        f"Clothing items affected in Nov-Dec 2025: "
        f"{clothing_affected_rows}"
    )

    original_clothing_discount = (
        order_items_df.loc[
            clothing_mask,
            "discount_percentage"
        ].copy()
    )

    new_clothing_discount = (
        original_clothing_discount
        + CLOTHING_EXTRA_DISCOUNT
    ).clip(upper=50)

    original_clothing_sales = (
        order_items_df.loc[
            clothing_mask,
            "sales_amount"
        ]
        /
        (
            1
            - original_clothing_discount / 100
        )
    )

    new_clothing_sales = (
        original_clothing_sales
        *
        (
            1
            - new_clothing_discount / 100
        )
    ).round(2)

    order_items_df.loc[
        clothing_mask,
        "discount_percentage"
    ] = new_clothing_discount

    order_items_df.loc[
        clothing_mask,
        "sales_amount"
    ] = new_clothing_sales

    order_items_df.loc[
        clothing_mask,
        "profit"
    ] = (
        order_items_df.loc[
            clothing_mask,
            "sales_amount"
        ]
        -
        order_items_df.loc[
            clothing_mask,
            "total_cost"
        ]
    ).round(2)
    # --------------------------------------------------------
    # REMOVE TEMPORARY COLUMNS
    # --------------------------------------------------------

    order_items_df = order_items_df.drop(
        columns=[
            "category",
            "order_date"
        ]
    )


    print(
        "Controlled business scenarios applied successfully!"
    )


    return order_items_df
# ============================================================
# GENERATE RETURNS
# ============================================================

def generate_returns(
    orders_df,
    order_items_df,
    products_df
):

    print("Generating returns...")

    returns = []

    return_id = 1


    # --------------------------------------------------------
    # PRODUCT LOOKUP
    # --------------------------------------------------------

    product_lookup = (
        products_df
        .set_index("product_id")
        .to_dict("index")
    )


    # --------------------------------------------------------
    # SELECT HIGH RETURN PRODUCTS
    # --------------------------------------------------------

    product_ids = (
        products_df["product_id"]
        .tolist()
    )

    high_return_products = set(
        random.sample(
            product_ids,
            HIGH_RETURN_PRODUCTS_COUNT
        )
    )


    # --------------------------------------------------------
    # BASE RETURN PROBABILITIES
    # --------------------------------------------------------

    category_return_probability = {

        "Electronics": 0.08,

        "Clothing": 0.12,

        "Home & Kitchen": 0.05,

        "Sports": 0.04,

        "Beauty": 0.07
    }


    # --------------------------------------------------------
    # RETURN REASONS
    # --------------------------------------------------------

    return_reasons = [

        "Damaged Product",

        "Wrong Product Received",

        "Size/Fit Issue",

        "Defective Product",

        "Quality Not as Expected",

        "Changed Mind",

        "Late Delivery"
    ]


    # --------------------------------------------------------
    # CREATE LOOKUP FOR ORDER INFORMATION
    # --------------------------------------------------------

    order_lookup = (
        orders_df
        .set_index("order_id")
        .to_dict("index")
    )


    # --------------------------------------------------------
    # PROCESS EACH ORDER ITEM
    # --------------------------------------------------------

    for _, item in order_items_df.iterrows():

        order_id = item["order_id"]

        product_id = item["product_id"]

        product = product_lookup[
            product_id
        ]

        category = product["category"]


        # Base return probability based on category
        return_probability = (
            category_return_probability[
                category
            ]
        )


        # High-return products have increased probability
        if product_id in high_return_products:

            return_probability += 0.18


        # Higher discounts slightly increase return probability
        if item["discount_percentage"] >= 20:

            return_probability += 0.03


        # Decide whether this item is returned
        is_returned = random.random() < return_probability


        if not is_returned:
            continue


        # Get original order date
        order_date = pd.to_datetime(
            order_lookup[order_id]["order_date"]
        )


        # Return occurs 3 to 30 days after purchase
        return_days = random.randint(
            3,
            30
        )

        return_date = (
            order_date
            + pd.Timedelta(days=return_days)
        )


        # Generate category-aware return reason
        if category == "Clothing":

            reason = random.choices(
                return_reasons,
                weights=[
                    10,  # Damaged
                    5,   # Wrong Product
                    40,  # Size/Fit
                    5,   # Defective
                    20,  # Quality
                    15,  # Changed Mind
                    5    # Late Delivery
                ]
            )[0]

        elif category == "Electronics":

            reason = random.choices(
                return_reasons,
                weights=[
                    20,
                    5,
                    0,
                    35,
                    15,
                    10,
                    15
                ]
            )[0]

        else:

            reason = random.choice(
                return_reasons
            )


        return_record = {

            "return_id": (
                f"R{return_id:06d}"
            ),

            "order_id": order_id,

            "product_id": product_id,

            "return_reason": reason,

            "return_date": return_date.date()
        }


        returns.append(
            return_record
        )

        return_id += 1


    df = pd.DataFrame(
        returns
    )


    # --------------------------------------------------------
    # SAVE HIGH RETURN PRODUCT INFORMATION
    # --------------------------------------------------------

    print(
        f"High-return products intentionally created: "
        f"{len(high_return_products)}"
    )


    return df


# ============================================================
# GENERATE MARKETING CAMPAIGNS
# ============================================================

def generate_marketing_campaigns(num_campaigns):

    print("Generating marketing campaigns...")

    campaigns = []

    campaign_id = 1


    channels = [
        "Email",
        "Social Media",
        "Search Ads",
        "Influencer Marketing",
        "Affiliate Marketing"
    ]


    campaign_prefixes = [
        "Summer Sale",
        "Festive Offer",
        "New Launch",
        "Weekend Deal",
        "Mega Savings",
        "Premium Collection",
        "Flash Sale",
        "Seasonal Promotion"
    ]


    # --------------------------------------------------------
    # GENERATE CAMPAIGNS
    # --------------------------------------------------------

    for i in range(num_campaigns):

        channel = random.choice(
            channels
        )

        campaign_prefix = random.choice(
            campaign_prefixes
        )


        # Random campaign start date
        start_date = fake.date_between(
            start_date=date(2024, 1, 1),
            end_date=date(2025, 11, 30)
        )


        # Campaign duration
        duration_days = random.randint(
            7,
            30
        )


        end_date = (
            start_date
            + pd.Timedelta(days=duration_days)
        )


        # Campaign cost
        campaign_cost = round(
            random.uniform(
                50000,
                500000
            ),
            2
        )


        # ----------------------------------------------------
        # CAMPAIGN PERFORMANCE PATTERN
        # ----------------------------------------------------

        performance_type = random.choices(
            [
                "High",
                "Average",
                "Poor"
            ],
            weights=[
                25,
                55,
                20
            ]
        )[0]


        if performance_type == "High":

            revenue_multiplier = random.uniform(
                4.0,
                7.0
            )

        elif performance_type == "Poor":

            revenue_multiplier = random.uniform(
                0.6,
                1.4
            )

        else:

            revenue_multiplier = random.uniform(
                1.8,
                3.5
            )


        revenue_generated = round(
            campaign_cost
            * revenue_multiplier,
            2
        )


        roi = round(
            (
                revenue_generated
                - campaign_cost
            )
            / campaign_cost
            * 100,
            2
        )


        campaign = {

            "campaign_id": (
                f"MC{campaign_id:04d}"
            ),

            "campaign_name": (
                f"{campaign_prefix} "
                f"{channel} "
                f"{campaign_id}"
            ),

            "channel": channel,

            "start_date": start_date,

            "end_date": end_date,

            "campaign_cost": campaign_cost,

            "revenue_generated": revenue_generated,

            "roi_percentage": roi,

            "performance_type": performance_type
        }


        campaigns.append(
            campaign
        )

        campaign_id += 1


    df = pd.DataFrame(
        campaigns
    )

    return df
def main():

    # ========================================================
    # GENERATE CUSTOMERS
    # ========================================================

    customers_df = generate_customers(
        NUM_CUSTOMERS
    )

    customers_path = RAW_DATA_DIR / "customers.csv"

    customers_df.to_csv(
        customers_path,
        index=False
    )

    print("\nCustomers generated successfully!")
    print(
        f"Total customers: {len(customers_df)}"
    )
    print(
        f"Saved to: {customers_path}"
    )


    # ========================================================
    # GENERATE PRODUCTS
    # ========================================================

    products_df = generate_products(
        NUM_PRODUCTS
    )

    products_path = RAW_DATA_DIR / "products.csv"

    products_df.to_csv(
        products_path,
        index=False
    )

    print("\nProducts generated successfully!")
    print(
        f"Total products: {len(products_df)}"
    )
    print(
        f"Saved to: {products_path}"
    )
        # ========================================================
    # GENERATE ORDERS
    # ========================================================

    orders_df = generate_orders(
        NUM_ORDERS,
        customers_df
    )

    orders_path = RAW_DATA_DIR / "orders.csv"

    orders_df.to_csv(
        orders_path,
        index=False
    )

    print("\nOrders generated successfully!")
    print(
        f"Total orders: {len(orders_df)}"
    )
    print(
        f"Saved to: {orders_path}"
    )
        # ========================================================
    # GENERATE ORDER ITEMS
    # ========================================================

    order_items_df = generate_order_items(
        orders_df,
        products_df
    )
        # ========================================================
    # APPLY CONTROLLED BUSINESS SCENARIOS
    # ========================================================

    order_items_df = apply_controlled_business_scenarios(
        orders_df,
        order_items_df,
        products_df
    )

    order_items_path = (
        RAW_DATA_DIR
        / "order_items.csv"
    )

    order_items_df.to_csv(
        order_items_path,
        index=False
    )

    print(
        "\nOrder items generated successfully!"
    )

    print(
        f"Total order items: {len(order_items_df)}"
    )

    print(
        f"Saved to: {order_items_path}"
    )
        # ========================================================
    # GENERATE RETURNS
    # ========================================================

    returns_df = generate_returns(
        orders_df,
        order_items_df,
        products_df
    )

    returns_path = (
        RAW_DATA_DIR
        / "returns.csv"
    )

    returns_df.to_csv(
        returns_path,
        index=False
    )

    print(
        "\nReturns generated successfully!"
    )

    print(
        f"Total returns: {len(returns_df)}"
    )

    print(
        f"Saved to: {returns_path}"
    )
        # ========================================================
    # GENERATE MARKETING CAMPAIGNS
    # ========================================================

    campaigns_df = generate_marketing_campaigns(
        NUM_CAMPAIGNS
    )

    campaigns_path = (
        RAW_DATA_DIR
        / "marketing_campaigns.csv"
    )

    campaigns_df.to_csv(
        campaigns_path,
        index=False
    )

    print(
        "\nMarketing campaigns generated successfully!"
    )

    print(
        f"Total campaigns: {len(campaigns_df)}"
    )

    print(
        f"Saved to: {campaigns_path}"
    )


    # ========================================================
    # PREVIEW DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("CUSTOMERS PREVIEW")
    print("=" * 60)

    print(
        customers_df.head()
    )


    print("\n" + "=" * 60)
    print("PRODUCTS PREVIEW")
    print("=" * 60)

    print(
        products_df.head()
    )
    print("\n" + "=" * 60)
    print("ORDERS PREVIEW")
    print("=" * 60)

    print(
        orders_df.head()
    )
    print("\n" + "=" * 60)
    print("ORDER ITEMS PREVIEW")
    print("=" * 60)

    print(
        order_items_df.head()
    )
    print("\n" + "=" * 60)
    print("RETURNS PREVIEW")
    print("=" * 60)

    print(
        returns_df.head()
    )
    print("\n" + "=" * 60)
    print("MARKETING CAMPAIGNS PREVIEW")
    print("=" * 60)

    print(
        campaigns_df.head()
    )


if __name__ == "__main__":
    main()