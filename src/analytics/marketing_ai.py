import sys
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# MARKETING ANALYTICS IMPORTS
# ============================================================

from analytics.marketing_analytics import (
    calculate_marketing_kpis,
    channel_performance,
    performance_type_summary,
    top_campaigns_by_roi,
    lowest_campaigns_by_roi,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")


# ============================================================
# GROQ CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# MARKETING QUESTION DETECTION
# ============================================================

def is_marketing_question(question):

    question_lower = question.lower()

    marketing_keywords = [
        "marketing",
        "campaign",
        "campaigns",
        "roi",
        "advertising",
        "advertisement",
        "channel",
        "channels",
        "email campaign",
        "search ads",
        "social media",
        "influencer",
        "affiliate",
        "marketing spend",
        "marketing cost",
        "marketing revenue",
        "campaign performance",
    ]

    return any(
        keyword in question_lower
        for keyword in marketing_keywords
    )


# ============================================================
# BUILD MARKETING EVIDENCE
# ============================================================

def build_marketing_evidence(df):

    kpis = calculate_marketing_kpis(df)

    channels = channel_performance(df)

    performance_types = performance_type_summary(df)

    top_campaigns = top_campaigns_by_roi(
        df,
        5,
    )

    lowest_campaigns = lowest_campaigns_by_roi(
        df,
        5,
    )

    evidence = []

    # ========================================================
    # OVERALL KPIs
    # ========================================================

    evidence.append(
        "OVERALL MARKETING KPIs"
    )

    evidence.append(
        f"Total campaigns: "
        f"{kpis['total_campaigns']}"
    )

    evidence.append(
        f"Total marketing cost: "
        f"₹{kpis['total_cost']:,.2f}"
    )

    evidence.append(
        f"Total revenue generated: "
        f"₹{kpis['total_revenue']:,.2f}"
    )

    evidence.append(
        f"Overall marketing ROI: "
        f"{kpis['overall_roi']:.2f}%"
    )

    evidence.append(
        f"Average campaign ROI: "
        f"{kpis['average_roi']:.2f}%"
    )

    evidence.append(
        f"Average campaign duration: "
        f"{kpis['average_duration']:.2f} days"
    )

    # ========================================================
    # CHANNEL PERFORMANCE
    # ========================================================

    if not channels.empty:

        evidence.append("")
        evidence.append(
            "MARKETING CHANNEL PERFORMANCE"
        )

        for _, row in channels.iterrows():

            evidence.append(
                f"- {row['channel']}: "
                f"{int(row['campaigns'])} campaigns, "
                f"cost ₹{row['total_cost']:,.2f}, "
                f"revenue ₹{row['total_revenue']:,.2f}, "
                f"average campaign ROI "
                f"{row['average_roi']:.2f}%, "
                f"aggregated ROI "
                f"{row['overall_roi']:.2f}%"
            )

    # ========================================================
    # PERFORMANCE TYPES
    # ========================================================

    if not performance_types.empty:

        evidence.append("")
        evidence.append(
            "CAMPAIGN PERFORMANCE CLASSIFICATION"
        )

        for _, row in performance_types.iterrows():

            evidence.append(
                f"- {row['performance_type']}: "
                f"{int(row['campaigns'])} campaigns, "
                f"cost ₹{row['total_cost']:,.2f}, "
                f"revenue ₹{row['total_revenue']:,.2f}, "
                f"average campaign ROI "
                f"{row['average_roi']:.2f}%, "
                f"aggregated ROI "
                f"{row['overall_roi']:.2f}%"
            )

    # ========================================================
    # TOP CAMPAIGNS
    # ========================================================

    if not top_campaigns.empty:

        evidence.append("")
        evidence.append(
            "TOP CAMPAIGNS BY ROI"
        )

        for _, row in top_campaigns.iterrows():

            evidence.append(
                f"- {row['campaign_name']} | "
                f"Channel: {row['channel']} | "
                f"ROI: {row['roi_percentage']:.2f}% | "
                f"Cost: ₹{row['campaign_cost']:,.2f} | "
                f"Revenue: ₹{row['revenue_generated']:,.2f} | "
                f"Type: {row['performance_type']}"
            )

    # ========================================================
    # LOWEST CAMPAIGNS
    # ========================================================

    if not lowest_campaigns.empty:

        evidence.append("")
        evidence.append(
            "LOWEST CAMPAIGNS BY ROI"
        )

        for _, row in lowest_campaigns.iterrows():

            evidence.append(
                f"- {row['campaign_name']} | "
                f"Channel: {row['channel']} | "
                f"ROI: {row['roi_percentage']:.2f}% | "
                f"Cost: ₹{row['campaign_cost']:,.2f} | "
                f"Revenue: ₹{row['revenue_generated']:,.2f} | "
                f"Type: {row['performance_type']}"
            )

    return "\n".join(evidence)


# ============================================================
# GENERATE MARKETING AI ANSWER
# ============================================================

def generate_marketing_answer(
    question,
    df,
):

    if df is None or df.empty:

        return (
            "The marketing dataset is unavailable, "
            "so there is not enough information to answer "
            "this question."
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY was not found in the .env file."
        )

    client = Groq(
        api_key=api_key
    )

    evidence = build_marketing_evidence(
        df
    )

    # ========================================================
    # STRICT MARKETING GROUNDING PROMPT
    # ========================================================

    system_prompt = """
You are InsightRAG's Marketing Business Analyst.

Your ONLY source of truth is the MARKETING BUSINESS DATA
provided in the user message.

You must answer strictly from that data.

============================================================
ABSOLUTE GROUNDING RULES
============================================================

1. NEVER invent or assume:
   - causes
   - reasons
   - explanations
   - trends
   - customer behavior
   - targeting problems
   - audience problems
   - creative problems
   - messaging problems
   - channel-fit problems
   - budget problems
   - market conditions
   - operational problems
   - business strategy
   - any other fact that is not explicitly present
     in the supplied evidence.

2. Do NOT provide hypothetical causes.

For example, NEVER write statements such as:

- "This may be due to poor targeting."
- "This could indicate weak creative."
- "This may be caused by channel fit."
- "This suggests a targeting problem."
- "Possible reasons include..."
- "This could be due to..."
- "The campaign may have suffered from..."

Even when presented as possibilities, these are unsupported
unless the supplied evidence explicitly contains those causes.

3. Do NOT introduce examples of possible causes.

If the dataset does not contain the reason for a result,
simply state that the reason is not available in the dataset.

4. Distinguish strictly between:

DATA SHOWS:
Facts, numbers, rankings, comparisons, and classifications
directly supported by the supplied evidence.

REQUIRES INVESTIGATION:
Questions or explanations that cannot be answered from
the supplied evidence.

5. When discussing poor-performing campaigns, state only
their measured performance.

For example:

"The data shows that these campaigns generated low ROI."

Do NOT continue with an assumed explanation.

If appropriate, add:

"The specific reasons for this performance are not available
in the current marketing dataset and require further
investigation."

6. When discussing high-performing campaigns or channels,
do not claim why they performed well unless the evidence
explicitly provides the reason.

You may state the measured result.

For example:

"Influencer Marketing has the highest aggregated ROI at
321.56%."

Do NOT say:

"This happened because influencer marketing has better
audience targeting."

unless that fact is explicitly contained in the evidence.

7. When comparing marketing channels:

Use AGGREGATED ROI.

Do not use average campaign ROI to determine which channel
has the highest aggregated ROI.

8. When discussing individual campaigns:

Use the supplied campaign-level ROI.

9. Do not confuse:
- aggregated ROI
- average campaign ROI

They are different metrics.

10. Never create numbers.

Every number in the answer must come directly from the
provided marketing evidence or be a simple calculation
directly supported by that evidence.

11. Do not use outside knowledge.

12. Do not make causal claims from correlation or ranking.

13. Do not describe a result as a "trend" unless the supplied
evidence explicitly contains a time-based trend.

14. Do not describe a result as "significant" unless the
evidence itself establishes significance.

15. Do not describe a campaign as "ineffective" when the
dataset only provides a low ROI. Prefer factual wording such
as "generated low ROI" or "had a low ROI."

16. Do not use words such as:
- suggests
- indicates
- implies
- likely
- probably
- may be due to
- could be caused by
- possibly because
when they introduce an unsupported explanation.

============================================================
WHEN INFORMATION IS INSUFFICIENT
============================================================

If the question asks WHY something happened and the evidence
does not contain the reason, say:

"The available marketing data does not contain enough
information to determine the specific reason."

You may still provide the relevant measured result.

Example:

"The campaign had an ROI of -22.44%. The available marketing
data does not contain enough information to determine the
specific reason for this result."

============================================================
RECOMMENDATIONS
============================================================

Only provide recommendations that are directly connected to
a measured fact in the evidence.

Recommendations must NOT assume an unverified cause.

Good:

"Action: Review the campaigns with negative ROI.
Why: These campaigns generated negative ROI in the supplied
data."

Good:

"Action: Review the performance of the Poor campaign group.
Why: The Poor group has an aggregated ROI of 2.42%."

Bad:

"Action: Improve campaign targeting.
Why: Poor targeting caused the low ROI."

The second example is NOT allowed because the cause is not
supported by the evidence.

============================================================
ANSWER STYLE
============================================================

Keep answers concise, factual, and business-focused.

Use this format:

ANSWER

Direct factual answer.

KEY EVIDENCE

- Important supporting number or comparison.
- Additional supporting evidence if necessary.

BUSINESS INSIGHT

A short interpretation that remains completely grounded
in the supplied evidence.

If the question requires information that is not available,
clearly state that the specific explanation requires further
investigation.

If recommendations are requested, add:

RECOMMENDATIONS

- Action: ...
  Why: ...

============================================================
FINAL SELF-CHECK
============================================================

Before producing the answer, check every sentence.

Ask:

"Can this sentence be directly supported by the supplied
marketing evidence?"

If NO:
- remove it, or
- explicitly state that the information is unavailable.

Never fill missing information with assumptions.
"""

    user_prompt = f"""
MARKETING BUSINESS DATA
=======================

{evidence}

QUESTION
========

{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    from analytics.marketing_analytics import (
        load_marketing_data
    )

    print("=" * 70)
    print("INSIGHTRAG - MARKETING AI ANALYST")
    print("=" * 70)

    marketing_df = load_marketing_data()

    question = (
        "Which marketing channel has the highest ROI?"
    )

    print(
        f"\nQuestion: {question}"
    )

    print(
        "\nGenerating answer..."
    )

    answer = generate_marketing_answer(
        question,
        marketing_df,
    )

    print("\n")
    print(answer)