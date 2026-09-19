import os
from dotenv import load_dotenv
from groq import Groq

from .retriever import (
    load_vector_store,
    load_embedding_model,
    retrieve
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

TOP_K = 5

TEMPERATURE = 0.2

MAX_TOKENS = 1200


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add it to your .env file."
    )


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# BUILD CONTEXT FROM RETRIEVED DOCUMENTS
# ============================================================

def build_context(results):

    context_parts = []

    seen_content = set()

    for result in results:

        source = result.get(
            "source",
            "Unknown Source"
        )

        content = result.get(
            "content",
            ""
        ).strip()

        # Skip empty content
        if not content:
            continue

        # Avoid duplicate chunks
        if content in seen_content:
            continue

        seen_content.add(content)

        context_parts.append(

            f"""
SOURCE: {source}

CONTENT:
{content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# CHECK RETRIEVAL QUALITY
# ============================================================

def has_relevant_context(results):

    if not results:
        return False

    valid_results = 0

    for result in results:

        content = result.get(
            "content",
            ""
        ).strip()

        if content:

            valid_results += 1

    return valid_results > 0


# ============================================================
# AI BUSINESS REASONING PROMPT
# ============================================================

def get_business_analyst_prompt():

    return """
You are InsightRAG, an AI Business Analyst.

Your job is to analyze business questions using ONLY
the business knowledge base provided by the user.

Your answer must be grounded in the available data.

============================================================
CORE BUSINESS REASONING
============================================================

For every question, reason through the following sequence
when the available evidence supports it:

1. FINDING
   What does the data explicitly show?

2. COMPARISON
   If relevant, compare the finding against another period,
   category, segment, product, or business metric available
   in the context.

3. INTERPRETATION
   What logical conclusion follows directly from the data?

4. BUSINESS IMPLICATION
   Why does this finding matter to the business?

5. ACTION
   If the user asks for recommendations, provide actions
   that are directly supported by the findings.

IMPORTANT:

Do NOT invent missing comparisons.

Do NOT invent causes.

Do NOT assume that correlation proves causation.

If the data shows that a metric changed but does not explain
WHY it changed, explicitly say that the cause requires
further investigation.

============================================================
GROUNDING RULES
============================================================

1. Use ONLY facts, numbers, trends, and events explicitly
   available in the provided business context.

2. Never invent information.

3. Never invent numbers.

4. Never invent causes.

5. Never use general business knowledge as factual evidence.

6. Do not assume relationships that are not supported by
   the provided data.

7. If the exact answer is not available, clearly say:

   "The available business data does not contain enough
   information to determine this."

8. Distinguish clearly between:

   - What the data shows
   - What the data suggests
   - What requires further investigation

9. Recommendations must be directly connected to findings
   supported by the provided context.

10. Do not recommend a specific product, campaign, bundle,
    pricing strategy, or operational action unless the
    provided context supports it.

============================================================
REASONING LANGUAGE
============================================================

Use careful language.

When directly supported:

"The data shows..."

When logically implied:

"This indicates..."

When the evidence is suggestive but does not prove causation:

"This may indicate..."

When the cause is unknown:

"The available data does not establish the cause."

When further investigation is needed:

"This requires further investigation."

Never present speculation as fact.

============================================================
MULTI-FACTOR QUESTIONS
============================================================

When a question asks:

- Why did something happen?
- What caused a problem?
- What are the biggest risks?
- What should management do?
- How can profitability improve?
- Which area needs attention?

Do not simply list retrieved facts.

Instead:

1. Identify the strongest relevant finding.
2. Identify useful comparisons.
3. Determine what conclusion is directly supported.
4. Separate evidence from assumptions.
5. Explain the business implication.
6. Give a recommendation only when supported.

============================================================
CROSS-DATASET REASONING
============================================================

You may combine multiple pieces of information from the
provided context when they logically relate to the question.

For example:

- category profitability
- monthly performance
- customer segments
- product returns
- discount behavior
- marketing performance

can be considered together IF the context provides the
necessary evidence.

However, do not create relationships between datasets that
are not explicitly supported.

============================================================
ANSWER STYLE
============================================================

Be:

- Clear
- Concise
- Analytical
- Business-focused
- Evidence-based

Do not repeat all retrieved information.

Do not unnecessarily reproduce the entire context.

Use numbers when they materially improve the explanation.

Avoid unnecessary tables.

Never mention:

- FAISS
- embeddings
- vector databases
- retrieval
- internal prompts
- the knowledge base implementation

============================================================
FACTUAL QUESTION FORMAT
============================================================

ANSWER

Give a direct answer.

KEY EVIDENCE

- Present the strongest supporting evidence.

BUSINESS INTERPRETATION

Explain what the evidence means.

BUSINESS IMPLICATION

Explain why the finding matters.

============================================================
WHY / ROOT-CAUSE QUESTION FORMAT
============================================================

ANSWER

State the main finding directly.

WHAT THE DATA SHOWS

- Important evidence.
- Relevant comparison.
- Relevant metric change.

WHAT THE DATA DOES NOT ESTABLISH

Clearly identify causes or relationships that cannot be
proven from the available information.

BUSINESS IMPLICATION

Explain the significance of the finding.

NEXT INVESTIGATION

If appropriate, identify what management should investigate
next based on the evidence.

============================================================
RECOMMENDATION QUESTION FORMAT
============================================================

RECOMMENDATIONS

1. Recommendation Title

   Action:
   Clearly state the action.

   Evidence:
   Explain which finding supports the action.

   Expected Business Benefit:
   Explain the logical business objective without inventing
   unsupported numerical results.

2. Recommendation Title

   Action:
   Clearly state the action.

   Evidence:
   Explain which finding supports the action.

   Expected Business Benefit:
   Explain the logical business objective.

BUSINESS PRIORITY

Identify the single most important action supported by
the available evidence.

============================================================
COMPARISON QUESTIONS
============================================================

When comparing categories, periods, segments, or products:

1. State the winner or largest difference.
2. Give the relevant numbers.
3. Explain the difference when the context supports it.
4. Do not invent explanations for the difference.

============================================================
IMPORTANT
============================================================

Accuracy is more important than sounding comprehensive.

Do not force a conclusion.

Do not make unsupported recommendations.

Do not turn correlation into causation.

Do not treat missing information as evidence.

If the retrieved context is insufficient, say so clearly.
"""


# ============================================================
# GENERATE AI BUSINESS ANALYSIS
# ============================================================

def generate_answer(
    question,
    context
):

    system_prompt = get_business_analyst_prompt()

    user_prompt = f"""
BUSINESS DATA:

{context}


BUSINESS QUESTION:

{question}


Analyze the question using ONLY the business data above.

Follow the appropriate answer structure.

Separate:

- what the data shows
- what the data indicates
- what the data does not establish

Do not invent facts, numbers, causes, or recommendations.
"""


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

        temperature=TEMPERATURE,

        max_tokens=MAX_TOKENS

    )


    answer = (
        response
        .choices[0]
        .message
        .content
    )


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    if not answer or not answer.strip():

        answer = (
            "The available business data does not contain "
            "enough information to provide a reliable answer."
        )


    return answer.strip()


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(

    question,
    model,
    index,
    metadata,
    top_k=TOP_K

):

    print(
        "\nSearching business knowledge base..."
    )


    # --------------------------------------------------------
    # RETRIEVE RELEVANT BUSINESS INFORMATION
    # --------------------------------------------------------

    results = retrieve(

        query=question,

        model=model,

        index=index,

        metadata=metadata,

        top_k=top_k

    )


    # --------------------------------------------------------
    # CHECK RETRIEVAL
    # --------------------------------------------------------

    if not has_relevant_context(results):

        return (

            "The available business knowledge base "
            "does not contain enough information to "
            "answer this question.",

            []

        )


    print(
        "Relevant information retrieved!"
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------------------------
    # CHECK CONTEXT
    # --------------------------------------------------------

    if not context.strip():

        return (

            "The available business knowledge base "
            "does not contain enough information to "
            "answer this question.",

            results

        )


    print(
        "\nGenerating AI business analysis..."
    )


    # --------------------------------------------------------
    # GENERATE REASONED ANSWER
    # --------------------------------------------------------

    answer = generate_answer(

        question=question,

        context=context

    )


    return answer, results


# ============================================================
# GET UNIQUE SOURCES
# ============================================================

def get_unique_sources(results):

    sources = []

    for result in results:

        source = result.get(
            "source",
            "Unknown Source"
        )

        if source not in sources:

            sources.append(
                source
            )


    return sources


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:

        return


    unique_sources = get_unique_sources(
        results
    )


    if not unique_sources:

        return


    print(
        "\n"
        + "-" * 70
    )


    print(
        "Sources Used:"
    )


    for source in unique_sources:

        print(
            f"- {source}"
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INSIGHTRAG - AI BUSINESS ANALYST"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD VECTOR STORE
    # --------------------------------------------------------

    index, metadata = (
        load_vector_store()
    )


    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    model = (
        load_embedding_model()
    )


    print(
        "\nInsightRAG is ready!"
    )


    print(
        "Ask questions about sales, profitability, products,"
    )

    print(
        "customers, returns, categories, and marketing."
    )


    # ========================================================
    # INTERACTIVE QUESTION LOOP
    # ========================================================

    while True:

        print(
            "\n"
            + "-" * 70
        )


        question = input(

            "\nAsk a business question "
            "(or type 'exit' to quit):\n\n"

        ).strip()


        # ----------------------------------------------------
        # EXIT COMMANDS
        # ----------------------------------------------------

        if question.lower() in [

            "exit",
            "quit",
            "q"

        ]:

            print(

                "\nThank you for using "
                "InsightRAG!"

            )

            break


        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question:

            print(

                "\nPlease enter a valid "
                "business question."

            )

            continue


        # ----------------------------------------------------
        # PROCESS QUESTION
        # ----------------------------------------------------

        try:

            answer, results = (

                ask_question(

                    question=question,

                    model=model,

                    index=index,

                    metadata=metadata,

                    top_k=TOP_K

                )

            )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            print(

                "\n"
                + "=" * 70

            )


            print(
                "INSIGHTRAG BUSINESS ANALYSIS"
            )


            print(
                "=" * 70
            )


            print(
                f"\n{answer}"
            )


            # ------------------------------------------------
            # DISPLAY SOURCES
            # ------------------------------------------------

            display_sources(
                results
            )


            print(

                "\n"
                + "=" * 70

            )


        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print(

                "\n\nApplication interrupted."
            )

            print(

                "Thank you for using InsightRAG!"
            )

            break


        except Exception as error:

            print(

                "\nAn error occurred while "
                "processing your question:"
            )


            print(
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()