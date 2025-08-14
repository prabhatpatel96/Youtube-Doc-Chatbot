from langchain_core.prompts import PromptTemplate

# ===================== IMPROVED EXISTING PROMPTS =====================

SUMMARY_PROMPT = PromptTemplate(
    template=(
        "You are an expert summarizer. Summarize the following CONTEXT into exactly two coherent paragraphs.\n"
        "- Each paragraph should be 150–250 words, highly detailed, and logically ordered.\n"
        "- Preserve important facts, names, numbers, and cause-effect relationships.\n"
        "- Avoid unnecessary filler or repetition.\n"
        "- Do not add information not present in CONTEXT.\n"
        "- The tone should be clear and formal.\n\n"
        "CONTEXT:\n{context}"
    ),
    input_variables=["context"]
)

QA_PROMPT = PromptTemplate(
    template=(
        "You are an assistant answering a question strictly from the given CONTEXT.\n"
        "- If the answer cannot be found in the CONTEXT, respond exactly with: 'this is not present in video/document'.\n"
        "- Provide exactly two well-structured paragraphs.\n"
        "- Do not use outside knowledge.\n"
        "- Be concise but thorough.\n\n"
        "QUESTION: {question}\n\n"
        "CONTEXT:\n{context}"
    ),
    input_variables=["question", "context"]
)



KEY_INSIGHTS_PROMPT = PromptTemplate(
    template=(
        "Extract the 5–10 most important key takeaways from the following CONTEXT.\n"
        "- Each takeaway should be in a concise bullet point.\n"
        "- Focus on facts, insights, and crucial information.\n"
        "- Do not include trivial details.\n\n"
        "CONTEXT:\n{context}"
    ),
    input_variables=["context"]
)

TOPIC_CLASSIFIER_PROMPT = PromptTemplate(
    template=(
        "Classify the following CONTEXT into one or more high-level topics from this list:\n"
        "['Technology', 'Science', 'Education', 'Politics', 'History', 'Health', 'Entertainment', 'Sports', 'Business', 'Other'].\n"
        "- Return only the category names as a comma-separated list.\n\n"
        "CONTEXT:\n{context}"
    ),
    input_variables=["context"]
)

CONVERSATIONAL_REWRITE_PROMPT = PromptTemplate(
    template=(
        "Rewrite the following CONTEXT in a friendly, conversational style, as if explaining to a curious friend.\n"
        "- Keep all important facts but simplify complex language.\n"
        "- Use short, engaging sentences and a natural flow.\n\n"
        "CONTEXT:\n{context}"
    ),
    input_variables=["context"]
)
