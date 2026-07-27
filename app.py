import os

import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

VECTOR_DB = "vector_db"

EMBEDDING_MODEL = "text-embedding-3-small"

CHAT_MODEL = "gpt-4.1"

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Analyzer")

st.write(
    "Ask questions about your PDF documents using "
    "OpenAI + LangChain + ChromaDB."
)

st.divider()

# ----------------------------------------------------
# Load Embeddings
# ----------------------------------------------------

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

vector_db = Chroma(
    persist_directory=VECTOR_DB,
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0
)

# ----------------------------------------------------
# Question Input
# ----------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="Example: Which month had the highest electricity bill?"
)

# ----------------------------------------------------
# Ask AI
# ----------------------------------------------------

if st.button("Ask AI"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents..."):

        docs = retriever.invoke(question)

    context = ""

    for doc in docs:

        context += doc.page_content
        context += "\n\n"

    prompt = f"""
You are an AI assistant.

Answer ONLY using the supplied context.

If the answer is not available,
say:

"I couldn't find that information in the uploaded PDFs."

Context:

{context}

Question:

{question}

Provide:

1. Answer

2. Explanation

3. Summary
"""

    with st.spinner("Analyzing..."):

        response = llm.invoke(prompt)

    st.divider()

    st.subheader("Answer")

    st.write(response.content)

    with st.expander("Retrieved Context"):

        for i, doc in enumerate(docs, start=1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)