import os
import shutil
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_chroma import Chroma


# ====================================================
# Environment
# ====================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found")
    st.stop()


# ====================================================
# Configuration
# ====================================================

UPLOAD_FOLDER = "uploaded_files"
VECTOR_DB = "vector_db"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ====================================================
# Page Configuration
# ====================================================

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide"
)


st.title("📄 AI Document Analyzer")

st.write(
    "Upload PDF documents and ask questions using AI."
)


# ====================================================
# Session State
# ====================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "knowledge_ready" not in st.session_state:
    st.session_state.knowledge_ready = False


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ====================================================
# Document Management
# ====================================================

st.divider()

st.header("📂 Document Management")


col1, col2 = st.columns(2)


with col1:

    uploaded_files = st.file_uploader(
        "📤 Upload Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )


with col2:

    delete_button = st.button(
        "🗑️ Delete Document(s)",
        use_container_width=True
    )


# ====================================================
# Delete Documents
# ====================================================

if delete_button:


    # Clear memory

    st.session_state.vector_db = None

    st.session_state.knowledge_ready = False


    # Delete uploaded documents

    if os.path.exists(UPLOAD_FOLDER):

        shutil.rmtree(
            UPLOAD_FOLDER,
            ignore_errors=True
        )


    # Delete vector database

    if os.path.exists(VECTOR_DB):

        shutil.rmtree(
            VECTOR_DB,
            ignore_errors=True
        )


    # Recreate empty upload folder

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )


    # Force a brand new file_uploader widget instance (both server and
    # browser side) instead of trying to reset the old one - this is
    # what actually prevents deleted files from being "re-uploaded"
    # when the widget's stale browser-side state gets read again on
    # the next rerun (e.g. when submitting a question).
    st.session_state.uploader_key += 1


    # Lightweight confirmation that survives the rerun below
    st.toast("Documents and Knowledge Base deleted successfully.", icon="✅")


    # Restart the script cleanly from the top instead of halting it.
    # st.stop() would have prevented everything below this block
    # (including the Ask Questions section) from ever rendering again.
    st.rerun()



# ====================================================
# Automatic Processing After Upload
# ====================================================

if uploaded_files:


    for uploaded_file in uploaded_files:


        file_path = os.path.join(
            UPLOAD_FOLDER,
            uploaded_file.name
        )


        with open(file_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )


    st.success(
        f"{len(uploaded_files)} document(s) uploaded."
    )


    pdf_files = list(
        Path(UPLOAD_FOLDER).glob("*.pdf")
    )


    documents = []


    with st.spinner(
        "📄 Reading PDF documents..."
    ):


        for pdf in pdf_files:

            loader = PyPDFLoader(
                str(pdf)
            )

            documents.extend(
                loader.load()
            )


    with st.spinner(
        "✂️ Creating document chunks..."
    ):


        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200

        )


        chunks = splitter.split_documents(
            documents
        )

        # PDF metadata can contain None values or complex types Chroma
        # can't store (only str/int/float/bool are supported) - this
        # strips anything unsupported so it doesn't crash the upsert
        chunks = filter_complex_metadata(chunks)


    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )


    # Remove previous vector database

    if os.path.exists(VECTOR_DB):

        shutil.rmtree(
            VECTOR_DB,
            ignore_errors=True
        )


    with st.spinner(
        "🧠 Creating embeddings and knowledge base..."
    ):


        st.session_state.vector_db = Chroma.from_documents(

            documents=chunks,

            embedding=embeddings,

            persist_directory=VECTOR_DB

        )


    st.session_state.knowledge_ready = True


    st.success(
        "Knowledge Base created automatically!"
    )


    # Processing Summary

    st.subheader(
        "📊 Processing Summary"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "PDF Files",
            len(pdf_files)
        )


    with c2:

        st.metric(
            "Pages",
            len(documents)
        )


    with c3:

        st.metric(
            "Chunks",
            len(chunks)
        )


    st.info(
        f"""
Embedding Model:
{EMBEDDING_MODEL}

Vector Database:
ChromaDB
"""
    )



# ====================================================
# Ask Questions
# ====================================================

st.divider()

st.header("💬 Ask Questions")


# Placeholder declared here, right above the chat input - Streamlit
# keeps this container pinned to this spot in the page even though
# we don't fill it with content until later in the script. Since
# st.chat_input is always CSS-pinned to the very bottom of the page
# regardless of code order, this is what makes the answer appear
# directly above the input box instead of at the very end of the page.
answer_area = st.container()


question = st.chat_input(
    "Ask a question about your documents..."
)



if question:


    if not st.session_state.knowledge_ready:

        with answer_area:
            st.warning(
                "Please upload documents first."
            )

        st.stop()



    with answer_area:

        retriever = st.session_state.vector_db.as_retriever(

            search_kwargs={
                "k": 4
            }

        )


        with st.spinner(
            "🔎 Searching documents..."
        ):

            docs = retriever.invoke(
                question
            )


        context = ""


        for doc in docs:

            context += (
                doc.page_content
                + "\n\n"
            )


        prompt = f"""

You are an AI document assistant.

Answer ONLY using the supplied context.

If the answer is not available, say:

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


        llm = ChatOpenAI(

            model=CHAT_MODEL,

            temperature=0

        )


        with st.spinner(
            "🤖 Generating answer..."
        ):


            response = llm.invoke(
                prompt
            )


        st.subheader(
            "Answer"
        )


        st.write(
            response.content
        )


        with st.expander(
            "📚 Retrieved Context"
        ):


            for i, doc in enumerate(

                docs,

                start=1

            ):


                st.markdown(
                    f"### Chunk {i}"
                )


                st.write(
                    doc.page_content
                )
