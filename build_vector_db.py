"""
Builds the Vector Database from PDFs.

Run:

python build_vector_db.py

"""

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found in .env")

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

DATA_FOLDER = "data"

VECTOR_DB = "vector_db"

EMBEDDING_MODEL = "text-embedding-3-small"

# -------------------------------------------------------
# Load PDFs
# -------------------------------------------------------

documents = []

pdf_folder = Path(DATA_FOLDER)

pdf_files = list(pdf_folder.glob("*.pdf"))

if len(pdf_files) == 0:
    raise Exception("No PDF files found inside /data folder.")

print(f"\nFound {len(pdf_files)} PDF(s)\n")

for pdf in pdf_files:

    print(f"Loading {pdf.name}")

    loader = PyPDFLoader(str(pdf))

    docs = loader.load()

    documents.extend(docs)

print(f"\nLoaded {len(documents)} pages.\n")

# -------------------------------------------------------
# Split into Chunks
# -------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.\n")

# -------------------------------------------------------
# Embedding Model
# -------------------------------------------------------

embedding_model = OpenAIEmbeddings(

    model=EMBEDDING_MODEL

)

# -------------------------------------------------------
# Create Chroma Vector Database
# -------------------------------------------------------

print("Generating embeddings...")

db = Chroma.from_documents(

    documents=chunks,

    embedding=embedding_model,

    persist_directory=VECTOR_DB

)

print("\n===================================")
print("Vector Database Created Successfully")
print("===================================\n")

print(f"Location : {VECTOR_DB}")

print(f"Chunks   : {len(chunks)}")

print(f"Model    : {EMBEDDING_MODEL}")