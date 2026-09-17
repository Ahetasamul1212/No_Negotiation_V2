import fitz
import faiss
import numpy as np
import os
import sqlite3
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------- CONFIGURATION -------------------
PDF_FILE = r"HuggingFace_embeder_chainlit\modeler\UKIL-DB-EN_train.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "legal_faiss.index"
DB_FILE = "complaints.db"

# ------------------- Sentence Embedder -------------------
model = SentenceTransformer(EMBEDDING_MODEL)

# ------------------- Feature Extraction -------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF and returns as a single string."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# ------------------- Chunk split -------------------
def split_text(text, chunk_size=500, chunk_overlap=100):
    """Splits text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# ------------------- FAISS Embedding -------------------
def store_embeddings(chunks):
    """Generates and stores embeddings in FAISS index."""
    embeddings = model.encode(chunks, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save("legal_text_chunks.npy", np.array(chunks))
    st.success("Embeddings stored successfully in FAISS!")

# ------------------- Query FAISS -------------------
def query_faiss(user_input, top_k=3):
    """Queries FAISS for the closest legal text based on user input."""
    if not os.path.exists(FAISS_INDEX_FILE):
        return ["FAISS index not found. Please run the script to build the database first."]
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    text_chunks = np.load("legal_text_chunks.npy", allow_pickle=True)
    query_embedding = model.encode([user_input]).astype("float32")
    D, I = index.search(query_embedding, k=top_k)
    
    return [text_chunks[i] for i in I[0] if i < len(text_chunks)]

# ------------------- Complaint Database -------------------
def init_db():
    """Initialize the SQLite database for complaints."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            complaint TEXT
        )
    """)
    conn.commit()
    conn.close()

def submit_complaint(user_name, complaint_text):
    """Handles complaint submission and stores it in the database."""
    if not user_name or not complaint_text.strip():
        st.error("Please enter your name and a complaint.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO complaints (user_name, complaint) VALUES (?, ?)", (user_name, complaint_text))
    conn.commit()
    conn.close()
    st.success("Your complaint has been recorded!")

def export_complaints_to_csv():
    """Exports complaint data to a CSV file."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM complaints", conn)
    conn.close()
    csv_file = "complaints.csv"
    df.to_csv(csv_file, index=False)
    st.download_button("Download Complaints CSV", csv_file, file_name="complaints.csv")

# ------------------- FAISS Index -------------------
if not os.path.exists(FAISS_INDEX_FILE):
    st.warning("FAISS index not found! Extracting and storing embeddings...")
    pdf_text = extract_text_from_pdf(PDF_FILE)
    text_chunks = split_text(pdf_text)
    store_embeddings(text_chunks)
else:
    st.success("FAISS index already exists. Ready to query.")

init_db()

# ------------------- STREAMLIT UI -------------------
st.title("No Negotiation Legal Search Engine")

# Legal Query Search
st.subheader("Search Legal Text")
user_query = st.text_input("Enter Legal Query")
if st.button("Search"):
    results = query_faiss(user_query)
    if results:
        for i, result in enumerate(results):
            st.write(f"**{i+1}.** {result}")
            st.markdown("---")
    else:
        st.warning("No relevant legal text found.")

# Complaint Submission
st.subheader("Submit a Complaint")
user_name = st.text_input("Your Name")
complaint_text = st.text_area("Enter Your Complaint")
if st.button("Submit Complaint"):
    submit_complaint(user_name, complaint_text)

# Export Complaints
st.subheader("Export Complaints")
if st.button("Export to CSV"):
    export_complaints_to_csv()
