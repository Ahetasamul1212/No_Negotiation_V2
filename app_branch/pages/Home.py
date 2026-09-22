import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="No Negotiation Digital Law Handbook",
    layout="wide"
)
@st.cache_resource
def load_data():
    data = joblib.load(r"E:\my_projects\NRC_MODEL_DEV\test\pages\international_law_embeddings.joblib")
    
    model = SentenceTransformer(data["model_name"])
    
    return data, model


data, model = load_data()

embeddings = data["embeddings"]
texts = data["texts"]
metadata = data["metadata"]

st.title("No Negotiation Digital Law Handbook")

st.write(
    "This app will allow users to carry a whole lawbook in their pocket.\n\n"
    "Not just a bulky PDF, but an actual chatbot.\n\n"
    "Prerequisites:\n"
    "1) Adjust the slider on how much resources to be presented.\n"
    "2) Only relevant and correct spelling is applicable.\n"
    "3) Some of the sources might be empty; the user can simply switch to another one.\n"
    "4) If something goes wrong, try re-prompting the chat or feel free to ask me."
)

st.markdown("---")

top_k = st.slider(
        "Number of relevant srcs",
        min_value=1,
        max_value=10,
        value=5
    )

st.info(
        f"Loaded {len(texts):,} text chunks "
        f"from {data['pdf_name']}."
    )

question = st.text_input(
    "🔍 Search your legal advices:",
    placeholder="Example: What are the consequences of stealing?"
)


# --------------------------------------------------
# Search
# --------------------------------------------------

if question:

    with st.spinner("Wait give me some time to search...."):
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )[0]

        similarities = np.dot(
            embeddings,
            query_embedding
        )
        top_indices = np.argsort(
            similarities
        )[-top_k:][::-1]

    st.subheader("Relevant passages")

    for rank, idx in enumerate(top_indices, start=1):

        score = similarities[idx]
        text = texts[idx]
        page = metadata[idx]["page"]
        chunk_id = metadata[idx]["chunk_id"]

        with st.expander(
            f"Result {rank} — Page {page} — Similarity: {score:.3f}"
        ):

            st.write(text)

            st.caption(
                f"Chunk ID: {chunk_id} | "
                f"Page: {page} | "
                f"Similarity score: {score:.3f}"
            )