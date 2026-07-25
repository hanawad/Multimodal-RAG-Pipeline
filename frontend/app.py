import requests
import streamlit as st

# Backend service address
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title Section ---
st.title("🤖 Multimodal RAG Assistant & Teaching Demo")
st.caption("An interactive platform demonstrating Retrieval-Augmented Generation across Documents, Images, and Tabular Datasets.")

# --- System Architecture Diagram ---
with st.expander("🌐 View RAG Pipeline Architecture", expanded=False):
    st.graphviz_chart(
        """
        digraph Architecture {
            rankdir=LR;
            node [shape=rectangle, style="rounded,filled", fillcolor="#f0f2f6", fontname="Sans-serif"];
            
            subgraph cluster_inputs {
                label = "Supported Inputs";
                style=dashed;
                PDF; DOCX; CSV; Image;
            }

            PDF -> TextSplitter [label=" Extract Text "];
            DOCX -> TextSplitter;
            CSV -> RowParser [label=" Tabular Rows "];
            Image -> VisionModel [label=" Vision OCR "];
            
            TextSplitter -> Embeddings;
            RowParser -> Embeddings;
            VisionModel -> Embeddings;
            
            Embeddings -> VectorDB [label=" Index Store "];
            
            UserQuery -> VectorDB [label=" Similarity Search "];
            VectorDB -> ContextChunks;
            
            ContextChunks -> AugmentedPrompt;
            UserQuery -> AugmentedPrompt;
            
            AugmentedPrompt -> LLM Engine;
            LLM Engine -> FinalResponse;
        }
        """
    )

st.divider()

# --- Section 1: Data Ingestion & Upload ---
st.header("1. Document Ingestion")
st.markdown("Upload files (`.pdf`, `.docx`, `.csv`, `.png`, `.jpg`) to extract, embed, and store in vector memory.")

uploaded_file = st.file_uploader(
    "Choose a file to index",
    type=["pdf", "png", "jpg", "jpeg", "csv", "docx"],
    help="Upload your document to augment model knowledge."
)

if uploaded_file is not None:
    # Preview image if uploaded
    if uploaded_file.type in ("image/png", "image/jpeg"):
        st.image(uploaded_file, caption=f"Uploaded Preview: {uploaded_file.name}", width=300)

    if st.button("🚀 Index Document", type="primary"):
        multipart_data = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }
        
        with st.spinner("Processing content, generating embeddings, and building index..."):
            try:
                response = requests.post(f"{API_BASE_URL}/upload", files=multipart_data)
                
                if response.status_code == 200:
                    payload = response.json()
                    file_kind = payload.get("file_type", "Unknown")
                    st.success(f"Successfully processed and indexed: **{uploaded_file.name}** ({file_kind.upper()})")

                    # Display extraction metrics using dynamic columns
                    metric_cols = st.columns(4)
                    if file_kind == "pdf":
                        metric_cols[0].metric("Total Pages", payload.get("pages", 0))
                        metric_cols[1].metric("Text Chunks", payload.get("chunks", 0))
                        metric_cols[2].metric("Extracted Figures", payload.get("images_extracted", 0))
                    elif file_kind == "csv":
                        metric_cols[0].metric("Parsed Rows", payload.get("total_rows", 0))
                        metric_cols[1].metric("Columns", len(payload.get("columns", [])))
                    elif file_kind == "docx":
                        metric_cols[0].metric("Document Chunks", payload.get("chunks", 0))
                    elif file_kind == "image":
                        st.info("Visual attributes extracted via Vision Multimodal OCR.")
                else:
                    st.error(f"Ingestion failed with status code: {response.status_code}")
            except Exception as err:
                st.error(f"Could not connect to backend service: {err}")

st.divider()

# --- Section 2: Query & Knowledge Synthesis ---
st.header("2. Knowledge Query & Model Comparison")
user_prompt = st.text_input("Enter your query:", placeholder="Ask a question about the uploaded document...")

if st.button("🔍 Submit Query"):
    if not user_prompt.strip():
        st.warning("Please enter a valid question before querying.")
    else:
        with st.spinner("Retrieving knowledge context and invoking LLM..."):
            try:
                api_response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": user_prompt.strip()}
                )

                if api_response.status_code == 200:
                    result_data = api_response.json()

                    # Response Comparison Tabs
                    tab_rag, tab_norag = st.tabs(["✨ Grounded Response (With RAG)", "🤖 Baseline Response (Without RAG)"])

                    with tab_rag:
                        st.success("Answer generated using retrieved context:")
                        st.write(result_data.get("rag_answer"))

                    with tab_norag:
                        st.info("Answer generated strictly from LLM general knowledge:")
                        st.write(result_data.get("no_rag_answer"))

                    # Section 3: Retrieved Context Details
                    st.subheader("3. Retrieved Context Chunks")
                    retrieved_chunks = result_data.get("retrieved_chunks", [])

                    if not retrieved_chunks:
                        st.warning("No relevant chunks were retrieved for this query.")
                    else:
                        for idx, chunk_info in enumerate(retrieved_chunks, start=1):
                            src_name = chunk_info.get("source", "N/A")
                            page_idx = chunk_info.get("page", "N/A")
                            dist_score = chunk_info.get("score", 0.0)

                            with st.expander(f"Chunk #{idx} | Source: {src_name} | Page/Row: {page_idx}"):
                                st.markdown(f"**Similarity Score:** `{dist_score:.4f}`")
                                st.markdown("---")
                                st.text(chunk_info.get("content", ""))
                else:
                    st.error("Error retrieving answers from server.")
            except Exception as conn_err:
                st.error(f"Connection to backend failed: {conn_err}")