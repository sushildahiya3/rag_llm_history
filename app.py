import streamlit as st
from vipas import model
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
from docx import Document
import pandas as pd
import numpy as np
import gc


class RAGProcessor:
    def __init__(self, model_id):
        self.client = model.ModelClient()
        self.model_id = model_id
        self.embedding_model = None
        self.faiss_index = None
        self.chunks = []
        self.last_file_name = None

    def load_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def preprocess_document(self, file):
        try:
            text = ""
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    text = "".join(page.extract_text() or "" for page in pdf.pages)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(file)
                text = " ".join(para.text for para in doc.paragraphs)
            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(file)
                text = data.to_string(index=False)
            else:
                st.error("Unsupported file type. Please upload a PDF, DOCX, or Excel file.")
            return text.strip()
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return ""

    def store_embeddings(self, text, batch_size=32):
        self.load_embedding_model()

        # Generate text chunks
        self.chunks = [text[i : i + 500] for i in range(0, len(text), 500) if text[i : i + 500].strip()]
        if not self.chunks:
            st.error("No valid text found in the document.")
            return None

        # Initialize FAISS index only when needed
        self.faiss_index = faiss.IndexFlatL2(384)
        embeddings = []

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, batch_size=8, show_progress_bar=True)
            embeddings.append(batch_embeddings)
            self.faiss_index.add(np.array(batch_embeddings, dtype=np.float32))

        # Clear intermediate data
        gc.collect()
        return self.chunks

    def retrieve_context(self, query):
        if self.faiss_index is None or not self.chunks:
            st.error("No document is indexed. Please upload a file first.")
            return ""

        self.load_embedding_model()
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, k=5)

        return " ".join(self.chunks[i] for i in indices[0])

    def query_llm(self, query, context):
        prompt = (
            "You are an expert. Answer the question using the provided context:\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        try:
            response = self.client.predict(model_id=self.model_id, input_data=prompt)
            return response.get("choices", [{}])[0].get("text", "No response text available.")
        except Exception as e:
            st.error(f"Error querying the LLM: {e}")
            return ""


# Streamlit session state setup
if "rag_processor" not in st.session_state:
    st.session_state.rag_processor = RAGProcessor(model_id="mdl-hy3grx9aoskqu")
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

rag_processor = st.session_state.rag_processor

# Streamlit app
st.title("RAG-based Q&A with Llama")
st.write("Upload a document (PDF, DOCX, or Excel) under 2 MB and ask questions using the LLM.")

# File upload
uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, or Excel):", type=["pdf", "docx", "xlsx"])
if uploaded_file:
    file_size = uploaded_file.size / (1024 * 1024)  # File size in MB
    if file_size > 2:
        st.error("File size exceeds 2MB. Please upload a smaller file.")
    else:
        st.success(f"File '{uploaded_file.name}' uploaded successfully! File size: {file_size:.2f} MB.")
        process_button = st.button("Submit File")
        if process_button:
            file_name = uploaded_file.name
            if file_name != rag_processor.last_file_name:
                st.write("Processing the uploaded file...")
                text = rag_processor.preprocess_document(uploaded_file)

                if text:
                    chunks = rag_processor.store_embeddings(text)
                    if chunks:
                        rag_processor.last_file_name = file_name
                        st.success("Document processed and indexed successfully!")

# Query and response handling
if rag_processor.last_file_name and rag_processor.faiss_index is not None:
    query = st.text_input("Enter your query:")

    if st.button("Query", disabled=not bool(query)):
        context = rag_processor.retrieve_context(query)
        st.write("Retrieved Context:")
        st.write(context)

        st.write("Generating response from LLM...")
        response = rag_processor.query_llm(query, context)
        st.write("### Response")
        st.write(response)

        # Add to history
        st.session_state.qa_history.append({"question": query, "response": response})

# Display QA history
if st.session_state.qa_history:
    st.write("### History of Questions and Answers")
    for i, qa in enumerate(st.session_state.qa_history, start=1):
        st.write(f"**Q{i}:** {qa['question']}")
        st.write(f"**A{i}:** {qa['response']}")
