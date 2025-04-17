from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Using a faster embedding model
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import tempfile
import streamlit as st
import time  # For measuring processing time

# Title
st.title("⚡️ RAG Chatbot using Ollama (Local & Faster)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize vectorstore and llm outside the conditional block
vectorstore = None
llm = Ollama(model="gemma")

if uploaded_file:
    start_time = time.time()
    with st.spinner("Processing document..."):
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        # Load & Split PDF
        loader = PyPDFLoader(tmp_pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Embedding + Vector Store (Using a faster model)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embedding=embeddings)

        # Remove temporary file
        os.remove(tmp_pdf_path)
    processing_time = time.time() - start_time
    st.success(f"Document processed in {processing_time:.2f} seconds.")

# User Input
user_question = st.text_input("Ask a question:")

if user_question:
    st.markdown("### 🤖 Response")

    # Option 1: Answer using RAG if vectorstore is available
    st.subheader("Answer using Document (RAG)")
    if vectorstore:
        start_time = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Adjust 'k' for more/fewer context docs
            return_source_documents=True
        )
        rag_result = qa_chain.invoke({"query": user_question})
        rag_answer = rag_result["result"]
        end_time = time.time()
        inference_time_rag = end_time - start_time
        st.write(rag_answer)
        st.info(f"RAG response generated in {inference_time_rag:.2f} seconds.")
        st.session_state.history.append(("You", user_question))
        st.session_state.history.append(("Bot (RAG)", rag_answer))

        # Basic Accuracy Indication (based on source documents)
        if rag_result.get("source_documents"):
            st.markdown("💡 *Based on the following document snippets:*")
            for doc in rag_result["source_documents"]:
                st.markdown(f"<small>{doc.page_content[:200]}...</small>") # Show first 200 chars
        else:
            st.warning("No relevant document snippets found for this answer.")

    else:
        st.info("Please upload a PDF document to use the document-based answer.")

    # Option 2: Answer without using RAG (just the base LLM)
    st.subheader("Answer without Document")
    start_time = time.time()
    base_llm_answer = llm.invoke(user_question)
    end_time = time.time()
    inference_time_base = end_time - start_time
    st.write(base_llm_answer)
    st.info(f"Direct response generated in {inference_time_base:.2f} seconds.")
    st.session_state.history.append(("You", user_question))
    st.session_state.history.append(("Bot (No RAG)", base_llm_answer))

# Display chat history
if st.session_state.history:
    st.markdown("### 💬 Chat History")
    for speaker, message in st.session_state.history:
        st.write(f"**{speaker}:** {message}")
