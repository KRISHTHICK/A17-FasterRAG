# A17-FasterRAG
GenAI

You are right, the current implementation can be slow, especially for larger PDF documents. The main bottlenecks are:

1.  **Embedding Generation:** Creating embeddings for all the document chunks can be time-consuming.
2.  **LLM Inference:** Generating the final answer by the Gemma model also takes time, and this is inherent to the model's processing speed on your local machine.

Here's an improved version of the code that addresses the speed issue by using a more efficient embedding model and provides a basic indication of answer accuracy.

```python
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
```

**Key Changes for Speed and Accuracy Indication:**

1.  **Faster Embedding Model:**
    * Replaced `OllamaEmbeddings` with `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`.
    * `all-MiniLM-L6-v2` is a highly efficient and relatively fast sentence transformer model that provides good performance for many RAG applications. You'll need to install the `sentence-transformers` library:
        ```bash
        pip install sentence-transformers
        ```

2.  **Reduced Context Documents (Optional):**
    * In the `RetrievalQA` setup, the `retriever`'s `as_retriever()` method now includes `search_kwargs={"k": 3}`. This tells the retriever to fetch only the top 3 most relevant document chunks. Reducing `k` can significantly speed up the process, but it might slightly reduce accuracy if the most relevant information is spread across more chunks. Adjust this value as needed.

3.  **Processing Time Feedback:**
    * The code now uses `time.time()` to measure the duration of document processing and LLM inference for both RAG and non-RAG answers. This provides direct feedback on how long each step takes.

4.  **Basic Accuracy Indication (Source Documents):**
    * For the RAG answer, the code now attempts to display the source document snippets that were used to generate the answer. This gives you a direct view of the information the bot relied on, helping you assess the relevance and thus, indirectly, the potential accuracy of the answer.
    * A warning message is shown if no relevant document snippets are found.

**Further Potential Optimizations (More Advanced):**

* **Vector Database Persistence:** Instead of creating the `vectorstore` every time a PDF is uploaded, you could persist it to disk (using Chroma's persistence features or other vector databases like FAISS). This way, if the same PDF is uploaded again, you can load the existing vector store, saving significant processing time.
* **Smaller Chunk Size (with Caution):** While smaller chunk sizes can sometimes improve granularity, they can also increase the number of chunks and thus the embedding time. Experiment with the `chunk_size` and `chunk_overlap` parameters.
* **Hardware Acceleration:** If you have a compatible GPU, you could explore using Ollama with GPU acceleration, which can significantly speed up LLM inference.
* **Asynchronous Operations (Advanced Streamlit):** For a more responsive UI, you could explore running the document processing and LLM inference asynchronously using Streamlit's `st.experimental_async` (or the newer `asyncio` integration). This would prevent the UI from freezing while these operations are in progress.

By implementing the changes in the provided code, you should experience a noticeable improvement in the response times, especially for the RAG-based answers. The inclusion of source documents will also give you a better understanding of the basis for the answers. Remember to install the `sentence-transformers` library.
