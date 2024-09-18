import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

# Load environment variables
load_dotenv()

# Initialize global variables
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="UTF-8")
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()

        # Update metadata for each document
        for doc in documents:
            doc.metadata["source"] = file.name
            if "page" not in doc.metadata:
                doc.metadata["page"] = doc.metadata.get("page_number") or doc.metadata.get("row") or 1

        return documents
    except Exception as e:
        st.sidebar.error(f"Error loading file {file.name}: {str(e)}")
        return None
    finally:
        os.remove(temp_file_path)

def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Ensure metadata is carried over to the chunks
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "Unknown source"
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever()

def get_response(query, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        llm=llm,
        return_source_documents=True,
    )

    response = qa_chain({"query": query})
    answer = response["result"]
    source_documents = response["source_documents"]

    # Find the document with the highest relevance score
    if source_documents:
        most_relevant_doc = max(source_documents, key=lambda doc: doc.metadata.get('score', 0))
        source_metadata = most_relevant_doc.metadata
        source_name = source_metadata.get("source", "Unknown source")
        page_number = source_metadata.get("page", "N/A")
        return answer, {"source": source_name, "page": page_number}

    return answer, None

def sidebar_file_uploader():
    st.sidebar.title("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = []
        for file in uploaded_files:
            docs = load_document(file)
            if docs:
                documents.extend(docs)

        if documents:
            retriever = process_documents(documents)
            st.session_state["retriever"] = retriever
            st.sidebar.success(f"Processed the documents. You can now ask questions.")
        else:
            st.sidebar.warning("No valid documents were uploaded or processed. Please check the error messages and try again.")

def main():
    st.set_page_config(page_title="DocQuery", layout="wide")
    st.markdown("<h1 style='text-align: center;'>DocQuery</h1>", unsafe_allow_html=True)

    sidebar_file_uploader()

    st.write("Enter your query about the uploaded documents:")
    query = st.text_input("Query", key="query_input")

    if query and "retriever" in st.session_state:
        with st.spinner("Generating response..."):
            answer, source_info = get_response(query, st.session_state["retriever"])

        st.subheader("Response:")
        st.write(answer)

        if source_info:
            st.subheader("Source:")
            source_name = source_info["source"]
            page_number = source_info["page"]
            if isinstance(page_number, int):
                page_number += 1
            elif page_number.isdigit():  # Handle if page number is a string that represents a number
                page_number = str(int(page_number) + 1)
            st.write(f"Document: {source_name}")
            st.write(f"Page/Section: {page_number}")

    elif query:
        st.info("Please upload documents before asking questions.")
    else:
        st.info("Upload documents using the sidebar and then ask a question.")

if __name__ == "__main__":
    main()
