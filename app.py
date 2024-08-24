import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
load_dotenv()

# Import FAISS
from langchain.vectorstores.faiss import FAISS


# Load the API key from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")
# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# Initialize the embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit app
st.markdown(
    """
    <h1 style="text-align: center;"> DocQuery </h1>
    """,
    unsafe_allow_html=True,
)
# st.title("DocQuery")

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_file:
    documents = []
    for file in uploaded_file:
        # temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
        temp_file.write(file.read())
        temp_file_path = temp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_file_path, encoding="UTF-8")
        else:
            st.error(f"Unsupported file type: {file.type}")
            continue
        # close the temp file
        # Load and split the document
        try:
            document = loader.load_and_split()
            documents.extend(document)
            temp_file.close()
        except Exception as e:
            st.error(f"Error loading file {file.name}: {e}")
        finally:
            os.remove(temp_file_path)  # Ensure the temporary file is removed

    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    # st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Generate embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    st.session_state["retriever"] = retriever

    query = st.text_input("Enter your query about the uploaded documents:")

    if query and "retriever" in st.session_state:
        retriever = st.session_state["retriever"]
        
        query_embedding = embeddings.embed_query(query)
        # Retrieve documents with embeddings
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Extract text from retrieved documents and compute embeddings
        doc_texts = [doc.page_content for doc in retrieved_docs]
        doc_embeddings = [embeddings.embed_query(text) for text in doc_texts]
        doc_embeddings = np.array(doc_embeddings)

        # Calculate cosine similarity between query embedding and document embeddings
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Get the maximum similarity score
        max_similarity = max(similarities) if similarities.size > 0 else 0
        relevance_threshold = 0.6
        
        if max_similarity < relevance_threshold:
            answer = "Your query doesn't have an answer in the provided documents."
        else:
            qa_with_chain = RetrievalQA.from_chain_type(
                retriever=retriever, 
                llm=llm,
                return_source_documents=True,
            )

            response = qa_with_chain({"query": query})
            answer = response["result"]

        st.write("Response:")
        st.write(answer)
        
