import subprocess
import sys
import streamlit as st
import pickle
import requests
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms.huggingface_hub import HuggingFaceHub
import os

# Function to install a package if it's not already installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Streamlit App Title
st.title("CROW: News Research Tool")

# Hugging Face API key from Streamlit Cloud Secrets
hf_api_key = st.secrets["HF_API_KEY"]

if not hf_api_key:
    st.error("Please set your Hugging Face API token in the Streamlit Cloud Secrets as 'HF_API_KEY'.")
    st.stop()

# Initialize the HuggingFaceHub LLM with API Key
llm = HuggingFaceHub(api_key=hf_api_key, model_name="gpt-neo-125M")

# Path to the .py file in your GitHub repository
github_py_url = st.text_input(
    "Enter the GitHub raw URL to the .py file",
    placeholder="https://raw.githubusercontent.com/yourusername/yourrepository/main/yourfile.py"
)

# Process the file once the button is clicked
process_file_clicked = st.button("Process File")

file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

if process_file_clicked and github_py_url:
    try:
        # Fetch the raw .py file from the GitHub URL
        response = requests.get(github_py_url)

        if response.status_code != 200:
            st.error(f"Failed to fetch the file. Status code: {response.status_code}")
            st.stop()

        # Extract the content of the file (as text)
        file_content = response.text
        main_placeholder.text("File Loading...Started...✅✅✅")

        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_text(file_content)

        if not docs:
            st.error("No documents created after splitting.")
            st.stop()

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_hf = FAISS.from_texts(docs, embeddings)
        main_placeholder.text("Building Embedding Vector Store...✅✅✅")

        # Save the vector store
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_hf, f)

        st.success("File processed successfully!")

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Handle user query
query = main_placeholder.text_input("Ask a question about the file: ")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()

    custom_prompt_template = """You are a knowledgeable assistant. Answer based on the extracted parts of a long document.
Question: {question}
Document: {context}
Answer:"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    # RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = chain({"query": query})

    # Display answer
    st.header("Answer")
    st.write(result["result"])

    # Display sources
    st.subheader("Sources:")
    sources = {doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]}
    for source in sources:
        st.write(f"- {source}")
