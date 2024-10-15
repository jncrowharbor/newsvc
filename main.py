import os
import requests
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

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

# Directory to store the Chroma vector store
vectorstore_path = "chroma_vectorstore"
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
        docs = text_splitter.create_documents([file_content])

        if not docs:
            st.error("No documents created after splitting.")
            st.stop()

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Remove existing vector store if it exists
        if os.path.exists(vectorstore_path):
            import shutil
            shutil.rmtree(vectorstore_path)

        # Create a new Chroma vector store and persist it
        vectorstore_hf = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
        vectorstore_hf.persist()
        main_placeholder.text("Building Embedding Vector Store...✅✅✅")

        st.success("File processed successfully!")

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Handle user query
query = main_placeholder.text_input("Ask a question about the file: ")
if query and os.path.exists(vectorstore_path):
    # Load the persisted Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
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
