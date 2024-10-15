import os
import streamlit as st
import pickle
import time
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pyngrok import ngrok

# Load environment variables from .env
load_dotenv()

# Check if we're running locally by looking for a specific environment variable
is_local = os.getenv("IS_LOCAL", "false").lower() == "true"

# Only run ngrok if we're running locally
if is_local:
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        http_tunnel = ngrok.connect(8501)  # Creates a tunnel for port 8501 (Streamlit's default port)
        public_url = http_tunnel.public_url
        st.write(f"App is running publicly at: {public_url}")
    else:
        st.error("NGROK_AUTH_TOKEN is missing from environment variables.")

st.title("CROW: News Research Tool")
st.sidebar.title("News Article URLs")

# Hard-coded Hugging Face API key (Replace 'your_api_key_here' with your actual API key)
hf_api_key = st.secrets["HF_API_KEY"]

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

if hf_api_key:
    # Initialize the LLM using a suitable model from Hugging Face
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.7, "max_length": 512},
        huggingfacehub_api_token=hf_api_key  # Pass the API key directly
    )

    if process_url_clicked and urls:
        try:
            # Initialize the loader with the URLs
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()

            # Check if data was successfully extracted
            if not data:
                st.error("No data was extracted from the URLs. Please check the URLs or try different ones.")
                st.stop()

            # Manually set the 'source' metadata for each document
            for doc, url in zip(data, urls):
                doc.metadata['source'] = url

            # Reduce chunk size to handle large documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced chunk size to avoid context length issues
                chunk_overlap=100  # Some overlap to maintain context
            )
            main_placeholder.text("Text Splitting...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("No documents were created after splitting. Please check the content of the URLs.")
                st.stop()

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Create the vector store
            vectorstore_hf = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Building Embedding Vector Store...✅✅✅")
            time.sleep(2)

            # Save the vector store to a file for future queries
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_hf, f)

            st.success("URLs processed successfully!")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    # Query input for the user to ask questions based on the processed content
    query = main_placeholder.text_input("Question: ")
    if query and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()

        # Custom prompt template to generate better answers
        custom_prompt_template = """You are a knowledgeable assistant. Given the following extracted parts of a long document, provide a clear and structured answer to the user's question. If you don't know the answer, just say you don't know.

Question: {question}

Document: {context}

Answer:"""

        # Create the prompt
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # Initialize the RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Get the result of the query
        result = chain({"query": query})

        # Display the answer
        st.header("Answer")
        st.write(result["result"])

        # Display the sources of the information
        st.subheader("Sources:")
        sources = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
        for source in sources:
            st.write(f"- {source}")

else:
    st.error("API Key is missing or invalid.")
