import streamlit as st
import pickle
import requests
from langchain.llms.huggingface_hub import HuggingFaceHub  # Updated import
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA  # Updated import

# Streamlit App Title
st.title("CROW: News Research Tool")
st.sidebar.title("News Article URLs")

# Hugging Face API key from Streamlit Cloud Secrets
hf_api_key = st.secrets["HF_API_KEY"]

if not hf_api_key:
    st.error("Please set your Hugging Face API token in the Streamlit Cloud Secrets as 'HF_API_KEY'.")
    st.stop()

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

# Process URLs if button clicked
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

# Initialize the Hugging Face Hub LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=hf_api_key
)

if process_url_clicked and urls:
    try:
        # Load documents from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Ensure data is extracted
        if not data:
            st.error("No data extracted. Check URLs.")
            st.stop()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents created after splitting.")
            st.stop()

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_hf = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Building Embedding Vector Store...✅✅✅")

        # Save the vector store
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_hf, f)

        st.success("URLs processed successfully!")

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Handle user query
query = main_placeholder.text_input("Question: ")
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
