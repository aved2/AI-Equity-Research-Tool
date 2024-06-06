import os
import streamlit as st
import time
import langchain
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from dotenv import load_dotenv

load_dotenv()

#UI
st.title("AI Equity Research Assistant")
st.caption("Analyze investment opportunities more efficiently with AI")
query = st.text_input("Ask a question", key="query")
st.sidebar.header("Sources", divider="rainbow")
st.sidebar.caption("Enter the URLs of the sources you want to analyze. Certain sources may not be supported due to their privacy settings.")

urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url{i+1}")
    urls.append(url)
file_path = "vector_index.faiss"
main_placefolder = st.empty()
valid_url = True


if query:
    try:
        url_not_empty = False
        #Check if url empty or invalid
        for url in urls:
            if url:
                url_not_empty = True
            if url and not url.startswith("http"):
                valid_url = False
                break
        if not url_not_empty:
            st.sidebar.warning("Please enter at least one source URL")
        elif valid_url == False:
            st.sidebar.warning("Please enter a valid URL")
        else:
            #load URLs
            loader = UnstructuredURLLoader(urls)
            main_placefolder.text("Loading data from URLs...")
            data = loader.load()

            #split data into chunks
            main_placefolder.text("Splitting data into chunks...")
            rsplitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = rsplitter.split_documents(data)
            #docs2 = rsplitter.split_text(text)
            #docs.extend(docs2)

            #create text embeddings
            main_placefolder.text("Creating vector embeddings...")
            embeddings = OpenAIEmbeddings()
            vectorindex_openai = FAISS.from_documents(docs, embeddings)

            #save to FAISS index
            faiss.write_index(vectorindex_openai.index, file_path)

            main_placefolder.text("Retrieving your answer...")
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        vectorindex_openai.index = faiss.read_index(file_path)
        llm = OpenAI(openai_api_key= st.secrets["YOUR_API_KEY"], temperature=0.9, max_tokens=500)
        retriever = vectorindex_openai.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = retriever)
        result = chain({"question": query}, return_only_outputs=True)
        main_placefolder.text("")
        st.header("Answer")
        st.write(result["answer"])

        #Display sources
        sources = result.get("sources", )
        if sources:
            st.header("Sources")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)
    except Exception as e:
        st.error("An error occurred. Please try a different URL.")
        print(e)




    
