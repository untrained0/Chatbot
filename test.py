from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore
from langchain.llms import OpenAI
import os
import pandas as pd
from flask import Flask, request, jsonify
from waitress import serve
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the embedding model and set the service context
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

def construct_index(directory_path):
    index = None
    docstore = SimpleVectorStore()
    index_store = SimpleVectorStore()
    vector_store = SimpleVectorStore()
    graph_store = SimpleVectorStore()
    storage_context = StorageContext(docstore=docstore, index_store=index_store, vector_stores=[vector_store], graph_store=graph_store)

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            print(filename)
            docs = SimpleDirectoryReader(input_files=[os.path.join(directory_path, filename)]).load_data()
            if index is None:
                index = VectorStoreIndex.from_documents(docs, service_context=service_context, storage_context=storage_context, show_progress=True)
            else:
                parser = SimpleNodeParser()
                new_nodes = parser.get_nodes_from_documents(docs)
                index.insert_nodes(new_nodes)

    if index is not None:
        index.set_index_id("brsr_index")
        index.storage_context.persist()
        print("Stored index")
    return index

def chatbot(input_text):
    embeddings = OpenAIEmbedding()
    faiss_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query_engine = faiss_store.as_query_engine()
    response = query_engine.query(input_text)
    print(response)
    return response

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbedding()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = OpenAIEmbedding()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    index = construct_index(r"data")
    main()
