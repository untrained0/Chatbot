import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import pytesseract

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
    return text

# def get_excel_text(excel_docs):
#     text = ""
#     for excel in excel_docs:
#         try:
#             df = pd.read_excel(excel)
#             text += df.to_string()
#         except Exception as e:
#             st.error(f"Error reading Excel file {excel.name}: {e}")
#     return text

# def get_image_text(image_files):
#     text = ""
#     for image in image_files:
#         try:
#             img = Image.open(image)
#             text += pytesseract.image_to_string(img)
#         except Exception as e:
#             st.error(f"Error reading image file {image.name}: {e}")
#     return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the provided context about Xangars Infratech Solutions Pvt. Ltd. If the answer is not in the context, say "The answer is not available in the context provided." Do not provide any information not included in the context.

    Context:
    {context}
    
    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Enable dangerous deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    # st.header("Chat with PDF, Excel, and Images using Gemini💁")
    st.header("Chat with using Gemini💁")

    user_question = st.text_input("Ask a Question from the Uploaded Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        # excel_docs = st.file_uploader("Upload your Excel Files", accept_multiple_files=True, type=['xlsx'])
        # image_files = st.file_uploader("Upload your Image Files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                # if excel_docs:
                #     raw_text += get_excel_text(excel_docs)
                # if image_files:
                #     raw_text += get_image_text(image_files)

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.warning("No files uploaded or no text extracted from the files.")

if __name__ == "__main__":
    main()
