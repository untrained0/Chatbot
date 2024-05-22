import os
import re
import google.generativeai as genai
from pypdf import PdfReader
from typing import List
import chromadb
from dotenv import load_dotenv
load_dotenv()

# Load PDF Function
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load PDFs from Directory Function
def load_pdfs_from_directory(directory_path):
    combined_text = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            combined_text += load_pdf(file_path)
    return combined_text

# Split Text Function
def split_text(text: str):
    split_text = re.split(r'\n\s*\n', text)
    return [i for i in split_text if i.strip() != ""]

# Define GeminiEmbeddingFunction
class GeminiEmbeddingFunction:
  def __call__(self, input: list) -> list:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = "models/embedding-001"
    title = "Custom query"
    response = genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)
    return [embedding["embedding"] for i, embedding in enumerate(response)]  # Use enumerate for index and element access

# Create ChromaDB Function
def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])

    return db, name

# Load Chroma Collection Function
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name)
    return db

# Get Relevant Passage Function
def get_relevant_passage(query, db, n_results):
    response = db.query(query_texts=[query], n_results=n_results)
    return response['documents'][0]

# Make RAG Prompt Function
def make_rag_prompt(query, relevant_passage):
    return f"Query: {query}\nRelevant Passage: {relevant_passage}\nAnswer:"

# Generate Answer Function
def generate_text_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = "models/generative-001"
    response = genai.generate_text(model=model, prompt=prompt)
    return response['generated_text']

# Get Answer Function
def get_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_text_answer(prompt)
    return answer

# Load and Process PDFs
directory_path = r"data"  # Replace with your directory path
all_pdfs_text = load_pdfs_from_directory(directory_path)
chunked_text = split_text(text=all_pdfs_text)

# Create ChromaDB with Documents
db, name = create_chroma_db(documents=chunked_text, 
                            path="contents",  # Replace with your path
                            name="Piyush5")

# Generate Answer
db = load_chroma_collection(path="contents", name="Piyush5")  # Replace with your path and collection name
query = "what is the purpose of SOP?"
answer = get_answer(db, query)
print(answer)
