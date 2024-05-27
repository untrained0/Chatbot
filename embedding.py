import os
from openai import OpenAI
import pandas as pd
from PyPDF2 import PdfReader
import numpy as np
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

client = OpenAI()


# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_pdfs_in_directory(directory_path):
    pdf_texts = []
    filenames = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            pdf_text = get_pdf_text(pdf_path)
            pdf_texts.append(pdf_text)
            filenames.append(filename)
    
    return filenames, pdf_texts

def embed_pdf_texts(pdf_texts):
    embeddings = []
    for text in pdf_texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

def save_embeddings_to_csv(filenames, embeddings, output_csv_path):
    df = pd.DataFrame({'filename': filenames, 'embedding': embeddings})
    df.to_csv(output_csv_path, index=False)

def main():
    directory_path = "data"
    output_csv_path = "output/embedded_pdfs.csv"
    
    # Process PDFs in the directory
    filenames, pdf_texts = process_pdfs_in_directory(directory_path)
    
    # Embed PDF texts
    embeddings = embed_pdf_texts(pdf_texts)
    
    # Save embeddings to CSV
    save_embeddings_to_csv(filenames, embeddings, output_csv_path)
    print(f"Embeddings saved to {output_csv_path}")

if __name__ == "__main__":
    main()
