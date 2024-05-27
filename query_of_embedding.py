import os
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the embeddings from the CSV file
df = pd.read_csv('output/embedded_pdfs.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# Extract filenames and embeddings
filenames = df['filename'].tolist()
embeddings = df['embedding'].tolist()

# Check if the lengths of filenames and embeddings are equal
if len(filenames) != len(embeddings):
  print("Error: The number of filenames and embeddings don't match!")
  # Optionally add logic to handle the mismatch here (e.g., exit or fix)
  # Here are some possible actions:
  #   - Exit the program: raise Exception("Unequal list lengths!")
  #   - Fix the lists: remove invalid entries or pad the shorter list (caution advised)
else:
  # Initialize the vector store with the filenames and embeddings
  vector_store = FAISS.from_embeddings(filenames, embeddings)

  # Save the vector store locally
  vector_store.save_local("faiss_index")

  # Initialize the Gemini model
  model = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)

  # Define a prompt template
  prompt_template = """
  Answer the question as detailed as possible from the provided context. 
  Make sure to provide all the details. If the answer is not in the provided context, 
  just say, "The answer is not available in the context." Don't provide a wrong answer.

  Context:
  {context}

  Question: 
  {question}

  Answer:
  """

  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

  # Load the question-answering chain
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

  def query_with_gemini(query, faiss_index_path="faiss_index"):
      # Load the FAISS vector store
      vector_store = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

      # Perform a similarity search with the query
      docs = vector_store.similarity_search(query)

      # Generate a response using the Gemini model
      response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

      return response["output_text"]

  # Example query
  query = "What is the main topic of the first PDF?"
  response = query_with_gemini(query)
  print(response)
