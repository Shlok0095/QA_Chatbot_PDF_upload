import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import cohere
from PyPDF2 import PdfReader

# Load Cohere client (replace with your API key)
cohere_api_key = "bCos1WNgZSvrBLWSYGfG6yiDO638h1k3y4Whjr0X"
co_client = cohere.Client(cohere_api_key)

# Load the embedding model and FAISS index
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Replace with your fine-tuned model
index = faiss.read_index('faiss_index.bin') 

df = pd.read_csv('data.csv')  # Load document segments

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to embed and add new data to FAISS index
def embed_and_add(texts):
    embeddings = embed_model.encode(texts)
    index.add(np.array(embeddings).astype('float32'))
    return embeddings

# Function to perform RAG QA using Cohere
def rag_qa(query, top_k=3):
    query_embedding = embed_model.encode([query])[0]
    _, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    
    contexts = [df['text'].iloc[i] for i in I[0]]
    
    # Join contexts, but limit to ~1000 tokens
    context = " ".join(contexts)[:4000]  # Approximate 1000 tokens
    
    input_text = f"""Context: {context}

Question: {query}

Instructions: Provide a single, focused answer to the question based on the given context. If the context doesn't contain relevant information, state that you don't have enough information to answer accurately.

Answer:"""
    
    response = co_client.generate(
        model='command-xlarge-nightly',
        prompt=input_text,
        max_tokens=150,
        temperature=0.7,
        stop_sequences=["\n"]  # Stop at new line to prevent multiple answers
    )
    answer = response.generations[0].text.strip()
    return context, answer

# Streamlit layout
st.title("Document QA Bot")

# Upload PDF and ask questions directly
st.header("Upload PDF and Ask Questions")
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file is not None:
    text = extract_text_from_pdf(pdf_file)
    st.write("PDF successfully uploaded and text extracted.")
    
    # Embed the extracted text and add to FAISS index
    new_texts = [text[i:i+500] for i in range(0, len(text), 500)]
    embed_and_add(new_texts)
    
    # Use pd.concat instead of append
    df = pd.concat([df, pd.DataFrame(new_texts, columns=['text'])], ignore_index=True)
    
    st.write("PDF content embedded into FAISS index.")
    
    # Asking question based on the uploaded PDF
    query = st.text_input("Ask a question from the uploaded PDF:")
    
    if st.button("Get Answer"):
        if query:
            context, answer = rag_qa(query)
            st.write(f"Context: {context}")
            st.write(f"Answer: {answer}")
        else:
            st.write("Please enter a question.")