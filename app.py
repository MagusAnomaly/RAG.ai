import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import os
from langchain.schema import Document  # Schema created in backend
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# API Configuration
genai.configure(api_key=os.getenv('GOOGLE-GEMINI-API'))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Cache the HF embeddings to avoid slow reload of the Embeddings
@st.cache_resource(show_spinner='Cooking stew for the wizard.....')
def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = embeddings()

# User interface
st.header("Vitus: The Wizard Apprentice")
st.subheader("HF Embeddings + Gemini LLM")

uploaded_file = st.file_uploader(label="Enter your parchment", type=['pdf'])

if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)
    for page in pdf.pages:  # ❗ Fixed loop over PdfReader, not uploaded_file
        text = page.extract_text()
        if text:
            raw_text += text

    if raw_text.strip():
        document = Document(page_content=raw_text)

        # Using char text splitter we will create chunks and pass it into the model
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # ❗ Fixed: chunks_size → chunk_size
        chunks = splitter.split_documents([document])

        # Store the chunks in the FAISS Vector DB
        chunk_pieces = [chunk.page_content for chunk in chunks]
        vectordb = FAISS.from_texts(chunk_pieces, embedding_model)
        retriever = vectordb.as_retriever()

        st.success('Document is fed into the stew')

        # User QnA
        user_input = st.text_input(label="What do u want to ask my master")  # ❗ Fixed: st.text → st.text_input

        if user_input:
            with st.chat_message('user'):
                st.write(user_input)

            with st.spinner('Waking my master...'):
                relevant_docs = retriever.get_relevant_documents(user_input)
                context = '\n\n'.join(doc.page_content for doc in relevant_docs)

                prompt = f"""
You are an ancient, scraggy, and sleepy wizard who has just been woken up by your eager apprentice. Despite your laziness and occasional grumbling, you are a master of deep analysis and arcane knowledge.

Respond in the tone of a tired, wise wizard reluctantly helping his apprentice. Be whimsical and in-character — but do not lose your analytical edge.

Use the following context to answer the query. If the information is not available or unclear, say:
"Bah! The scrolls don't whisper such things. Look into other sources, young one..."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

--- QUERY ---
{user_input}
--- ANSWER ---
"""
  # ❗ Fixed string formatting and indentation

                response = gemini_model.generate_content(prompt)
                st.markdown('This is what the wizard thinks')
                st.write(response.text)

    else:
        st.warning('Curses! , Theres nothing in the parchment.')
