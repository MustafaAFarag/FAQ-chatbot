from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader    
from langchain.prompts import PromptTemplate  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import CTransformers  
from langchain.chains import RetrievalQA  
import chainlit as cl
from pydantic import BaseModel
import os
from transformers import AutoTokenizer

DB_FAISS_PATH = 'faiss_db'
DATA_PATH = './data'

# Custom prompt template
custom_prompt_template = """You are a medical chatbot. Use the following pieces of information to answer the user's question.
If you don't know the answer, say that you don't know. Do not attempt to fabricate an answer.

Context: {context}
Question: {question}

Please provide a concise and factual response.
Helpful answer:
"""

def clean_text(text):
    """Remove newline characters and leading/trailing whitespace."""
    return text.replace('\n', ' ').strip()

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code for handling messages
@cl.on_chat_start
async def start():
    chain = qa_bot()  # Initialize the chatbot model
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi there, Welcome to Medical Bot. Please enter your Query Below"
    await msg.update()

    cl.user_session.set("chain", chain)  # Store the chain in the session

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # Retrieve the stored chain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        # Append a simple message about the source without the full path
        answer += "\nSources: Extracted from trusted medical references."
    else:
        answer += "\nNo sources found."

    await cl.Message(content=answer).send()  # Send the response back to the user

# # Start the Chainlit app and ensure it runs a server
# if __name__ == "__main__":
#     print("Starting Chainlit server...")
#     cl.launch()