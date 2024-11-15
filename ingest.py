import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# Define paths
DATA_PATH = "data/"
DB_FAISS_PATH = "/faiss_db"

# Create vector database
def create_vector_db():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(f"{DATA_PATH}/faqs.csv", encoding='ISO-8859-1')

    # Assuming each row is a document and converting them to a list of `Document` objects
    documents = [Document(page_content=' '.join(row.astype(str)), metadata={}) for _, row in df.iterrows()]

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Use HuggingFace embeddings for document embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store from the documents and embeddings
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS vector store locally
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()