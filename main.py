from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from langchain_community.document_loaders import (
    TextLoader,
    PDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


def get_loader(file_path: str, file_type: str):
    """Get the appropriate loader based on file type"""
    loaders = {
        '.txt': TextLoader,
        '.pdf': PDFLoader,
        '.docx': Docx2txtLoader,
        '.md': UnstructuredMarkdownLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader
    }

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")

    return loaders[ext](file_path)


@app.get("/")
def home():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Load and process the document
            loader = get_loader(temp_file.name, file.filename)
            documents = loader.load()

            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Add to vector store
            vector_store.add_documents(splits)
            vector_store.persist()

            # Clean up
            os.unlink(temp_file.name)

            return {"message": "Document processed and added to vector store"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask")
def ask(prompt: str):
    try:
        # First, search the vector store for relevant context
        relevant_docs = vector_store.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Combine context with the prompt
        enhanced_prompt = f"Context: {context}\n\nQuestion: {prompt}"

        res = requests.post(
            "http://llama3.2-webui:11434/api/generate",
            json={
                "prompt": enhanced_prompt,
                "stream": False,
                "model": "llama3.2:latest"
            },
            timeout=30
        )
        res.raise_for_status()
        return Response(content=res.text, media_type="application/json")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Ollama: {str(e)}")
