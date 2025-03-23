from fastapi import FastAPI, Response, HTTPException
import requests
import os

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello World"}


@app.get("/ask")
def ask(prompt: str):
    try:
        res = requests.post(
            "http://llama3.2-webui:11434/api/generate",
            json={
                "prompt": prompt,
                "stream": False,
                "model": "llama3.2:latest"
            },
            timeout=30  # Add timeout
        )
        res.raise_for_status()  # Raise exception for bad status codes
        return Response(content=res.text, media_type="application/json")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Ollama: {str(e)}")
