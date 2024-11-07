from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import subprocess

app = FastAPI() # Create FastAPI instance

class Message(BaseModel):
    role: str
    content: str

class RequestData(BaseModel):
    messages: List[Message]
    model: str
    stream: bool

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/run_tiny_llama_q/chat/completions", description="Run Tiny Llama Quantized")
async def run_tiny_llama_q(request: RequestData):
    try:
        print(request)
        print(request.messages[0].content, request.messages[1].content) # Remove

        # Run ollama model with input prompt
        model_response = subprocess.run(
            ["wsl", "ollama", "run", "tinyllama:1.1b-chat-v1-q2_K", request.messages[1].content],
            capture_output=True, text=True
        ) # TODO: sent request must be a sum of system prompt [0] and user prompt [1]

        # Check for errors
        if model_response.returncode != 0:
            raise HTTPException(status_code=500, detail="Error running Ollama model.")
        
        # Return model response
        print(model_response.stdout.strip())
        return {"message": model_response.stdout.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_tiny_llama", description="Run Tiny Llama Chat")
async def run_tiny_llama_chat(request: RequestData):
    try:
        print(request)
        print(request.messages[0].content, request.messages[1].content) # Remove
        
        # Run ollama model with input prompt
        model_response = subprocess.run(
            ["wsl", "ollama", "run", "tinyllama:chat", request.messages[1].content],
            capture_output=True, text=True
        )

        # Check for errors
        if model_response.returncode != 0:
            raise HTTPException(status_code=500, detail="Error running Ollama model.")
        
        # Return model response
        return {"response": model_response.stdout.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
