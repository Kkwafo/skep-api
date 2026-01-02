import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import UserMessage, AgentResponse
from app.core.agent import run_agent
from app.core.memory import doc_context
from pypdf import PdfReader

app = FastAPI(title="SKEP AI Agent API")

# Configure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTION_PATTERN = re.compile(r"<<<ACTION_INSERT: (.*?) >>>", re.DOTALL)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Reads a PDF file and stores its text content in volatile memory.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        doc_context.save_text(text)
        return {"status": "ok", "message": "Documento leído. Bubblebee ya puede consultarlo."}
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")

@app.post("/api/chat", response_model=AgentResponse)
async def chat_endpoint(request: UserMessage):
    """
    Intelligent Chat Endpoint powered by Bubblebee Agent.
    Intersects responses to parse Action Protocol tags.
    """
    try:
        raw_response = await run_agent(
            message=request.message,
            current_path=request.current_path,
            chat_history=request.chat_history
        )
        
        # Parse for Action Protocol
        match = ACTION_PATTERN.search(raw_response)
        if match:
            # Action detected
            content_to_insert = match.group(1).strip()
            # We want to return a user-friendly message saying we did it, 
            # and pass the payload separately.
            # Clean the tag from the 'content' field to keep the chat clean.
            clean_response = ACTION_PATTERN.sub("Aquí tienes, lo he enviado al editor.", raw_response)
            
            return AgentResponse(
                type="action_insert",
                content=clean_response,
                action_payload=content_to_insert
            )
        else:
            # Normal text response
            return AgentResponse(content=raw_response)
            
    except Exception as e:
        print(f"Agent Error: {e}")
        return AgentResponse(content="Bzz! Algo salió mal en mi colmena. Inténtalo de nuevo. 🐝")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
