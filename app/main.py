import re
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import UserMessage, AgentResponse
from app.core.agent import run_agent
from app.core.memory import doc_context
import PyPDF2

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
        # PyPDF2 logic
        pdf_reader = PyPDF2.PdfFileReader(file.file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText() + "\n"
            
        doc_context.save_text(text)
        return {"status": "ok", "message": "Documento leído y memorizado para esta sesión."}
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")

@app.post("/api/chat", response_model=AgentResponse)
async def chat_endpoint(request: UserMessage):
    """
    Intelligent Chat Endpoint.
    """
    try:
        # Pydantic v1 models are dictionaries when passed, but request is a Model object here
        # request.message is correct
        
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
            clean_response = ACTION_PATTERN.sub("¡Entendido! He preparado el contenido para el editor.", raw_response)
            
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

# For running via `python app/main.py` if needed, 
# though standard is `uvicorn app.main:app` or `python -m app.main`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
