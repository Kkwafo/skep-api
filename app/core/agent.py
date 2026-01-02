import requests
import json
import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from app.core.memory import doc_context

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("AI_API_KEY")
API_URL = os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions")
MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")

# Local fallback config
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2") 

SYSTEM_PROMPT_TEMPLATE = """
Eres el Asistente Académico de SKEP, una plataforma educativa.
Tu personalidad es profesional, directiva pero amable, similar a un director de institución educativa que guía a los profesores.
Evita el lenguaje coloquial excesivo. Sé claro, conciso y autoritario de manera constructiva.

CONTEXTO ACTUAL:
Ruta: {current_path}
Texto del Documento Cargado: {document_snippet}

INSTRUCCIONES:
1. Responde preguntas sobre el documento si hay contenido cargado.
2. Si el usuario te pide una guía de la pantalla actual, usa el contexto de ruta.
3. Si generas un resumen o texto para el editor, usa el siguiente formato OBLIGATORIO:
   <<<ACTION_INSERT: El contenido exacto aquí >>>
"""

async def run_agent(message: str, current_path: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
    # 1. Prepare Document Context
    doc_text = doc_context.get_text()
    if doc_text:
        # Context limit (adjust based on provider limits)
        doc_snippet = doc_text[:2000] + "..." if len(doc_text) > 2000 else doc_text
    else:
        doc_snippet = "No hay documento cargado."
        
    # 2. Build System Prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        current_path=current_path,
        document_snippet=doc_snippet
    )

    # 3. Build Messages
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        for human, ai in chat_history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
            
    messages.append({"role": "user", "content": message})

    # 4. Determine Client (Remote vs Local)
    target_url = OLLAMA_URL if USE_LOCAL_OLLAMA else API_URL
    target_model = OLLAMA_MODEL if USE_LOCAL_OLLAMA else MODEL
    
    try:
        headers = {}
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False
        }

        if USE_LOCAL_OLLAMA:
             payload["options"] = {
                "temperature": 0.7,
                "num_predict": 512,
                "num_ctx": 2048
             }
        else:
             # Remote API headers
             headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
             }
             payload["temperature"] = 0.7

        # Call API
        response = requests.post(target_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
             # If using local ollama and 404, give specific advice
             if USE_LOCAL_OLLAMA and response.status_code == 404:
                 return "No encuentro el servicio de IA local. Por favor, verifica que el modelo requerido esté disponible."
             
             print(f"API Error: {response.status_code} - {response.text}")
             return f"Error en el servicio de IA ({response.status_code})."

        result = response.json()
        
        # Handle Output Parsing based on structure
        # OLLAMA response structure: { "message": { "content": "..." } }
        # OPENAI response structure: { "choices": [ { "message": { "content": "..." } } ] }
        
        content = ""
        if "message" in result and "content" in result["message"]:
             content = result["message"]["content"]
        elif "choices" in result and len(result["choices"]) > 0:
             content = result["choices"][0].get("message", {}).get("content", "")
        else:
             content = "Respuesta vacía del servicio de IA."
             
        return content
        
    except Exception as e:
        print(f"Agent Connection Error: {e}")
        return "Error de conexión con el servicio de IA. Por favor, verifica tu configuración y conexión."
