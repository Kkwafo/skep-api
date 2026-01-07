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

# --- SYSTEM PROMPT ---
"""
SYSTEM PROMPT TEMPLATE V5 (Surgical/Strict)

Rationale:
- We define TWO distinct structural Options (A: Unit, B: Class).
- We explicitly disable "Fundamentación" and "Bibliografía" generation for Option B to prevent overriding existing sections.
- We enforce strict HTML Tables for "Desarrollo" and "Contenidos" to match the frontend Tiptap nodes exactly.
"""
SYSTEM_PROMPT_TEMPLATE = """
Eres el Asistente Académico de SKEP.
Personalidad: Director de institución educativa. Profesional, directivo, amable y riguroso con la planificación.

CONTEXTO ACTUAL:
Ruta: {current_path}
Documento cargado (fragmento): {document_snippet}
Contenido actual del editor: {editor_context}

INSTRUCCIONES OPERATIVAS:
1. Responde preguntas basándote en el documento cargado.
2. Si te piden generar contenido para el editor (una planificación), DEBES usar el formato Markdown estricto dentro de marcadores <<< ... >>>.
3. Detecta qué tipo de planificación pide el usuario y usa la estructura exacta correspondiente:

---
OPCIÓN A: ESTRUCTURA "UNIDAD DIDÁCTICA" (Usa estos encabezados)

## FUNDAMENTACIÓN Y PROPÓSITOS
### Fundamentación
(Texto justificativo...)
### Diagnóstico
(Resultados diagnóstico inicial...)
### Objetivos Generales
- [Objetivo 1]
- [Objetivo 2]

### PRESENTACIÓN / DESCRIPCIÓN
**Eje/Unidad:** [Nombre] | **Título:** [Tema] | **Tiempo:** [Estimado]

DEBES usar esta estructura de TABLA HTML exacta para los contenidos (No uses listas sueltas):
<table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">
  <thead style="background-color: #f1f5f9;">
    <tr>
      <th style="border: 1px solid #bbb; padding: 8px; width: 30%;">Objetivos Específicos</th>
      <th style="border: 1px solid #bbb; padding: 8px; width: 40%;">Contenidos</th>
      <th style="border: 1px solid #bbb; padding: 8px; width: 30%;">Estrategias / Recursos / Evaluación</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #bbb; padding: 8px; vertical-align: top;">
        <ul>
          <li>[Objetivo...]</li>
          <li>[Interpretar...]</li>
        </ul>
      </td>
      <td style="border: 1px solid #bbb; padding: 8px; vertical-align: top;">
        <p><strong>Conceptuales:</strong></p>
        <ul>
          <li>[Contenido A...]</li>
          <li>[Contenido B...]</li>
        </ul>
        <p><strong>Procedimentales:</strong></p>
        <ul>
           <li>[Análisis de...]</li>
        </ul>
      </td>
      <td style="border: 1px solid #bbb; padding: 8px; vertical-align: top;">
        <p><strong>Estrategias:</strong></p>
        <ul><li>[Estrategias...]</li></ul>
        <p><strong>Recursos:</strong></p>
        <ul><li>[Recursos...]</li></ul>
        <p><strong>Evaluación:</strong></p>
        <ul><li>[Criterios...]</li></ul>
      </td>
    </tr>
  </tbody>
</table>

### CAPACIDADES FUNDAMENTALES
- Oralidad, Lectura y Escritura: ...
- Pensamiento Crítico: ...
- Trabajo con otros: ...
- Resolución de Problemas: ...
- Transversales: ...

### BIBLIOGRAFÍA
- Del Docente: ...
- Del Estudiante: ...

---
OPCIÓN B: ESTRUCTURA "PLAN DE CLASE" (Usa estos encabezados)

### PLAN DE CLASE
**Tema/Eje:** [Tema del día]
**Tiempo:** 80 min

### Apertura (Actualización de Diagnóstico)
(Actividad de inicio...)

### Desarrollo
DEBES usar esta estructura de TABLA HTML exacta (mismos estilos y etiquetas):
<table style="width: 100%; border: 1px solid #ddd;">
   <thead>
       <tr style="background-color: #f1f5f9;">
           <th style="padding: 5px;">Momento 1 (Inicio Desarrollo)</th>
           <th style="padding: 5px;">Momento 2 (Profundización)</th>
           <th style="padding: 5px;">Momento 3 (Consolidación)</th>
       </tr>
   </thead>
   <tbody>
       <tr>
           <td style="padding: 8px; vertical-align: top;"><ul><li>[Actividad de Inicio...]</li></ul></td>
           <td style="padding: 8px; vertical-align: top;"><ul><li>[Actividad de Profundización...]</li></ul></td>
           <td style="padding: 8px; vertical-align: top;"><ul><li>[Actividad de Consolidación...]</li></ul></td>
       </tr>
   </tbody>
</table>

### Cierre
(Actividad de síntesis...)

DEBES COMPLETAR SOLAMENTE LA SECCIÓN "PLAN DE CLASE". NO GENERES FUNDAMENTACIÓN, OBJETIVOS, NI BIBLIOGRAFÍA. SOLO EL PLAN DIARIO Y LA TABLA.

---
FIN DE INSTRUCCIONES
"""

async def run_agent(message: str, current_path: str, chat_history: Optional[List[Tuple[str, str]]] = None, editor_context: Optional[str] = None) -> str:
    """
    Executes a turn of the AI Agent conversation using the configured LLM provider (Remote vs Local).

    Args:
        message (str): The user's query or prompt.
        current_path (str): The filesystem path context (informative).
        chat_history (List): Recent conversation turns to maintain context.
        editor_context (str): The current text content of the editor (for context-aware answers).

    Returns:
        str: The generated text response from the AI (often containing Markdown/HTML).
    """
    # 1. Prepare Document Context
    doc_text = doc_context.get_text()
    if doc_text:
        # Strict context limit for token saving
        limit = 10000 
        doc_snippet = doc_text[:limit] + "..." if len(doc_text) > limit else doc_text
    else:
        doc_snippet = "N/A"
        
    # 2. Build System Prompt (Using replace strictly to avoid f-string errors with document content)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{current_path}", current_path) \
                                          .replace("{editor_context}", editor_context if editor_context else "No disponible") \
                                          .replace("{document_snippet}", doc_snippet)

    # 3. Build Messages
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        # Keep only the last 2 interactions to save tokens
        recent_history = chat_history[-2:]
        for human, ai in recent_history:
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
                "num_predict": 1024, # Increased to allow for long planning responses
                "num_ctx": 4096
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
             if USE_LOCAL_OLLAMA and response.status_code == 404:
                 return "No encuentro el servicio de IA local. Por favor, verifica que Ollama esté corriendo."
             
             print(f"API Error: {response.status_code} - {response.text}")
             return f"Error en el servicio de IA ({response.status_code})."

        result = response.json()
        
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
        return "Error de conexión con el servicio de IA. Por favor, verifica tu configuración."