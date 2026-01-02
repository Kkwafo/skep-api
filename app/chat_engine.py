from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize LLM - using llama3.2 as primary preference
llm = ChatOllama(model="llama3.2")

def generate_reply(message: str, history: list[str] | None = None) -> str:
    """
    Generates a reply using Ollama, maintaining simple context from history.
    """
    # System Instruction
    system_msg = SystemMessage(content="You are a helpful and knowledgeable AI assistant for the SKEP platform. Answer concisely and accurately.")
    messages = [system_msg]
    
    # Add history as context if present
    if history:
        # Concatenating history into a context block
        history_text = "\n".join(history)
        messages.append(SystemMessage(content=f"Previous conversation context:\n{history_text}"))
    
    # Add current user message
    messages.append(HumanMessage(content=message))
    
    # Invoke LLM
    response = llm.invoke(messages)
    return response.content
