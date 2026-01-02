from pydantic import BaseModel
from typing import List, Tuple, Optional

class UserMessage(BaseModel):
    message: str
    current_path: str
    chat_history: Optional[List[Tuple[str, str]]] = None # List of (human_message, ai_message) tuples

class AgentResponse(BaseModel):
    type: str = "text"
    content: str
    action_payload: Optional[str] = None
