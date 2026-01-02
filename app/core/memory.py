class DocumentContext:
    _instance = None
    _text = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentContext, cls).__new__(cls)
        return cls._instance

    def save_text(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text

# Global instance
doc_context = DocumentContext()
