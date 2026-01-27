import json
from types import SimpleNamespace
from typing import Any, Optional, Type

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class GroqLLM:
    """Minimal Groq model wrapper used by the agent.

    This wrapper provides a very small surface compatible with how the
    repository uses an LLM: `invoke(prompt)` and `with_structured_output(schema)`.
    It uses LangChain's ChatGroq model for API calls.
    """

    def __init__(self, model: str, temperature: float = 0.0, api_key: Optional[str] = None, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._structured_schema: Optional[Type[BaseModel]] = None
        
        # Initialize the ChatGroq model
        self._chat_model = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    def with_structured_output(self, schema: Type[BaseModel]):
        """Return a copy configured to parse structured JSON output into `schema`."""
        clone = GroqLLM(self.model, self.temperature, self.api_key, self.max_tokens)
        clone._structured_schema = schema
        clone._chat_model = self._chat_model.with_structured_output(schema)
        return clone

    def invoke(self, prompt: str) -> Any:
        """Invoke the model and return an object with a `.content` attribute.

        If `with_structured_output` was used, the structured output will be
        parsed and the model instance returned.
        """
        # Create a message for the chat model
        message = HumanMessage(content=prompt)
        
        # Invoke the model
        result = self._chat_model.invoke([message])
        
        if self._structured_schema is not None:
            # When using structured output, the result is already parsed into the schema
            return result
        
        # For unstructured output, return as simple namespace with content attribute
        if hasattr(result, 'content'):
            return SimpleNamespace(content=result.content)
        else:
            return SimpleNamespace(content=str(result))
