import json
from types import SimpleNamespace
from typing import Any, Optional, Type

import requests
from pydantic import BaseModel


class GroqLLM:
    """Minimal Groq model wrapper used by the agent.

    This wrapper provides a very small surface compatible with how the
    repository uses an LLM: `invoke(prompt)` and `with_structured_output(schema)`.
    It calls the Groq HTTP API and attempts to return a simple object with a
    `content` attribute. For structured output it parses JSON into the provided
    Pydantic model.
    """

    def __init__(self, model: str, temperature: float = 0.0, api_key: Optional[str] = None, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._structured_schema: Optional[Type[BaseModel]] = None

    def with_structured_output(self, schema: Type[BaseModel]):
        """Return a copy configured to parse structured JSON output into `schema`."""
        clone = GroqLLM(self.model, self.temperature, self.api_key, self.max_tokens)
        clone._structured_schema = schema
        return clone

    def _call_api(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GROQ API key is not set for GroqLLM")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prefer Groq's OpenAI-compatible Responses API which the docs recommend
        # Example: POST https://api.groq.com/openai/v1/responses { model, input }
        last_exc = None
        data = None

        try:
            oa_url = "https://api.groq.com/openai/v1/responses"
            payload = {
                "model": self.model,
                "input": prompt,
                "temperature": float(self.temperature),
                "max_output_tokens": int(self.max_tokens),
            }
            resp = requests.post(oa_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            last_exc = exc

        # Fallback: try legacy/non-openai shapes across several hosts/endpoints
        if data is None:
            bases = [
                "https://api.groq.com/v1",
                "https://api.groq.ai/v1",
                "https://gateway.groq.ai/v1",
            ]
            endpoints = ["outputs", "completions", "generate", "responses", "predict"]

            for base in bases:
                for ep in endpoints:
                    url = f"{base}/models/{self.model}/{ep}"
                    payload = {
                        "input": prompt,
                        "temperature": float(self.temperature),
                        "max_output_tokens": int(self.max_tokens),
                    }
                    try:
                        resp = requests.post(url, headers=headers, json=payload, timeout=60)
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except requests.exceptions.HTTPError as he:
                        last_exc = he
                        try:
                            status = resp.status_code
                        except Exception:
                            status = None
                        if status == 404:
                            continue
                        raise
                    except Exception as exc:
                        last_exc = exc
                        continue
                if data is not None:
                    break

        if data is None:
            raise last_exc if last_exc is not None else RuntimeError("Failed to call Groq API")

        # Try to extract textual output from the OpenAI-compatible Responses API
        # Groq examples expose `output_text` or nested `output` structures.
        text = None
        if isinstance(data, dict):
            # Response.create style: response.output_text
            if "output_text" in data and isinstance(data["output_text"], str):
                text = data["output_text"]

            # Newer Responses API: output -> list of content blocks
            if text is None and "output" in data:
                out = data["output"]
                # output could be a list of blocks or a dict
                if isinstance(out, list) and len(out) > 0:
                    # try common nested paths
                    parts = []
                    for item in out:
                        if isinstance(item, dict):
                            # common: item.content -> list of {'type':'output_text','text':...}
                            content = item.get("content") or item.get("data")
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and "text" in c:
                                        parts.append(c["text"])
                                    elif isinstance(c, str):
                                        parts.append(c)
                            elif isinstance(content, str):
                                parts.append(content)
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        text = "".join(parts)

            # OpenAI-style choices
            if text is None and "choices" in data:
                try:
                    ch = data["choices"][0]
                    if isinstance(ch, dict) and "text" in ch:
                        text = ch["text"]
                    elif isinstance(ch, dict) and "message" in ch:
                        # chat-style
                        msg = ch["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            # content may be a list or string
                            content = msg["content"]
                            if isinstance(content, str):
                                text = content
                            elif isinstance(content, list) and len(content) > 0:
                                # pick first text-like element
                                for el in content:
                                    if isinstance(el, dict) and "text" in el:
                                        text = el["text"]
                                        break
                except Exception:
                    pass

            # top-level heuristics
            if text is None:
                if "text" in data and isinstance(data["text"], str):
                    text = data["text"]
                elif "output" in data and isinstance(data["output"], str):
                    text = data["output"]

        # As ultimate fallback, pretty-print the JSON
        if text is None:
            text = json.dumps(data)

        return text

    def invoke(self, prompt: str) -> Any:
        """Invoke the model and return an object with a `.content` attribute.

        If `with_structured_output` was used, the JSON output will be parsed
        by the provided Pydantic `schema` and the model instance returned.
        """
        raw = self._call_api(prompt)

        if self._structured_schema is not None:
            # try to extract JSON object from raw text
            try:
                # If the model returns surrounding text, try to locate first '{'..'}'
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_text = raw[start : end + 1]
                else:
                    json_text = raw
                # Attempt to make common model-produced JSON valid:
                # - Strip markdown fences
                if json_text.strip().startswith("```"):
                    # remove fencing like ```json ... ```
                    parts = json_text.split("```")
                    # take the part that contains a brace
                    for p in parts:
                        if "{" in p and "}" in p:
                            json_text = p
                            break

                # Remove trailing commas before closing braces/brackets
                import re

                json_text = re.sub(r",\s*([}\]])", r"\1", json_text)

                parsed = self._structured_schema.parse_raw(json_text)
                return parsed
            except Exception as exc:
                raise ValueError(f"Failed to parse structured output: {exc}\nRaw output:\n{raw}")

        # Return a simple namespace with content attribute for compatibility
        return SimpleNamespace(content=raw)
