import os
from agent.groq_client import GroqLLM

print('GROQ_API_KEY present:', 'GROQ_API_KEY' in os.environ)
llm = GroqLLM(model=os.getenv('GROQ_MODEL','llama-3.3-70b-versatile'), api_key=os.getenv('GROQ_API_KEY'), temperature=0)
res = llm.invoke('Say hello and identify yourself in one sentence.')
print('Result type:', type(res))
try:
    print('Result content:', res.content)
except Exception:
    print('Result raw:', res)
