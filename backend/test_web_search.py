import os
from agent.groq_client import GroqLLM
from agent.prompts import web_searcher_instructions, get_current_date
from agent.tools_and_schemas import WebSearchResult

prompt = web_searcher_instructions.format(current_date=get_current_date(), research_topic="test")
llm = GroqLLM(model=os.getenv('GROQ_MODEL','llama-3.3-70b-versatile'), api_key=os.getenv('GROQ_API_KEY'), temperature=0)
structured = llm.with_structured_output(WebSearchResult)
print('Invoking structured web_search...')
res = structured.invoke(prompt + '\n\nRespond ONLY with a JSON object with keys: summary and sources (list of {title,url,snippet}).')
print('Parsed web search result summary:', res.summary)
print('Number of sources:', len(res.sources))
for s in res.sources[:5]:
    print('-', s.title, s.url)
