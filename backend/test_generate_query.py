import os
from agent.groq_client import GroqLLM
from agent.prompts import get_current_date, query_writer_instructions
from agent.tools_and_schemas import SearchQueryList
from agent.utils import get_research_topic

prompt = query_writer_instructions.format(
    current_date=get_current_date(),
    research_topic="find info about education and secular jobs of last five Popes",
    number_queries=3,
)
llm = GroqLLM(model=os.getenv('GROQ_MODEL','llama-3.3-70b-versatile'), api_key=os.getenv('GROQ_API_KEY'), temperature=0)
structured = llm.with_structured_output(SearchQueryList)
print('Invoking structured generate_query...')
res = structured.invoke(prompt)
print('Parsed result:', res)
print('Queries:', res.query)
