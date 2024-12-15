from langchain_openai import AzureChatOpenAI
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field
from langchain.agents import create_openai_tools_agent, create_react_agent, AgentExecutor
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
import requests, json

# MODEL
model = AzureChatOpenAI(
    azure_endpoint="https://openaixpanseaisandbox.openai.azure.com",
    openai_api_key="0172189207fb4f08bd7fee99906b7b51",
    openai_api_version="2023-05-15",
    azure_deployment="gpt-35-turbo",
    temperature=0,
)

# TOOL
@tool(return_direct=True)
def get_post_data(id:int) -> dict:
    """Use the jsonplaceholder API to get post data for the given id"""
    """params: id:int"""
    response = requests.get(f"https://jsonplaceholder.typicode.com/posts/{id}")
    return response.json()

# TOOLS
tools = [get_post_data]

agent_template = """
Answer the following questions as best you can.
You have access to the following tools:{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: pass input to action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {question}
Thought:{agent_scratchpad}"""

#AGENT
agent = create_react_agent(llm=model, tools=tools, prompt=PromptTemplate.from_template(agent_template))

# EXECUTOR
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# EXECUTE
response = executor.invoke(input={"question": "What is the title of the post with id 1?"})

print(response.get("output"))
