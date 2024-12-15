from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent, AgentExecutor


# Define a simple tool
@tool
def add(x: float, y: float) -> float:
    """
    use this tool to add two numbers
    example:
       input: add 2 and 3
       output: 5
    Args:
        x (float): first number
        y (float): second number
    
    """
    return x + y

tools = [add]

# Create an LLM
llm = AzureChatOpenAI(
    azure_endpoint="https://openaixpanseaisandbox.openai.azure.com",
    openai_api_key="0172189207fb4f08bd7fee99906b7b51",
    openai_api_version="2023-05-15",
    azure_deployment="gpt-35-turbo",
    temperature=0,
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agents
tool_calling_agent = create_tool_calling_agent(llm, tools, prompt)
openai_tools_agent = create_openai_tools_agent(llm, tools, prompt)

# Execute the agent
execute_agent = lambda agent, input: AgentExecutor(agent=agent, tools=tools, verbose=False).invoke({"input": input})

# Run the agent
result = execute_agent(tool_calling_agent, "add 4 and 2")
print(result)