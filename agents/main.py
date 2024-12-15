from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.document_loaders import PyPDFLoader

from dataclasses import dataclass
import uuid, json

from models import model
from prompts import CHAT_PROMPT, CHAT_DOCUMENT_PROMPT

@dataclass
class Message:
    question: str
    answer: str
    id: str = str(uuid.uuid4())


class ChatService:

    def convert_list_to_json(data: list[object]) -> str:
        return json.dumps([i.__dict__ for i in data], indent=4)

    def convert_to_json(data: object) -> str:
        return json.dumps(data, indent=4)

    def convet_to_chat_history(messages: list[Message]) -> list[BaseMessage]:
        chat_history = []
        for message in messages:
            chat_history.append(HumanMessage(content=message.question))
            chat_history.append(AIMessage(content=message.answer))
        return chat_history

    def execute_chain(prompt:ChatPromptTemplate, input:dict) -> str:
        chain = prompt | model
        answer = chain.invoke(input).content
        print(answer)
        return answer
    
    def perform_chat(messages: list[Message], question: str) -> str:
        chat_history = ChatService.convet_to_chat_history(messages)
        return ChatService.execute_chain(CHAT_PROMPT, {"question": question, "chat_history": chat_history})
    
    def perform_chat_with_document(messages: list[Message], question: str, document: str) -> str:
        chat_history = ChatService.convet_to_chat_history(messages)
        return ChatService.execute_chain(CHAT_DOCUMENT_PROMPT, {"question": question, "chat_history": chat_history, "document": document})

messages:list[Message] = [
    Message(question="Hello", answer="Hello, how can I help you?"),
    Message(question="How to access salesforce tab ?", answer="To access a Salesforce tab, you need to log in to your Salesforce account and navigate to the tab section. The tabs are usually located at the top of the page, and you can click on the tab you want to access. If you cannot find the tab you are looking for, you can use the search bar to search for it. Once you have accessed the tab, you can view and interact with the data and information it contains."),
    Message(question="what is delinquent loans ?", answer="Delinquent loans are loans that are past due or overdue on their payments. In other words, a delinquent loan is a loan that has not been paid on time according to the terms of the loan agreement. The length of time a loan can be delinquent before it is considered in default varies depending on the type of loan and the lender's policies. Delinquent loans can have serious consequences for borrowers, including late fees, penalties, damage to credit scores, and even legal action by the lender to recover the debt."),
    Message(question="what is covid ?", answer="I'm sorry, but I cannot find any information related to COVID in the document you provided. However, COVID-19 is a highly infectious respiratory illness caused by the SARS-CoV-2 virus. It was first identified in Wuhan, China in December 2019 and has since spread globally, leading to a pandemic. The symptoms of COVID-19 can range from mild to severe and include fever, cough, and difficulty breathing. The virus is primarily spread through respiratory droplets when an infected person talks, coughs, or sneezes. To prevent the spread of COVID-19, it is recommended to practice social distancing, wear masks, wash hands frequently, and get vaccinated when possible.")
]

# # GA-1932
ChatService.perform_chat(messages, "tell me about mortgage refinancing")

# # GA-1932
ChatService.perform_chat(messages, "give me more details on this")
ChatService.perform_chat(messages, "summarize the document")


# document = PyPDFLoader(file_path="./escrowanalysis_onepage.pdf").load()
# ChatService.perform_chat_with_document(messages, "where is india ?", document)