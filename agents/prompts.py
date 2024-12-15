from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

system_message = SystemMessage(content="You are Halo, a helpful AI assistant.")

CHAT_PROMPT = ChatPromptTemplate.from_messages(messages=[
    system_message,
    MessagesPlaceholder('chat_history'),
    SystemMessage(content="chat_history is not a document"),
    ("user", "{question}"),
])


CHAT_DOCUMENT_PROMPT = ChatPromptTemplate.from_messages(messages=[
    system_message,
    MessagesPlaceholder('chat_history'),
    SystemMessage(content="chat_history is not a document"),
    ("system", "{document}"),
    SystemMessage(content="use the document and answer the question"),
    ("user", "{question}"),
])