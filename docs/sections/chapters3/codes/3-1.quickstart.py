print("####################      chap1 PromptTemplate     ####################")
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
print(
    prompt_template.format(adjective="funny", content="chickens")
)  # Tell me a funny joke about chickens.

print("####################      chap2 PromptTemplate     ####################")
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
print(prompt_template.format())  # Tell me a joke

print("####################      chap3 ChatPromptTemplate     ####################")
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)
messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
print(
    messages
)  # [SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name?')]

print("####################      chap4 ChatPromptTemplate     ####################")

from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to sound more upbeat."
            ),
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(
    messages
)  # [SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."), HumanMessage(content="I don't like eating tasty things")]

print("####################      chap5 LCEL     ####################")
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_val = prompt_template.invoke({"adjective": "funny", "content": "chickens"})
print(prompt_val)  # text='Tell me a funny joke about chickens.'

print("####################      chap6 LCEL     ####################")
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to sound more upbeat."
            ),
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
chat_val = chat_template.invoke({"text": "i dont like eating tasty things."})
print(
    chat_val
)  # messages=[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."), HumanMessage(content='i dont like eating tasty things.')]
