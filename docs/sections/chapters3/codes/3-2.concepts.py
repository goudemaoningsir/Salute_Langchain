print("####################      chap1 字符串提示组合     ####################")
from langchain.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
print(
    prompt.format(topic="sports", language="spanish")
)  # Tell me a joke about sports, make it funny and in spanish


print("####################      chap2 字符串提示组合     ####################")
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["language", "topic"],
    output_parser=None,
    partial_variables={},
    template="Tell me a joke about {topic}, make it funny\n\nand in {language}",
    template_format="f-string",
    validate_template=True,
)

print(
    prompt.format(topic="sports", language="spanish")
)  # Tell me a joke about sports, make it funny and in spanish

print("####################      chap3 字符串提示组合 - 使用     ####################")
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["language", "topic"],
    output_parser=None,
    partial_variables={},
    template="Tell me a joke about {topic}, make it funny\n\nand in {language}",
    template_format="f-string",
    validate_template=True,
)

model = ChatOpenAI(model="gpt-3.5-turbo")
chain = LLMChain(llm=model, prompt=prompt)
print(
    chain.run(topic="sports", language="spanish")
)  # ¿Por qué los futbolistas son tan buenos escalando montañas? Porque siempre anotan goles. ¡Jajaja!

print("####################      chap4 聊天提示组合     ####################")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

prompt = SystemMessage(content="You are a nice pirate")
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)
print(
    new_prompt.format_messages(input="i said hi")
)  # [SystemMessage(content='You are a nice pirate'), HumanMessage(content='hi'), AIMessage(content='what?'), HumanMessage(content='i said hi')]

print("####################      chap5 聊天提示组合 - 使用     ####################")
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

prompt = SystemMessage(content="You are a nice pirate")
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

model = ChatOpenAI(model="gpt-3.5-turbo")
chain = LLMChain(llm=model, prompt=new_prompt)
print(chain.run(input="i said hi"))  # Oh, hi there! How can I help you today?
