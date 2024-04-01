from langchain_openai import ChatOpenAI, OpenAI

print("####################      step1 初始化模型     ####################")
chat = ChatOpenAI(model="gpt-3.5-turbo")

print("####################      step2 调用模型       ####################")
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]
print("####################       invoke       ####################")

print(chat.invoke(messages))

print("####################       stream       ####################")

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

print("####################       batch       ####################")

print(chat.batch([messages]))
