from langchain_openai import ChatOpenAI, OpenAI

print("####################      step1 初始化模型     ####################")
chat_model = ChatOpenAI(model="gpt-3.5-turbo").bind(logprobs=True)

print("####################      step2 调用模型       ####################")
msg = chat_model.invoke(("human", "how are you today"))
print(msg)
print(msg.response_metadata["logprobs"]["content"][:5])


ct = 0
full = None
for chunk in chat_model.stream(("human", "how are you today")):
    if ct < 5:
        full = chunk if full is None else full + chunk
        if "logprobs" in full.response_metadata:
            print(full.response_metadata["logprobs"]["content"])
    else:
        break
    ct += 1
