from langchain_openai import OpenAI

print("####################      step1 初始化模型     ####################")
llm = OpenAI()

print(
    llm.invoke(
        "What are some theories about the relationship between unemployment and inflation?"
    )
)

for chunk in llm.stream(
    "What are some theories about the relationship between unemployment and inflation?"
):
    print(chunk, end="", flush=True)
