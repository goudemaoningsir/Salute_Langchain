print("############## chap 基于n-gram重叠选择 ##############")
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 假设任务是创建反义词的示例。
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 可供选择的示例列表。
    examples,
    # 用于生成嵌入（用于测量语义相似度）的嵌入类。
    OpenAIEmbeddings(),
    # 用于存储嵌入并进行相似性搜索的VectorStore类。
    Chroma,
    # 要生成的示例数量。
    k=1,
)
similar_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
print(similar_prompt.format(adjective="worried"))
print(similar_prompt.format(adjective="large"))

print("############## chap 添加示例 ##############")

similar_prompt.example_selector.add_example(
    {"input": "enthusiastic", "output": "apathetic"}
)
print(similar_prompt.format(adjective="passionate"))
