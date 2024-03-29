print("############## chap 基于最大边际相关性（MMR）选择 ##############")
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_community.vectorstores import FAISS
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

print("############## chap2 MaxMarginalRelevanceExampleSelector ##############")

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 可供选择的示例列表。
    examples,
    # 用于生成嵌入（用于测量语义相似度）的嵌入类。
    OpenAIEmbeddings(),
    # 用于存储嵌入并进行相似性搜索的VectorStore类。
    FAISS,
    # 要生成的示例数量。
    k=2,
)

mmr_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# 输入是一种感觉，所以应该选择happy/sad示例作为第一个。
print(mmr_prompt.format(adjective="worried"))

print("############## chap3 SemanticSimilarityExampleSelector ##############")
# 让我们比较一下，如果我们只根据相似度选择，使用SemanticSimilarityExampleSelector代替MaxMarginalRelevanceExampleSelector会得到什么。
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 可供选择的示例列表。
    examples,
    # 用于生成嵌入（用于测量语义相似度）的嵌入类。
    OpenAIEmbeddings(),
    # 用于存储嵌入并进行相似性搜索的VectorStore类。
    FAISS,
    # 要生成的示例数量。
    k=2,
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
