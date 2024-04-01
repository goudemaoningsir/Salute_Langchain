print("############## chap 少样本提示模板 ##############")
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 这是一个用于格式化每个单独示例的提示模板。
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
print(few_shot_prompt.format())
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-3.5-turbo")

chain = final_prompt | chat_model

print(chain.invoke({"input": "What's the square of a triangle?"}))

print("############## chap 动态少样本提示 ##############")
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)
# 提示模板将通过传递输入到`select_examples`方法来加载示例
example_selector.select_examples({"input": "horse"})
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 定义少样本提示。
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # 输入变量选择传递给`example_selector`的值。
    input_variables=["input"],
    example_selector=example_selector,
    # 定义每个示例如何格式化。
    # 在这种情况下，每个示例将成为2条消息：
    # 1个人类，和1个AI
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    ),
)
print(few_shot_prompt.format(input="What's 3+3?"))
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
print(few_shot_prompt.format(input="What's 3+3?"))

from langchain_community.chat_models import ChatAnthropic

chain = final_prompt | ChatAnthropic(temperature=0.0)
chain.invoke({"input": "What's 3+3?"})
