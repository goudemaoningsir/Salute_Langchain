from langchain_openai import ChatOpenAI, OpenAI

print("####################      step1 初始化模型     ####################")
llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

print("####################      step2 调用模型       ####################")
from langchain_core.messages import HumanMessage

text = "为一家生产彩色袜子的公司起一个好名字是什么？"
messages = [HumanMessage(content=text)]
print("messages:", messages)
print("llm.invoke(text):", llm.invoke(text))  # Rainbow Threads
print(
    "chat_model.invoke(messages):", chat_model.invoke(messages)
)  # content='“彩虹袜子公司”' response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 30, 'total_tokens': 40}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}

print("####################      step3 使用提示模板     ####################")
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("一家生产{product}的公司的好名字是什么？")

print(prompt.format(product="彩色袜子"))  # 一家生产彩色袜子的公司的好名字是什么？

print("####################  step4 ChatPromptTemplate  ####################")

from langchain.prompts.chat import ChatPromptTemplate

template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", human_template),
    ]
)
out = chat_prompt.format_messages(
    input_language="English", output_language="French", text="I love programming."
)
print(
    out
)  # [SystemMessage(content='You are a helpful assistant that translates English to French.'), HumanMessage(content='I love programming.')]

print("####################      step5 输出解析       ####################")

from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
out = output_parser.parse("你好, 再见")
print(out)  # 返回 ['你好', '再见']


print("####################      step6 组合链        ####################")

template = "Generate a list of 5 {text}.\n\n{format_instructions}"
chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)
chain = chat_prompt | chat_model | output_parser
out = chain.invoke({"text": "colors"})
print(out)  # ['red', 'blue', 'green', 'yellow', 'orange']
