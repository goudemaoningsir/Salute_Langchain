# !/usr/bin/env python
# -*- coding:utf-8 -*-

### [start]
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 配置国内跳转address
os.environ["OPENAI_API_BASE"] = "https://hk.xty.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-***"

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(
    chain.invoke({"topic": "ice cream"})
)  # Why did the ice cream truck break down? It had too many sundaes on the menu!
### [start]

### [prompt]

import os
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

prompt_value = prompt.invoke({"topic": "ice cream"})

print(
    prompt_value
)  # messages=[HumanMessage(content='tell me a short joke about ice cream')]

print(
    prompt_value.to_messages()
)  # [HumanMessage(content='tell me a short joke about ice cream')]

print(prompt_value.to_string())  # Human: tell me a short joke about ice cream

### [prompt]

### [model]
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# 配置国内跳转address
os.environ["OPENAI_API_BASE"] = "https://hk.xty.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-***"

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

prompt_value = prompt.invoke({"topic": "ice cream"})

model = ChatOpenAI(model="gpt-3.5-turbo")
message = model.invoke(prompt_value)
print(
    message
)  # content='Why did the ice cream truck break down?\nIt had too many sundaes on the menu!' response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 15, 'total_tokens': 35}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}
### [model]

### [out]
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 配置国内跳转address
os.environ["OPENAI_API_BASE"] = "https://hk.xty.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-PPbri9BfkuCZr6UH50097b26C52a4d63Be7c2c7aAe844e06"

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

prompt_value = prompt.invoke({"topic": "ice cream"})

model = ChatOpenAI(model="gpt-3.5-turbo")
message = model.invoke(prompt_value)
output_parser = StrOutputParser()
output = output_parser.invoke(message)
print(
    output
)  # Why did the ice cream truck break down? It had too many sundaes on the menu!
### [out]
