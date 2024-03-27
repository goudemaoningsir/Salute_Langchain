# 配置国内跳转address
import os

os.environ["OPENAI_API_BASE"] = "https://hk.xty.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-****"


print("===============llm=======================")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
print(llm.invoke("LangSmith能如何帮助测试？"))
"""
content='LangSmith可以帮助测试人员进行自动化测试，通过编写测试脚本和执行测试用例来快速准确地测试软件应用程序。LangSmith还可以帮助测试人员进行性能测试和负载测试，以确保软件应用程序在不同条件下的稳定性和可靠性。此外，LangSmith还可以帮助测试人员进行代码覆盖率分析和静态代码分析，以帮助发现潜在的问题和漏洞。LangSmith还可以帮助测试团队管理测试用例和缺陷跟踪，以便更好地组织和管理测试工作。' response_metadata={'token_usage': {'completion_tokens': 181, 'prompt_tokens': 18, 'total_tokens': 199}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}
"""
print("===============llm=======================")

print("=================prompt=====================")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名世界级的技术文档编写者。"),
        ("user", "{input}"),
    ]
)
chain = prompt | llm
print(chain.invoke("LangSmith能如何帮助测试？"))
"""
content='LangSmith是一种自然语言处理工具，它可以帮助测试团队在测试过程中进行文本分析、自然语言处理和语言理解。以下是LangSmith可以帮助测试团队的一些方面：\n\n1. 自然语言测试：LangSmith可以用于分析和理解应用程序的用户界面文本、日志文件和其他文本数据。测试团队可以利用LangSmith来检测语法错误、语义错误和其他文本相关的问题。\n\n2. 自动化测试：LangSmith可以集成到自动化测试框架中，帮助测试团队编写更智能的测试脚本。通过使用LangSmith，测试团队可以更轻松地处理动态文本内容、多语言支持和其他文本相关的测试任务。\n\n3. 情感分析：LangSmith可以帮助测试团队进行情感分析，评估应用程序的用户体验和情感反馈。测试团队可以利用LangSmith来检测用户评论、社交媒体反馈以及其他文本数据中的情感倾向。\n\n4. 文本生成：LangSmith可以用于生成测试数据、创建自然语言测试用例和模拟用户输入。测试团队可以利用LangSmith生成各种文本数据，以验证应用程序的各种功能和边界条件。\n\n总的来说，LangSmith可以帮助测试团队提高测试效率、准确性和覆盖范围，从而更好地发现和解决应用程序中的问题。' response_metadata={'token_usage': {'completion_tokens': 410, 'prompt_tokens': 41, 'total_tokens': 451}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}
"""
print("=================prompt=====================")

print("=================output=====================")
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
print(chain.invoke("LangSmith能如何帮助测试？"))
"""
LangSmith是一种自然语言处理工具，它可以帮助测试团队在测试过程中进行文本分析、自然语言处理和语言理解。以下是LangSmith可以帮助测试团队的一些方面：

1. 自然语言测试：LangSmith可以用于分析和理解应用程序的用户界面文本、日志文件和其他文本数据。测试团队可以利用LangSmith来检测语法错误、语义错误和其他文本相关的问题。

2. 自动化测试：LangSmith可以集成到自动化测试框架中，帮助测试团队编写更智能的测试脚本。通过使用LangSmith，测试团队可以更轻松地处理动态文本内容、多语言支持和其他文本相关的测试任务。

3. 情感分析：LangSmith可以帮助测试团队进行情感分析，评估应用程序的用户体验和情感反馈。测试团队可以利用LangSmith来检测用户评论、社交媒体反馈以及其他文本数据中的情感倾向。

4. 文本生成：LangSmith可以用于生成测试数据、创建自然语言测试用例和模拟用户输入。测试团队可以利用LangSmith生成各种文本数据，以验证应用程序的各种功能和边界条件。

总的来说，LangSmith可以帮助测试团队提高测试效率、准确性和覆盖范围，从而更好地发现和解决应用程序中的问题。
"""
print("=================output=====================")
