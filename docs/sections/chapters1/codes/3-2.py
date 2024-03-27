# 配置国内跳转address
import os

os.environ["OPENAI_API_BASE"] = "https://hk.xty.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-PPbri9BfkuCZr6UH50097b26C52a4d63Be7c2c7aAe844e06"
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

print("================== step1 准备数据 ==================")
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
print("================== step1 准备数据 ==================")

print("================== step2 构建索引 ==================")
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
print("================== step2 构建索引 ==================")

print("================== step3 设置检索链 ==================")
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """根据提供的上下文回答以下问题：
<context>{context}</context>问题：{input}"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
print("================== step3 设置检索链 ==================")

print("================== step4 运行检索链 ==================")
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "LangSmith能如何帮助测试？"})
print(response["answer"])
print("================== step4 运行检索链 ==================")
