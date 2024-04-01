print("############## chap 自定义角色的聊天消息 ##############")
from langchain.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"
chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
formatted_message = chat_message_prompt.format(subject="force")

print(formatted_message)

print("############## chap 控制格式化消息的 `MessagesPlaceholder` ##############")
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建一个聊天提示模板，其中包含动态对话内容的占位符
chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="dynamic_conversation")]
)
from langchain_core.messages import HumanMessage, AIMessage

# 模拟的对话内容
dynamic_conversation = [
    HumanMessage(content="你好！"),
    AIMessage(content="您好，我是聊天机器人。"),
    HumanMessage(content="聊天机器人能做什么？"),
    AIMessage(content="我可以帮您获取信息、解答问题等。"),
]
# 使用动态对话内容填充模板
filled_prompt = chat_prompt.format(dynamic_conversation=dynamic_conversation)
print(filled_prompt)

print("############## chap 动态格式化并插入对话 ##############")
from langchain_core.messages import AIMessage, HumanMessage

# 创建人工和 AI 消息
human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

# 格式化提示并插入对话内容
formatted_prompt = chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()

for message in formatted_prompt:
    print(message)
