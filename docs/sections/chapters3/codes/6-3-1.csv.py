from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="列出五个{subject}。\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)
chain = prompt | model | output_parser

print(chain.invoke({"subject": "ice cream flavors"}))

for s in chain.stream({"subject": "ice cream flavors"}):
    print(s)
