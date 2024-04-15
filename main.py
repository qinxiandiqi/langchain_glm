from chat_zhipu_ai import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
    if not ZHIPUAI_API_KEY:
        raise ValueError("ZHIPUAI_API_KEY environment variable not set.")

    llm = ChatZhipuAI(
        temperature = 0.1,
        api_key = ZHIPUAI_API_KEY,
        model_name="glm-4"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个世界级的技术文档作者。"),
        ("user", "{input}"),
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    result = chain.invoke({"input": "如何自动生成python项目的pip依赖库清单？"})
    print(result)