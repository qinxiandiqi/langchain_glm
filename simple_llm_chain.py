from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_chain import LLMChain


class SimpleLlmChain(LLMChain):
    """
    使用智普AI大模型实现一个简单的聊天对话。
    链结构：输入（input） -> prompt（提示） -> LLM（智普AI大模型） -> 输出解析器（output parser）
    """

    def __init__(self):
        super().__init__(name="SimpleLlmChain")
        # 创建一个简单的ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个世界级的技术文档作者。"),
            ("user", "{input}"),
        ])
        # 创建一个简单的字符串输出解析器
        self.output_parser = StrOutputParser()
        # 创建一个ChatChain，将上述组件组合起来
        self.chain = self.prompt | self.llm | self.output_parser

    def ask(self):
        while True:
            user_input = input("请输入您的问题（输入done结束问答）: ")

            if user_input == "done":
                break

            # 调用ChatChain的ask方法，传入用户输入
            response = self.chain.invoke({"input": user_input})
            print(response)
