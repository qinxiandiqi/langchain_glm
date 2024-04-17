from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from llm_chain import LLMChain


class RetrievalLLMChain(LLMChain):
    """
    使用检索链。
    步骤：
    1. WebBaseLoader加载外部web文档数据
    2. 使用分词器拆分文档数据，防止数据过长
    3. 使用嵌入模型将文档数据向量化
    4. 使用向量数据库存储向量数据，建立索引
    5. 创建文档链，文档链的作用是，接受检索到的数据和问题两个参数，一块送给大模型
    6. 创建检索器，检索器的作用是选择最相关文档
    7. 将检索器和文档链串起来，组成检索链，检索器负责给检索链提供需要的检索数据参数
    8. 调用检索链，输入剩余的问题参数
    """

    def __init__(self):
        super().__init__(name="RetrievalLLMChain")

        # 创建文档链：输入检索到的文档数据，以及问题
        # 先创建上游提示词模板
        self.prompt = ChatPromptTemplate.from_template(
            """
            仅根据提供的上下文回答以下问题：
            <context>
            {context}
            </context>
            问题：{input}
            """
        )
        self.document_chain = create_stuff_documents_chain(
            prompt=self.prompt, llm=self.llm)

    def _create_retrieval_chain(self, doc_url: str = "https://doc.rust-lang.org/book/"):
        # 创建向量数据库
        vectorstore = self._create_vectorstore(doc_rul=doc_url)
        # 创建检索器选择最相关文档，传给检索链
        retriever = vectorstore.as_retriever()
        # 创建检索链
        return create_retrieval_chain(retriever, self.document_chain)

    def ask(self):
        while True:
            doc = input("请输入外部文档地址（默认rust文档，输入exit结束程序）：")
            if doc == "exit":
                break
            retrieval_chain = self._create_retrieval_chain(
                doc) if doc else self._create_retrieval_chain()
            while True:
                question = input("请输入问题（输入done结束提问）：")
                if question == "exit":
                    return
                if question == "done":
                    break
                result = retrieval_chain.invoke({"input": question})
                print(result["answer"])
