import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from chat_zhipu_ai import ChatZhipuAI


class LLMChain:

    def _create_llm(self) -> ChatZhipuAI:
        """
        创建一个智普AI大模型实例。
        """
        # 加载环境变量，获取智普AI API密钥
        load_dotenv()
        ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
        if not ZHIPUAI_API_KEY:
            raise ValueError("ZHIPUAI_API_KEY environment variable not set.")

        # 创建智普AI大模型实例
        return ChatZhipuAI(
            temperature=0.1,
            api_key=ZHIPUAI_API_KEY,
            model_name="glm-4"
        )

    def _create_vectorstore(self, doc_rul: str):
        """
        创建一个向量数据库，用于存储rust编程语言的相关文档。
        """
        # 使用web加载器加载外部文档数据
        web_loader = WebBaseLoader(doc_rul)
        docs = web_loader.load()
        # 使用嵌入模型对外部数据进行向量化，并存入向量数据库
        # 创建嵌入模型
        embeddings = HuggingFaceEmbeddings()
        # 创建分词器
        text_splitter = RecursiveCharacterTextSplitter()
        # 使用分词器拆分外部文档数据
        documents = text_splitter.split_documents(docs)
        # 创建向量数据库
        return FAISS.from_documents(documents, embeddings)

    def __init__(self, name=''):
        self.name = name
        self.llm = self._create_llm()

    def ask(self):
        pass
