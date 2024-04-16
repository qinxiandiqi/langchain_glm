import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chat_zhipu_ai import ChatZhipuAI


def _create_llm() -> ChatZhipuAI:
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


def simple_llm_chain() -> None:
    """
    使用智普AI大模型实现一个简单的聊天对话。
    链结构：输入（input） -> prompt（提示） -> LLM（智普AI大模型） -> 输出解析器（output parser）
    """

    # 创建智普AI大模型实例
    llm = _create_llm()

    # 创建一个简单的ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个世界级的技术文档作者。"),
        ("user", "{input}"),
    ])

    # 创建一个简单的字符串输出解析器
    output_parser = StrOutputParser()

    # 创建一个ChatChain，将上述组件组合起来
    chain = prompt | llm | output_parser

    # 调用ChatChain，并打印结果
    result = chain.invoke({"input": "如何自动生成python项目的pip依赖库清单？"})
    print(result)


def _create_rust_vectorstore():
    """
    创建一个向量数据库，用于存储rust编程语言的相关文档。
    """
    # 使用web加载器加载外部文档数据
    web_loader = WebBaseLoader(
        "https://www.rust-lang.org/zh-CN/"
    )
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


def retrieval_llm_chain() -> None:
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

    # 创建一个智普AI大模型实例
    llm = _create_llm()

    # 创建向量数据库
    vectorstore = _create_rust_vectorstore()

    # 创建文档链：输入检索到的文档数据，以及问题
    # 先创建上游提示词模板
    prompt = ChatPromptTemplate.from_template(
        """
        仅根据提供的上下文回答以下问题：
        <context>
        {context}
        </context>
        问题：{input}
        """
    )
    document_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)

    # 创建检索器选择最相关文档，传给检索链
    retriever = vectorstore.as_retriever()
    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 调用检索链，并打印结果
    result = retrieval_chain.invoke({"input": "rust编程语言的优势在哪里？"})
    print(result["answer"])


def agent_llm_chain() -> None:
    """
    使用智普AI大模型实现一个简单的智能体。
    """

    llm = _create_llm()

    # 创建检索器
    vectorstore = _create_rust_vectorstore()
    retriever = vectorstore.as_retriever()

    # 为检索器设置一个工具，名为"rust_search"
    retriever_tool = create_retriever_tool(
        retriever, name="rust_search", description="Search for rust related documents.")

    # 创建Tavily搜索工具
    search_tool = TavilySearchResults()

    tools = [retriever_tool, search_tool]

    # 从hub上获取智能体提示词
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 调用智能体，并打印结果
    result = agent_executor.invoke({"input": "rust编程语言的优势在哪里？"})
    print(result)

if __name__ == "__main__":
    # simple_llm_chain()
    # retrieval_llm_chain()
    agent_llm_chain()
