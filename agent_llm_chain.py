from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults

from llm_chain import LLMChain


class AgentLLMChain(LLMChain):

    def __init__(self):
        super().__init__(name="AgentLLMChain")
        # 创建Tavily搜索工具
        self.tavily_tool = TavilySearchResults()
        # 从hub上获取智能体提示词
        self.prompt = hub.pull("hwchase17/openai-tools-agent")
        self.prompt.pretty_print()

    def _create_retriever_tool(self, doc_url: str, tool_name: str, tool_description: str):
        # 创建检索器
        vectorstore = self._create_vectorstore(doc_rul=doc_url)
        retriever = vectorstore.as_retriever()
        # 创建一个检索工具，用于从文档中检索信息
        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name=tool_name,
            description=tool_description,
        )
        return retriever_tool

    def ask(self):
        while True:
            doc = input("请输入外部文档地址：")
            if doc == "exit":
                return
            if not doc:
                print("请输入有效的文档地址")
            else:
                break
        while True:
            tool_name = input("请输入工具名称：")
            if tool_name == "exit":
                return
            if not tool_name:
                print("请输入有效的工具名称")
            else:
                break
        while True:
            tool_description = input("请输入工具描述：")
            if tool_description == "exit":
                return
            if not tool_description:
                print("请输入有效的工具描述")
            else:
                break

        retriever_tool = self._create_retriever_tool(
            doc_url=doc, tool_name=tool_name, tool_description=tool_description)
        tools = [retriever_tool, self.tavily_tool]
        agent = create_openai_tools_agent(
            llm=self.llm, tools=tools, prompt=self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        while True:
            question = input("请输入您的问题（输入exit结束程序）：")
            if question == "exit":
                return
            if not question:
                print("请输入有效的问题")
            else:
                # 调用智能体，并打印结果
                result = agent_executor.invoke({"input": question})
                print(result)
