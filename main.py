from llm_chain import LLMChain
from agent_llm_chain import AgentLLMChain
from retrieval_llm_chain import RetrievalLLMChain
from simple_llm_chain import SimpleLlmChain

if __name__ == "__main__":

    def create_chain(choice) -> LLMChain:
        if choice == "1":
            return SimpleLlmChain()
        elif choice == "2":
            return RetrievalLLMChain()
        elif choice == "3":
            return AgentLLMChain()
        else:
            print("无效的选项，请重新选择。")
            return None
        
    while True:
        print("请选择模型类型：")
        print("1. 简单模型")
        print("2. 检索模型")
        print("3. 代理模型")
        choice = input("请输入选项：")
        llm_chain = create_chain(choice)
        if llm_chain:
            llm_chain.ask()
            break
        
