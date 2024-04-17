from agent_llm_chain import AgentLLMChain
from retrieval_llm_chain import RetrievalLLMChain
from simple_llm_chain import SimpleLlmChain

if __name__ == "__main__":
    while True:
        print("请选择模型类型：")
        print("1. 简单模型")
        print("2. 检索模型")
        print("3. 代理模型")
        choice = input("请输入选项：")

        if choice == "1":
            simple_llm_chain = SimpleLlmChain()
            simple_llm_chain.ask()
            break
        elif choice == "2":
            retrieval_llm_chain = RetrievalLLMChain()
            retrieval_llm_chain.ask()
            break
        elif choice == "3":
            agent_llm_chain = AgentLLMChain()
            agent_llm_chain.ask()
            break
        else:
            print("无效的选项，请重新选择。")
