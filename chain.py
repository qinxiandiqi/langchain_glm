import os

from dotenv import load_dotenv

from chat_zhipu_ai import ChatZhipuAI

class Chain:

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
    
    def __init__(self, name=''):
        self.name = name
        self.llm = Chain._create_llm()

    def ask(self, question: str):
        print(f'ask {self.name}: {question}')