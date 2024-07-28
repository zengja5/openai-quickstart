import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from openai import OpenAI

import os
import random
import argparse
import ast

os.environ["OPENAI_API_KEY"] = "sk-iIWdN3LYHyHnD67P83E8E2Bf3e5f44F0Ac03E13175Af0a32"
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"


def initialize_sales_bot(vector_store_dir: str="network_service_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    SALES_BOT.return_source_documents = True

    return SALES_BOT

# 后备的question列表
fallback_questions = [
    "对不起，我需要更多的信息才能回答您的问题。您能提供一下吗？",
    "我需要更多的信息来帮助您。您能详细描述一下吗？",
    "我暂时无法回答您的问题。您有其他问题我可以帮助的吗？",
]

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    ans = SALES_BOT({"query": message})

    client = OpenAI()
    messages=[
        {"role": "system", "content": '你是一个礼貌的、乐于助人的网络运营商的人工客服。'},
    ]
    
    # 如果检索出结果，返回结果
    if ans["source_documents"]:
        return ans["result"]
    # 如果没有检索出结果但启用了语言模型，返回设定模型的系统身份并回答该问题
    elif enable_chat:
        messages.append({"role": "user", "content": message})
        data = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = messages
        )
        return data.choices[0].message.content
    # 否则返回后备的question
    else:
        return random.choice(fallback_questions)
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="网络运营商客服",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

def parse_arguments():
    parser = argparse.ArgumentParser(description='A network customer service assistant that answer customer questions.')
    parser.add_argument('--enable_chat', type=ast.literal_eval, help='whether to use LLM assistant.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # enable_chat 参数化
    enable_chat = parse_arguments().enable_chat
    print(f"enable_chat: {enable_chat}")
    # 初始化客服机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
