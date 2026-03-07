import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=r"D:\RAG\.env")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


def generate_answer_openai(query: str, context: str) -> str:
    system_prompt = (
        "你是一个严谨的中文问答助手。只能依据给定【上下文】回答；"
        "如果信息不足，不要编造，答案尽量不超过200字。"
    )

    user_prompt = f"""【上下文】
{context}

【问题】
{query}
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()