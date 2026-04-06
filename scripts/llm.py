import os
from dotenv import load_dotenv
import requests


# Load environment variables from the .env file
load_dotenv()

def request(text:str, rag_txts: list[str]):
    '''Request response from llm'''
    if len(rag_txts)>0:
        system_txt = f"{os.environ['SYSTEM_P']}"
        for index, rag_txt in enumerate(rag_txts):
            system_txt += f"\n{index}. {rag_txt}"
        system_txt += f"\n\n"
    else:
        system_txt = f"{os.environ['SYSTEM_P']}\n1. 沒有相關資料。"
    response = requests.post(
        f"{os.environ['API_URL']}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['API_TOKEN']}",
            "Content-Type": "application/json"
        },
        json={
            "model": os.environ['MODEL'],
            "messages": [
                {"role": "system", "content": system_txt},
                {"role": "user", "content": text}
            ]
        }
    )
    # print(json.dumps(response.json(), indent=2))
    result = response.json()["choices"][0]["message"]["content"]
    return result