import os

import requests
from dotenv import load_dotenv


def get_response_from_groq(sentence):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}

    req = requests.post(
        url,
        headers=headers,
        json={
            "model": "gemma2-9b-it",
            "messages": [
                {
                    "role": "user",
                    "content": f"generate sentence out of this words: {sentence}",
                }
            ],
        },
    )
    return req.json()["choices"][0]["message"]["content"]
