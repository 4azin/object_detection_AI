import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_gms_vision():
    gms_key = os.getenv("GMS_KEY")
    url = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gms_key}"
    }

    # 1x1 투명 PNG
    tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    payload = {
        "model": "gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{tiny_b64}"
                        }
                    }
                ]
            }
        ],
        "max_completion_tokens": 100
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print("Status", response.status_code)
        print("Response", response.text)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test_gms_vision()
