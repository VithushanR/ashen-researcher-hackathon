from dotenv import load_dotenv
load_dotenv()

import os
import requests

api_key = os.environ["OPENROUTER_API_KEY"]

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "nex-agi/nex-n2.5-pro:free",
        "messages": [{"role": "user", "content": "hi"}],
    },
)

print("Status:", r.status_code)
print("Body:", r.text)