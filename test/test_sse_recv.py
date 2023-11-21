import json
import requests
import sseclient
import ssl

ssl._create_default_https

query = ("What is Montary policy？")

payload = json.dumps(
    {
        "inputs": query,
        "parameters": {
            "do_sample": False,
            "max_new_tokens": 512
        }
    }
)


headers = {
    "Accept": "*/*",
    "Cache-Control": "no-cache",
    "Content-type": "application/json",
    "accept-encoding": "gzip, default",
}

response_bs = requests.get("https://127.0.0.1:9811/serving/get_bs", headers=headers, stream=False, verify=False)
response_req_num = requests.get("https://127.0.0.1:9811/serving/get_request_numbers", headers=headers, stream=False, verify=False)
#response = requests.request("POST", "https://127.0.0.1:9811/models/llama2/generate_stream", stream=True, headers=headers, data=payload)

#client = sseclient.SSEClient(response)
client_bs = sseclient.SSEClient(response_bs)
response_req_num = sseclient.SSEClient(response_req_num)
"""
for event in client.events():
    print(event.data)
    print("-----------------------------------------------")
"""
for event in client_bs.events():
    print(event.data)
    print("-----------------------------------------------")

for event in client_req_num.event():
    print(event.data)
    print("-----------------------------------------------")
