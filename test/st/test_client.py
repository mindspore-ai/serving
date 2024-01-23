from mindspore_serving.client import MindsporeInferenceClient

client = MindsporeInferenceClient(model_type="llama2", server_url="http://127.0.0.1:8080")

# 1. test generate
text = client.generate("what is Monetary Policy?").generated_text
print('text: ', text)

# 2. test generate_stream
text = ""
for response in client.generate_stream("what is Monetary Policy?", do_sample=False, max_new_tokens=200):
    print("response 0", response)
    if response.token:
        text += response.token.text
    else:
        text = response.generated_text
print(text)

# 3. test do_sample, return_full_text
text = ""
for response in client.generate_stream("what is Monetary Policy?", do_sample=True, return_full_text=False):
    print("response 1 ", response)
    if response.token:
        text += response.token.text
print(text)

# 4. test top_k temperature
text = ""
for response in client.generate_stream("what is Monetary Policy?", do_sample=True, temperature=0.9, top_k=3):
    print("response 2 ", response)
    if response.token:
        text += response.token.text
    else:
        text = response.generated_text
print(text)

# 5. test top_k=100 top_p max_new_tokens
text = ""
for response in client.generate_stream("what is Monetary Policy?", do_sample=True, temperature=0.9, top_p=0.8,
                                       top_k=100, max_new_tokens=200):
    print("response 3 ", response)
    if response.token:
        text += response.token.text
    else:
        text = response.generated_text
print(text)