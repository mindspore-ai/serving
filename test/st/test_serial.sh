PORT=${1}

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":100, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Where is beijing?","parameters":{"max_new_tokens":20, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is?","parameters":{"max_new_tokens":80, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":65, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":78, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Where is beijing?","parameters":{"max_new_tokens":300, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is?","parameters":{"max_new_tokens":50, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":200, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":55, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

# 流式
curl 127.0.0.1:${PORT}/models/llama2/generate_stream     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'

curl 127.0.0.1:${PORT}/models/llama2/generate_stream     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"False"}, "stream":"True"}' -H 'Content-Type: application/json'
# 默认值
curl 127.0.0.1:${PORT}/models/llama2/generate_stream     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{}}' -H 'Content-Type: application/json'

# 非流式
curl 127.0.0.1:${PORT}/models/llama2/generate     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"False"}' -H 'Content-Type: application/json'
# 默认值
curl 127.0.0.1:${PORT}/models/llama2/generate     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{}}' -H 'Content-Type: application/json'

