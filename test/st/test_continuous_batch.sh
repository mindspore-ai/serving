PORT=${1}

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":200, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":200, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":512, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":500, "do_sample":"False", "top_k":5.0, "top_p":0.8,"return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":300, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":100, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":300, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":256, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":600, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":400, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &


