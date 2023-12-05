PORT=${1}

echo "batch size is 1"
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json'
sleep 3
echo "batch size is 4"
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
sleep 3
echo "batch size is 8"
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
sleep 3
echo "batch size is 2"
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
sleep 3
echo "batch size is 1"
curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &
