PORT=${1}

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":30, "do_sample":"True", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client1.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":100, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client2.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Where is beijing?","parameters":{"max_new_tokens":20, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client3.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is?","parameters":{"max_new_tokens":80, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client4.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":65, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client5.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":78, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client6.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Where is beijing?","parameters":{"max_new_tokens":300, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client7.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is?","parameters":{"max_new_tokens":50, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client8.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"what is Monetary Policy?","parameters":{"max_new_tokens":200, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client9.log &

curl 127.0.0.1:${PORT}/models/llama2     -X POST     -d '{"inputs":"Hello?","parameters":{"max_new_tokens":55, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' -H 'Content-Type: application/json' &> test_client10.log &

