import time
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer

def test_comparison():
    tokenizer = Baichuan2Tokenizer("./baichuan2_tokenizer.model")
    tokens = [100] * 10000
    start_time = time.time()
    for token in tokens:
        tokenizer.decode([token], skip_special_tokens=True)
    print("iteratively detokenize takes ", (time.time() - start_time) * 1000, "ms")

    start_time = time.time()
    tokenizer.decode(tokens, skip_special_tokens=True)
    print("batch detokenize takes ", (time.time() - start_time) * 1000, "ms")

if __name__ == "__main__":
    test_comparison()
    # iteratively detokenize takes  650.489091873169 ms
    # batch detokenize takes  534.1897010803223 ms
    # 结论：batch推理确实会提升性能