"""Token ids and a rendered chat template from the official OLMoE-1B-7B-0125-Instruct
tokenizer, via transformers.

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/olmoe_tokenizer.py <dir with tokenizer.json> \
        > fixtures/oracle/olmoe_tokenizer.json

<dir> holds tokenizer.json, tokenizer_config.json and special_tokens_map.json from
huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct. Checked by the env-gated tests in
crates/qubivex-runtime/tests/tokenizer.rs against the tokenizer QUBIVEX rebuilds from
the GGUF file, and against the same tokenizer.json loaded by QUBIVEX.
"""
import json
import sys

import transformers
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
texts = [
    "Hello world, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and trailing   ",
    "runs of spaces:" + " " * 2 + "a" + " " * 7 + "b" + " " * 24 + "c" + " " * 30 + "d",
    "tabs\tand\nnewlines\n\n\nand\r\nCRLF",
    "café vs café (precomposed vs combining accent)",
    "naïve résumé façade jalapeño über straße",
    "数字: 一二三, 日本語のテキスト, 한국어 텍스트",
    "emoji 🙂👍🏽🇰🇪 and symbols © ™ ∑ ∞ → ←",
    "def f(x):\n    return x ** 2  # square\n",
    "3.14159 2026-11-03 1,000,000 0x1F 1e-6 -42",
    "<|endoftext|> special inside text <|padding|>",
    "|||IP_ADDRESS||| and |||EMAIL_ADDRESS||| and |||PHONE_NUMBER|||",
    "<|user|>\nWhat is a mixture of experts?\n<|assistant|>\n",
    "",
    "a",
    "'s 't 're 've 'm 'll 'd contractions: don't, I'm, we'll",
]
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a mixture-of-experts model?"},
    {"role": "assistant", "content": "A model that routes each token to a few experts."},
    {"role": "user", "content": "Why does that save memory bandwidth?"},
]
chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
json.dump({
    "generator": "fixtures/oracle/olmoe_tokenizer.py",
    "transformers": transformers.__version__,
    "source": "allenai/OLMoE-1B-7B-0125-Instruct tokenizer.json",
    "cases": [
        {"text": t, "ids": (ids := tok.encode(t, add_special_tokens=False)), "decoded": tok.decode(ids)}
        for t in texts
    ],
    "chat": {
        "messages": messages,
        "add_generation_prompt": True,
        "rendered": chat,
        "ids": tok.encode(chat, add_special_tokens=False),
    },
}, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")
