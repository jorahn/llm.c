from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def sharegpt_to_chatml(example):
    chatml_conversations = []
    for conv in example["conversations"]:
        if conv["from"] == "human":
            role = "user"
        elif conv["from"] == "system":
            role = "system"
        elif conv["from"] == "gpt":
            role = "assistant"
        else:
            role = "user"
        chatml_format = {"role": role, "content": conv["value"]}
        chatml_conversations.append(chatml_format)
    formatted = tokenizer.apply_chat_template(chatml_conversations, tokenize=False, add_generation_prompt=False)
    return {"text": formatted}

ds = load_dataset("BAAI/Infinity-Instruct", "0625", split="train")
ds = ds.map(sharegpt_to_chatml)
ds = ds.select_columns(["text"])

ds.push_to_hub("jrahn/Infinity-Instruct_chatml")
