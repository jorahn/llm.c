"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}

Example of downloading the 100B dataset of FineWebEDU, from root directory:
python dev/data/fineweb.py -t edu -v 100B
100B runs for small few hours, depending on your internet and computer.
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from tqdm import tqdm
import argparse

from data_common import write_datafile
# ------------------------------------------

from transformers import AutoTokenizer

tkc = AutoTokenizer.from_pretrained("gpt2")
tkc.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

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
    formatted = tkc.apply_chat_template(chatml_conversations, tokenize=False, add_generation_prompt=False)
    return {"text": formatted}
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, default="edu", help="Fineweb type, edu|classic")
parser.add_argument("-v", "--version", type=str, default="10B", help="Fineweb data sample size, 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in {"10B", "100B"}, "version must be one of: 10B, 100B"
assert args.type in {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B_hermes", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B_hermes", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B_hermes", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B_hermes", "sample-100BT")
}
local_dir, remote_name = directories[(args.type, args.version)]

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
if args.type == "classic":
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    name = "fineweb_hermes"
elif args.type =="edu":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    name = "edu_fineweb_hermes"

oh = load_dataset("teknium/OpenHermes-2.5", split="train")
oh = oh.map(sharegpt_to_chatml)
oh = oh.select_columns(["text"])

fw_1 = fw.select(range(len(fw)//2))
fw_2 = fw.select(range(len(fw)//2, len(fw)-500_000))
fw_3 = fw.select(range(len(fw)-500_000, len(fw)))
oh_1 = oh.select(range(100_000))
oh_2 = oh.select(range(100_000, 500_000))
oh_3 = oh.select(range(500_000, len(oh)))

p1 = len(fw_1) / (len(fw_1) + len(oh_1))
p2 = len(fw_2) / (len(fw_2) + len(oh_2))
p3 = len(fw_3) / (len(fw_3) + len(oh_3))

# interleaving the two datasets
ds = concatenate_datasets([
    interleave_datasets([fw_1, oh_1], probabilities=[p1, 1-p1]),
    interleave_datasets([fw_2, oh_2], probabilities=[p2, 1-p2]),
    interleave_datasets([fw_3, oh_3], probabilities=[p3, 1-p3])
])
print(f"Dataset proportions:")
print(f"Part 1: FWE {len(fw_1):,} + OH {len(oh_1):,} ({1-p1:.2%}) = {len(fw_1) + len(oh_1):,}")
print(f"Part 2: FWE {len(fw_2):,} + OH {len(oh_2):,} ({1-p2:.2%}) = {len(fw_2) + len(oh_2):,}")
print(f"Part 3: FWE {len(fw_3):,} + OH {len(oh_3):,} ({1-p3:.2%}) = {len(fw_3) + len(oh_3):,}")
print(f"Total documents: {len(ds):,}")

#Dataset proportions:
#Part 1: FWE 4,836,050 + OH 100,000 (2.03%) = 4,936,050
#Part 2: FWE 4,336,051 + OH 400,000 (8.45%) = 4,736,051
#Part 3: FWE 500,000 + OH 501,551 (50.08%) = 1,001,551
#Total documents: 10,669,024

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, ds, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
