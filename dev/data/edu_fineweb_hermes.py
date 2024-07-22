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
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from tqdm import tqdm
import argparse

from data_common import write_datafile
from chat_template import sharegpt_to_chatml
# ------------------------------------------

shard_size = 10**8
fw_version = "10B" # "100B"
local_dir = f"edu_fineweb{fw_version}_hermes"
remote_name = f"sample-{fw_version}T"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

d1 = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
d1 = d1.select_columns(["text"])

d2 = load_dataset("teknium/OpenHermes-2.5", split="train")
d2 = d2.map(sharegpt_to_chatml)
d2 = d2.select_columns(["text"])

# fineweb-edu 10B has 9.7M documents, OpenHermes-2.5 has 1.0M documents

# create three parts of interleaved datasets with the proportion of d2 increasing
# Dataset proportions:
# Part 1: FWE 7,500,000 + OH 0 (0.00%) = 7,500,000
# Part 2: FWE 2,050,000 + OH 500,000 (19.61%) = 2,550,000
# Part 3: FWE 122,101 + OH 501,551 (80.42%) = 623,652
# Total documents: 10,667,043

# TODO: adjust for 100B dataset

d1_p1 = d1.select(range(7_500_000))
d1_p2 = d1.select(range(7_500_000, 9_550_000))
d1_p3 = d1.select(range(9_550_000, len(d1)))
#d2_p1 = d2.select(range(150_000))
d2_p2 = d2.select(range(500_000))
d2_p3 = d2.select(range(500_000, len(d2)))
d1_p1_prob = len(d1_p1) / (len(d1_p1) + 0)
d1_p2_prob = len(d1_p2) / (len(d1_p2) + len(d2_p2))
d1_p3_prob = len(d1_p3) / (len(d1_p3) + len(d2_p3))

print("Dataset proportions:")
print(f"Part 1: FWE {len(d1_p1):,} + OH {0:,} ({1-d1_p1_prob:.2%}) = {len(d1_p1) + 0:,}")
print(f"Part 2: FWE {len(d1_p2):,} + OH {len(d2_p2):,} ({1-d1_p2_prob:.2%}) = {len(d1_p2) + len(d2_p2):,}")
print(f"Part 3: FWE {len(d1_p3):,} + OH {len(d2_p3):,} ({1-d1_p3_prob:.2%}) = {len(d1_p3) + len(d2_p3):,}")

ds = concatenate_datasets([
    d1_p1,
    #interleave_datasets([d1_p1, d2_p1], probabilities=[d1_p1_prob, 1-d1_p1_prob]),
    interleave_datasets([d1_p2, d2_p2], probabilities=[d1_p2_prob, 1-d1_p2_prob]),
    interleave_datasets([d1_p3, d2_p3], probabilities=[d1_p3_prob, 1-d1_p3_prob])
])
print(f"Total documents: {len(ds):,}")

name = "edu_fineweb_hermes"

# init the tokenizer
gpt2_base = tiktoken.get_encoding("gpt2")
# add special tokens for instructions
enc = tiktoken.Encoding(
    name="gpt2_im",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={
        **gpt2_base._special_tokens,
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
    }
)

eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode(doc["text"], allowed_special="all")) # adjusted for chatml special tokens
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, ds, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 100 else "train" # use shard 100 (from Part 3) as validation set
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
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
        split = "val" if shard_index == 100 else "train" # use shard 100 (from Part 3) as validation set
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
