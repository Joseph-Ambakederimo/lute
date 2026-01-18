import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

# Get the directory of this file and resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenize")
LOCAL_DIR = os.path.join(PROJECT_ROOT, "data")

class TextDataset(Dataset):
    def __init__(self, tokenizer_path=TOKENIZER_PATH, seq_len=512, num_docs=20000):
        print("loading tokenizer...")
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt")
        )

        print("loading local + wikitext2 data...")
        self.texts = self._load_texts(num_docs)
        print(f"loaded {len(self.texts)} documents")

        print("tokenizing and chunking...")
        self.tokens = self._tokenize_and_chunk(seq_len)
        print(f"created {len(self.tokens)} training chunks (sequences of length {seq_len})")


    def _load_texts(self, num_docs):
        texts = []
        
        # local .txt files
        for root, _, files in os.walk(LOCAL_DIR):
            for f in files:
                if f.endswith(".txt"):
                    with open(os.path.join(root, f), "r", encoding="utf-8") as src:
                        texts.append(src.read())

        # wikitext-2 dataset (used for evaluation/generalization)
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        wiki_train = [x["text"] for x in wiki if len(x["text"]) > 0]
        texts.extend(wiki_train[:num_docs])

        return texts

    def _tokenize_and_chunk(self, seq_len):
        """Encodes all texts, flattens them, and splits into fixed-length chunks."""
        # 1. Tokenize all texts deeply
        all_token_ids = []
        print(f"Tokenizing {len(self.texts)} documents...")
        for text in tqdm(self.texts, desc="tokenizing"):
            # Encode text to list of token IDs (no truncation related to seq_len here)
            ids = self.tokenizer.encode(text).ids
            # Optional: Add EOS token between documents if desired. 
            # For now, we'll just concatenate them directly (typical for pretraining)
            # or you might want to add a specialized separator.
            all_token_ids.extend(ids)
            # If your model expects end-of-text tokens, append them here:
            # all_token_ids.append(self.tokenizer.token_to_id("</s>"))

        print(f"Total tokens: {len(all_token_ids)}")

        # 2. Chunk the flat list
        chunks = []
        # We drop the last partial chunk if it's smaller than seq_len
        # to ensure all batches are the same shape.
        for i in range(0, len(all_token_ids) - seq_len + 1, seq_len):
            chunk = all_token_ids[i : i + seq_len]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
        
        return chunks

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # The key for next token prediction:
        # Input (x) is tokens [0] to [T-1]
        # Target (y) is tokens [1] to [T] (shifted by one)
        data = self.tokens[idx]
        return data[:-1], data[1:] # input, target