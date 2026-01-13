import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

# Get the directory of this file and resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_PATH = os.path.join(PROJECT_ROOT, "tokenize")
os.makedirs(SAVE_PATH, exist_ok=True)

def get_local_files(local_dir):
    """Recursively finds all .txt files in the local data directory."""
    txt_files = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    return txt_files

def prepare_corpus():
    """Combines local and HuggingFace data into a single corpus file for tokenizer training."""
    # 1. Collect all local .txt files
    local_files = get_local_files(LOCAL_DIR)
    print(f"found {len(local_files)} local text files")

    # 2. Load wikitext-2 from Hugging Face
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_train = [x["text"] for x in wiki if len(x["text"]) > 0]

    # 3. Write combined corpus
    combined_path = "./corpus.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for file_path in local_files:
            with open(file_path, "r", encoding="utf-8") as src:
                f.write(src.read() + "\n")
        
        # Add WikiText lines
        for line in wiki_train:
            f.write(line + "\n")

    print(f"combined corpus saved to {combined_path}")
    return combined_path

def train_tokenizer(corpus_path):
    """Trains a ByteLevelBPE Tokenizer."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_path],
        vocab_size=50000, # Should match ModelConfig.vocab_size
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    # The tokenizer saves vocab.json and merges.txt inside the SAVE_PATH
    tokenizer.save_model(SAVE_PATH)
    print(f"✅ tokenizer saved to {SAVE_PATH}")

def load_checkpoint(model, optimizer, scaler, path, device):
    """Load model, optimizer, and scaler state from a checkpoint."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"✅ Checkpoint loaded from {path}, resuming from step {start_step}")
        return start_step
    return 0


def save_checkpoint(model, optimizer, scaler, step, path):
    """Save model, optimizer, and scaler state to a checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved to {path}")


def train(model, config, checkpoint_dir="./checkpoints"):
    """Main training loop."""
    from training.dataset import TextDataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = TextDataset(seq_len=config.max_seq_len, num_docs=config.num_documents)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer and Scaler
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=config.muon_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    step = 0
    
    print(f"🚀 Starting training for {config.max_steps} steps on {device}")
    
    with tqdm(total=config.max_steps, desc="Training") as pbar:
        for epoch in range(100):  # Large number of epochs, we'll break on max_steps
            for batch_idx, (x, y) in enumerate(dataloader):
                if step >= config.max_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass with AMP if enabled
                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                else:
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                
                # Backward pass with gradient accumulation
                if config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
                
                # Evaluation and checkpoint
                if step % config.eval_every == 0:
                    print(f"\n💾 Step {step}: Saving checkpoint...")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                    save_checkpoint(model, optimizer, scaler, step, checkpoint_path)
                    
                    # Also save as latest
                    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
                    save_checkpoint(model, optimizer, scaler, step, latest_path)
                
                if step >= config.max_steps:
                    break
            
            if step >= config.max_steps:
                break
    
    print(f"✅ Training completed at step {step}")


def evaluate(model, seq_len=512, num_docs=200, device="cpu"):
    """Simple evaluation function."""
    from training.dataset import TextDataset
    
    model.to(device)
    model.eval()
    
    print("📊 Loading evaluation dataset...")
    dataset = TextDataset(seq_len=seq_len, num_docs=num_docs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, model.config.vocab_size), y.view(-1))
            total_loss += loss.item()
            num_batches += 1
    
    if num_batches == 0:
        print("⚠️  Evaluation dataset is empty (no chunks created). Skipping evaluation.")
        return None
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"📈 Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss


if __name__ == "__main__":
    # Ensure the data directory exists for the combined corpus file
    os.makedirs(".data", exist_ok=True)
    # You must manually ensure ../data/local_texts contains some .txt files or create it.
    corpus = prepare_corpus()
    train_tokenizer(corpus)