'''
Data Loading and doing the Pre-processing of the data before
feeding it to the Transformer.
'''
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import math

print("Loading TinyStories (10%)...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")
print(f"Dataset size: {len(dataset)} stories")

# Buidling the Custom Tokenizer for the TinyStories Dataset
print("\nBuilding custom tokenizer from TinyStories...")

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Train tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<pad>", "<eos>", "<bos>"],
    show_progress=True,
    min_frequency=2
)

# Train on TinyStories text
tokenizer.train_from_iterator(
    dataset["text"],
    trainer=trainer,
    length=len(dataset)
)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    eos_token="<eos>",
    bos_token="<bos>",
    unk_token="<pad>"  
)
vocab_size = custom_tokenizer.vocab_size
print(f"✓ Custom tokenizer built!")
print(f"Vocabulary size: {vocab_size}")
print(f"Sample tokens: {list(custom_tokenizer.get_vocab().keys())[:20]}")
print("\nTokenizing dataset...")

def tokenize_function(examples):
    return custom_tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4
)

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
print(f"✓ Tokenization complete!")
print("\n" + "="*50)
print("CREATING TRAIN/VALIDATION/TEST SPLITS")
print("="*50)

from torch.utils.data import random_split

# Calculate split sizes
total_size = len(tokenized_dataset)
train_size = int(0.8 * total_size)  # 80% train
val_size = int(0.1 * total_size)    # 10% validation
test_size = total_size - train_size - val_size  # 10% test

print(f"\nTotal dataset size: {total_size:,} stories")
print(f"Split ratios: 80% train / 10% val / 10% test")

# Split dataset with fixed random seed for reproducibility
train_dataset, val_dataset, test_dataset = random_split(
    tokenized_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_dataset):,} stories ({len(train_dataset)/total_size*100:.1f}%)")
print(f"  Val:   {len(val_dataset):,} stories ({len(val_dataset)/total_size*100:.1f}%)")
print(f"  Test:  {len(test_dataset):,} stories ({len(test_dataset)/total_size*100:.1f}%)")

print("\n" + "="*50)
print("CREATING DATALOADERS")
print("="*50)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,      # Shuffle training data
    num_workers=2,
    pin_memory=True,
    drop_last=True     # Drop incomplete last batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,     # Don't shuffle validation
    num_workers=2,
    pin_memory=True,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,     # Don't shuffle test
    num_workers=2,
    pin_memory=True,
    drop_last=False
)

print(f"\n✓ DataLoaders ready!")
print(f"\nDataLoader Statistics:")
print(f"  Train batches: {len(train_loader):,} (batch size: 64)")
print(f"  Val batches:   {len(val_loader):,} (batch size: 64)")
print(f"  Test batches:  {len(test_loader):,} (batch size: 64)")
print(f"\nTokens per batch: {64 * 256:,}")
print(f"Train tokens per epoch: {len(train_loader) * 64 * 256:,}")


'''
Model Training Part - Loading the Custom ChatBot(Transformer-Decoder) code
and writing the Training Loop to train the model on the 
preprocessed TinyStories Dataset.
'''
from torch.optim.lr_scheduler import OneCycleLR
import math

model = Chatbot(vocab_size=vocab_size, heads = 6, d_model = 384, hid_layer = 1536, seq_len = 256, num_layers = 4, dropout = 0.1).cuda()
# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,  # Start with lower LR
    betas=(0.9, 0.999),  # More conservative beta2
    eps=1e-8,
    weight_decay=0.01
)
criterion = nn.CrossEntropyLoss(
    ignore_index=custom_tokenizer.pad_token_id,
    label_smoothing=0.05  # Reduced smoothing
)

# Scheduler - reduces LR when validation loss plateaus
epochs = 30
total_steps = len(train_loader) * epochs
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25,  # Start lr = max_lr/25
    final_div_factor=1000  # End lr = max_lr/1000
)

# A100 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training config
best_val_loss = float('inf')
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50 + "\n")

import time
start_time = time.time()

for epoch in range(epochs):
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch+1}/{epochs}")
    print(f"{'='*50}")
    
    model.train()
    train_loss = 0
    train_steps = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].cuda()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            
            # Shift for next-token prediction
            logits = outputs[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            
            # Calculate loss
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Track metrics
        train_loss += loss.item()
        train_steps += 1
        
        # Print progress every 100 steps
        if (batch_idx + 1) % 100 == 0:
            avg_loss = train_loss / train_steps
            perplexity = math.exp(min(avg_loss, 10))
            print(f"  Step {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"PPL: {perplexity:.2f}")
    
    # Calculate average training loss
    avg_train_loss = train_loss / train_steps
    train_perplexity = math.exp(min(avg_train_loss, 10))
    
    print(f"\n  ✓ Training complete")
    print(f"    Avg Loss: {avg_train_loss:.4f}")
    print(f"    Perplexity: {train_perplexity:.2f}")
    
    print(f"\n  Running validation...")
    model.eval()
    val_loss = 0
    val_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].cuda()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                
                logits = outputs[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                
                loss = criterion(
                    logits.view(-1, vocab_size),
                    targets.view(-1)
                )
            
            val_loss += loss.item()
            val_steps += 1
    
    # Calculate average validation loss
    avg_val_loss = val_loss / val_steps
    val_perplexity = math.exp(min(avg_val_loss, 10))
    
    print(f"  ✓ Validation complete")
    print(f"    Val Loss: {avg_val_loss:.4f}")
    print(f"    Val Perplexity: {val_perplexity:.2f}")
    print(f"\n  Epoch Summary:")
    print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"    Train PPL: {train_perplexity:.2f} | Val PPL: {val_perplexity:.2f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, 'best_tinystories_model.pt')
        print(f"\n  💾 Best model saved! Val Loss: {avg_val_loss:.4f}")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)

# Save tokenizer
custom_tokenizer.save_pretrained("./tinystories_tokenizer")
print("✓ Tokenizer saved to './tinystories_tokenizer'")

# Save final model
torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, 'final_tinystories_model.pt')
print("✓ Final model saved to 'final_tinystories_model.pt'")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Perplexity: {math.exp(min(best_val_loss, 10)):.2f}")
print("\n🎉 Training complete! Ready for testing and generation.")