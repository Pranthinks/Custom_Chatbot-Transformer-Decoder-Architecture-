# train_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# Import your model
from transformer_code import Chatbot

def train(rank, world_size, train_data, val_data, tokenizer):
    # Initializing the Process Group and device
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Creating the Train_data loader and Sampler for training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, batch_size=32, sampler=train_sampler, 
        num_workers=2, pin_memory=True, drop_last=True
    )

    # Creating sampler and loaders for Val data
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(
        val_data, batch_size=32, sampler=val_sampler, 
        num_workers=2, pin_memory=True, drop_last=False
    )

    vocab_size = tokenizer.vocab_size

    # Loading the model
    model = Chatbot(
        vocab_size=vocab_size, heads=6, d_model=384, 
        hid_layer=1536, seq_len=256, num_layers=4, dropout=0.1
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    # Loading Optimizer and Criterion
    epochs = 1
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.05)

    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-4, total_steps=total_steps, 
        pct_start=0.1, anneal_strategy='cos', 
        div_factor=25, final_div_factor=1000
    )
    scaler = GradScaler('cuda')

    # Training Loop
    for i in range(epochs):
        train_sampler.set_epoch(i)

        model.train()
        train_loss = 0
        
        for key, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(rank)

            with autocast('cuda'):
                output = model(input_ids)
                first = output[:, :-1, :].contiguous().view(-1, vocab_size)
                second = input_ids[:, 1:].contiguous().view(-1)
                loss = criterion(first, second)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
            if rank == 0 and (key + 1) % 100 == 0:
                avg_loss = train_loss / (key + 1)
                print(f"Epoch {i+1}/{epochs} | Batch {key+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        # Proper indentation - INSIDE epoch loop
        avg_train_loss = train_loss / len(train_loader)
        train_loss_tensor = torch.tensor([avg_train_loss], device=rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        avg_train_loss = train_loss_tensor.item()

        if rank == 0:
            print(f"Epoch {i+1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation code - properly indented INSIDE epoch loop
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for key, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(rank)

                with autocast('cuda'):
                    val_op = model(input_ids)
                    first = val_op[:, :-1, :].contiguous().view(-1, vocab_size)
                    second = input_ids[:, 1:].contiguous().view(-1)
                    loss = criterion(first, second)
                    
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_tensor = torch.tensor([avg_val_loss], device=rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_val_loss = val_loss_tensor.item()
        
        if rank == 0:
            print(f"Epoch {i+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")

    # Model saving - OUTSIDE the epoch loop
    if rank == 0:
        print("Saving model checkpoint...")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, 'chatbot_checkpoint.pt')
        print("Model saved!")

    dist.barrier()
    dist.destroy_process_group()


def main(train_data, val_data, tokenizer):
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size, train_data, val_data, tokenizer),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    print("Loading preprocessed data...")
    data = torch.load('preprocessed_data.pt', weights_only=False)
    
    train_data = data['train_data']
    val_data = data['val_data']
    tokenizer = data['tokenizer']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("\nStarting DDP training...")
    
    main(train_data, val_data, tokenizer)
