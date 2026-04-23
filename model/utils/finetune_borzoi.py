import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from borzoi_pytorch import Borzoi
import pysam
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py

import random
import gc
from Bio import SeqIO
import math 
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)          
torch.cuda.manual_seed(seed)      
torch.cuda.manual_seed_all(seed)

torch.set_float32_matmul_precision('high')
num_epochs = 4
max_lr = 3e-3
min_lr = max_lr * 0.1
warmup_steps = 1000

def get_lr(it, warmup_steps, max_steps):
    # 1) Linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2) If it >= max_steps, return min_lr
    if it >= max_steps:
        return min_lr

    # 3) Cosine decay between warmup and max_steps
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) 


def configure_optimizers(model, weight_decay, learning_rate, device):
    # Start with all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optimizer groups
    # Any parameter that is 2D (e.g., weights in linear layers) will have weight decay
    # All others (biases, layernorms) will not
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Use fused AdamW if available and running on CUDA
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"using fused AdamW: {use_fused}")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=use_fused if fused_available else False
    )

    return optimizer



class BorzoiFineTune(nn.Module):
    """Simple wrapper to finetune Borzoi on new tracks"""
    def __init__(self, pretrained_model, num_new_tracks=10):
        super().__init__()
        self.borzoi = pretrained_model
        
        del self.borzoi.human_head
        
        del self.borzoi.final_softplus
            
        
        self.new_head = nn.Conv1d(1920, num_new_tracks, kernel_size=1)
        self.final_activation = nn.Softplus() 
        
    def forward(self, sequence):
        """Forward through Borzoi backbone + new head"""
        # Run through Borzoi up to final_joined_convs
        x = self.borzoi.conv_dna(sequence)
        x_unet0 = self.borzoi.res_tower(x)
        x_unet1 = self.borzoi.unet1(x_unet0)
        x = self.borzoi._max_pool(x_unet1)
        x_unet1 = self.borzoi.horizontal_conv1(x_unet1)
        x_unet0 = self.borzoi.horizontal_conv0(x_unet0)
        x = self.borzoi.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.borzoi.upsampling_unet1(x)
        x += x_unet1
        x = self.borzoi.separable1(x)
        x = self.borzoi.upsampling_unet0(x)
        x += x_unet0
        x = self.borzoi.separable0(x)
        x = self.borzoi.crop(x.permute(0, 2, 1))
        x = self.borzoi.final_joined_convs(x.permute(0, 2, 1))
        
        # Apply new head
        x = self.new_head(x)
        x = self.final_activation(x)  
        return x




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="/Users/arif/Downloads/Borzoi_enformer_ft/enformer_train_data/h5py/all_combined.h5", help='path to train data')
    parser.add_argument('--genome', default="/Users/arif/Downloads/Borzoi_enformer_ft/resources/genome/hg38.ml.fa", help='Genome FASTA')
    parser.add_argument('--bed_file', default="/Users/arif/Downloads/Borzoi_enformer_ft/data/overlap_with_borzoi.bed", help='Targets file')
    
    args = parser.parse_args()
    device = "cuda"
    print(f'Using device: {device}')
    genome = args.genome
    genome_dict = SeqIO.to_dict(SeqIO.parse(genome, "fasta"))

    combined = args.data
    bed_file = args.bed_file
    
    # Create datasets
    print("Loading training data...")
    train_dataset = EnformerDataset(combined, bed_file, seqlen=524288, genome_dict=genome_dict,
                             shift_aug=True, rc_aug=True, fold="train")   

    val_dataset = EnformerDataset(combined, bed_file, seqlen=524288, genome_dict=genome_dict,
                             shift_aug=False, rc_aug=False, fold="valid")   
    # Create data loaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    print(f"Total training batches: {len(train_loader)} with batch size {batch_size}")

    # Gradient accumulation settings
    desired_batch_size = 64
    accumulation_steps = desired_batch_size // batch_size 
    print(f"Using gradient accumulation with {accumulation_steps} steps (effective batch size: {desired_batch_size})")
    
    # Training loop
    max_steps = ((len(train_loader))// accumulation_steps) * num_epochs
    total_steps = len(train_loader) // accumulation_steps * num_epochs
    print(f"Starting training for {num_epochs} epochs...")
    print("Loading pretrained Enformer model...")
    pretrained = Borzoi.from_pretrained("johahi/borzoi-replicate-3")
    print("Loaded pretrained Borzoi model")
    num_new_tracks = 10
    model = BorzoiFineTune(pretrained, num_new_tracks)
    print(f"Created model with {num_new_tracks} output tracks")
    
    model = model.to(device)
   
    model.train()
    model = torch.compile(model)
    
    
    model_name = "finetune_borzoi_like_8_epochs_max_lr_e_3_rep3"
    
    #wandb.init(project=model_name)
    wandb.init(
    project=f"Borzoi_all_reps2",
    name=f"rep3",
    tags=[f"rep3"]
    )
    
    loss_accum = 0.0
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device=device)
    
    step = 0 
    stepi = []
    lossi = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []  # Store loss per optimization step, not per batch
        
        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader)):
    
            
            # Zero gradients at start of accumulation step
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
                loss_accum = 0.0
            
            seq = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Forward pass
                 pred = model(seq.permute(0,2,1))
                 loss = poisson_loss(pred.permute(0,2,1), target)
            
            # Scale loss for gradient accumulation 
            loss = loss / accumulation_steps
            
            
            loss_accum += loss.detach()
            #print(loss_accum)      
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation - update every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
                
                # Update learning rate per step (not per epoch)
                lr = get_lr(step, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Optimizer step
                optimizer.step()
                
                # Store loss for this optimization step (unscaled)
                epoch_losses.append(loss_accum.item())
                
                # Print progress every accumulation step
                print(f"Step {step}, Loss: {loss_accum.item():.4f}, LR: {lr:.4e}, Norm: {norm:.4f}")
                
                
                step += 1  # Increment global step counter
                stepi.append(step)
                lossi.append(loss_accum.item())
                #wandb.log({"train_loss": loss_accum.item(), "step": step})
               
                wandb.log({"train_loss": loss_accum.item(), "norm": norm, "lr": lr}, step=step)
                
    
        
        #Handle remaining gradients at end of epoch
        if len(train_loader) % accumulation_steps != 0:
            # Gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update learning rate
            lr =get_lr(step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            
            # Add the final partial step loss
            epoch_losses.append(loss_accum.item())
            step += 1
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                seq = batch['sequence'].to(device)
                target = batch['target'].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pred = model(seq.permute(0, 2, 1))
                    loss = poisson_loss(pred.permute(0, 2, 1), target)
                val_losses.append(loss.item())
        
                    
                    
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(val_losses)
        # At the end of each epoch, after computing avg_val_loss:
        
        wandb.log({
        "epoch_train_loss": avg_train_loss,
        "epoch_val_loss": avg_val_loss,
        "epoch": epoch + 1
        }, step=step)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Steps completed: {step}")
        print("-" * 50)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Save both the model weights AND useful metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'step': step,
            }
            torch.save(checkpoint, f"{model_name}_best.pth")
            print(f"  ✓ New best model saved! Val loss: {best_val_loss:.4f}")
        else:
            print(f"  (No improvement. Best val loss so far: {best_val_loss:.4f})")
    
    
    
    print("Saving fine-tuned model...")
    torch.save(model.state_dict(), f"{model_name}.pth")
    
        

if __name__ == "__main__":
    main()

    
    
    
    
