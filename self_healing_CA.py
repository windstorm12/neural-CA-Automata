import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# ═══════════════════════════════════════════════════════════════
# 1. THE "HIGH VOLTAGE" MODEL
# ═══════════════════════════════════════════════════════════════
class NeuralCA(nn.Module):
    def __init__(self, channels=16, hidden_channels=128):
        super().__init__()
        self.channels = channels
        
        self.perceive = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        
        # Noise init
        with torch.no_grad():
            self.perceive.weight.data.normal_(0.0, 0.02)
            self.update[1].weight.data.normal_(0.0, 0.01)
        
        self.update_probability = 0.5
        
    def forward(self, x, steps=64):
        for _ in range(steps):
            perception = self.perceive(x)
            dx = self.update(perception)
            
            if self.training:
                mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < self.update_probability).float()
                dx = dx * mask
            
            # Update state
            x = x + dx
            
            # --- THE FIX: Out-of-place operations ---
            # Split, Clamp, Concat (Creates a NEW tensor, preserving history)
            alive_channel = torch.clamp(x[:, 0:1], 0.0, 1.0)
            hidden_channels = torch.clamp(x[:, 1:], -3.0, 3.0)
            
            x = torch.cat([alive_channel, hidden_channels], dim=1)
            
        return x
# ═══════════════════════════════════════════════════════════════
# 2. UTILS
# ═══════════════════════════════════════════════════════════════
def create_simple_shape(size=40):
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([5, 5, size-5, size-5], fill=255)
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

def create_seed(size, channels):
    seed = torch.zeros(1, channels, size, size)
    seed[0, 0, size//2, size//2] = 1.0
    return seed

# ═══════════════════════════════════════════════════════════════
# 3. THE TRAINER
# ═══════════════════════════════════════════════════════════════
class CATrainer:
    def __init__(self, model, target, device='cuda'):
        self.model = model.to(device)
        self.target = target.to(device)
        self.device = device
        self.size = target.shape[-1]
        
        # Simple Pool
        self.pool = torch.zeros(1024, model.channels, self.size, self.size).to(device)
        for i in range(1024): self.pool[i:i+1] = create_seed(self.size, model.channels)
            
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2000, gamma=0.3)
        
    def train(self, epochs=2000):
        print(f"Training on {self.device}...")
        
        for epoch in range(epochs):
            # Sample
            idx = np.random.choice(1024, 8, replace=False)
            batch = self.pool[idx].clone()
            
            # DAMAGE (Wolverine)
            if epoch > 200:
                mask = torch.ones_like(batch)
                cx, cy = np.random.randint(10, 30), np.random.randint(10, 30)
                r = 8
                y, x = np.ogrid[:40, :40]
                mask_area = (x - cx)**2 + (y - cy)**2 >= r**2
                mask_area = torch.from_numpy(mask_area).to(self.device)
                batch = batch * mask_area.float()
            
            # Run
            result = self.model(batch, steps=np.random.randint(64, 96))
            
            # Loss
            loss = F.mse_loss(result[:, 0:1], self.target.repeat(8, 1, 1, 1))
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update Pool
            with torch.no_grad():
                self.pool[idx] = result.detach()
                # Reseed
                if epoch % 10 == 0:
                    idx_reseed = idx[:1] # Replace 1 sample per batch with fresh seed
                    self.pool[idx_reseed] = create_seed(40, 16).to(self.device)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        return self.model

# ═══════════════════════════════════════════════════════════════
# 4. RUN IT
# ═══════════════════════════════════════════════════════════════
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = create_simple_shape(40)
    model = NeuralCA(16)
    trainer = CATrainer(model, target, device=DEVICE)
    
    # TRAIN
    model = trainer.train(epochs=2000)
    
    # TEST
    print("TESTING...")
    model.eval()
    with torch.no_grad():
        seed = create_seed(40, 16).to(DEVICE)
        
        # Grow
        grown = model(seed, steps=100)
        
        # Damage
        damaged = grown.clone()
        damaged[:, :, :, 20:] = 0.0
        
        # Heal
        healed = model(damaged.clone(), steps=100)
        
        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(grown[0,0].cpu(), cmap='viridis'); ax[0].set_title("Grown")
        ax[1].imshow(damaged[0,0].cpu(), cmap='viridis'); ax[1].set_title("Damaged")
        ax[2].imshow(healed[0,0].cpu(), cmap='viridis'); ax[2].set_title("Healed")
        plt.show()

if __name__ == "__main__":
    main()