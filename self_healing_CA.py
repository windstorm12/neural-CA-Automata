import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json
import os


class NeuralCA(nn.Module):
    def __init__(self, channels=16, hidden_channels=128):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels

        self.perceive = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

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

            x = x + dx
            alive_channel = torch.clamp(x[:, 0:1], 0.0, 1.0)
            hidden_channels = torch.clamp(x[:, 1:], -3.0, 3.0)
            x = torch.cat([alive_channel, hidden_channels], dim=1)

        return x

    def step_single(self, x):
        perception = self.perceive(x)
        dx = self.update(perception)
        x = x + dx
        alive_channel = torch.clamp(x[:, 0:1], 0.0, 1.0)
        hidden_channels = torch.clamp(x[:, 1:], -3.0, 3.0)
        return torch.cat([alive_channel, hidden_channels], dim=1)


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


class CATrainer:
    def __init__(self, model, target, device='cuda'):
        self.model = model.to(device)
        self.target = target.to(device)
        self.device = device
        self.size = target.shape[-1]

        self.pool = torch.zeros(1024, model.channels, self.size, self.size).to(device)
        for i in range(1024):
            self.pool[i:i+1] = create_seed(self.size, model.channels)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2000, gamma=0.3)

    def train(self, epochs=2000):
        print(f"Training on {self.device}...")

        for epoch in range(epochs):
            idx = np.random.choice(1024, 8, replace=False)
            batch = self.pool[idx].clone()

            if epoch > 200:
                mask = torch.ones_like(batch)
                cx, cy = np.random.randint(10, 30), np.random.randint(10, 30)
                r = 8
                y, x = np.ogrid[:40, :40]
                mask_area = (x - cx)**2 + (y - cy)**2 >= r**2
                mask_area = torch.from_numpy(mask_area).to(self.device)
                batch = batch * mask_area.float()

            result = self.model(batch, steps=np.random.randint(64, 96))
            loss = F.mse_loss(result[:, 0:1], self.target.repeat(8, 1, 1, 1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                self.pool[idx] = result.detach()
                if epoch % 10 == 0:
                    self.pool[idx[:1]] = create_seed(40, 16).to(self.device)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        return self.model


def export_for_simulation(model, export_path="mblock_circle_nca"):
    os.makedirs(export_path, exist_ok=True)
    model.eval()
    model_cpu = model.cpu()

    # PyTorch checkpoint
    torch.save({
        'state_dict': model_cpu.state_dict(),
        'channels': model_cpu.channels,
        'hidden_channels': model_cpu.hidden_channels,
    }, f"{export_path}/model.pt")
    print(f"Saved: {export_path}/model.pt")

    # NumPy weights
    perceive_w = model_cpu.perceive.weight.detach().numpy()
    update_w = model_cpu.update[1].weight.detach().numpy()

    np.savez_compressed(
        f"{export_path}/weights.npz",
        perceive_weight=perceive_w,
        update_weight=update_w
    )
    print(f"Saved: {export_path}/weights.npz")

    # Config JSON
    config = {
        "name": "NeuralCA_Circle",
        "description": "Self-organizing circle pattern for M-Block simulation",
        "channels": model_cpu.channels,
        "hidden_channels": model_cpu.hidden_channels,
        "kernel_size": 3,
        "update_probability": 0.5,
        "state_clamp": {
            "channel_0": [0.0, 1.0],
            "channels_1_plus": [-3.0, 3.0]
        },
        "architecture": [
            "Conv2D(channels=16 -> 128, kernel=3x3, no bias)",
            "ReLU",
            "Conv2D(channels=128 -> 16, kernel=1x1, no bias)",
            "Residual add to input",
            "Clamp channel 0 to [0,1], others to [-3,3]"
        ],
        "usage": {
            "each_block_state": "16-dimensional vector",
            "channel_0": "alive/presence indicator (0-1)",
            "channels_1-15": "hidden communication state",
            "seed": "single block with state[0]=1, rest=0"
        }
    }

    with open(f"{export_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {export_path}/config.json")

    # Standalone inference code
    standalone_code = '''"""
M-Block Neural CA Inference - Pure NumPy
"""

import numpy as np


class MBlockNCA:
    def __init__(self, weights_path="weights.npz"):
        data = np.load(weights_path)
        self.perceive_w = data["perceive_weight"]
        self.update_w = data["update_weight"]
        self.channels = 16
        self.hidden = 128

    def conv3x3(self, x, weight):
        B, C_in, H, W = x.shape
        C_out = weight.shape[0]
        x_pad = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        out = np.zeros((B, C_out, H, W))
        for co in range(C_out):
            for ci in range(C_in):
                for ky in range(3):
                    for kx in range(3):
                        out[:, co] += x_pad[:, ci, ky:ky+H, kx:kx+W] * weight[co, ci, ky, kx]
        return out

    def conv1x1(self, x, weight):
        B, C_in, H, W = x.shape
        C_out = weight.shape[0]
        w = weight[:, :, 0, 0]
        x_flat = x.reshape(B, C_in, -1)
        out_flat = np.einsum('oi,bih->boh', w, x_flat)
        return out_flat.reshape(B, C_out, H, W)

    def step(self, state, stochastic=False, update_prob=0.5):
        perception = self.conv3x3(state, self.perceive_w)
        perception = np.maximum(perception, 0)
        dx = self.conv1x1(perception, self.update_w)

        if stochastic:
            mask = (np.random.random((state.shape[0], 1, state.shape[2], state.shape[3])) < update_prob).astype(np.float32)
            dx = dx * mask

        new_state = state + dx
        new_state[:, 0:1] = np.clip(new_state[:, 0:1], 0.0, 1.0)
        new_state[:, 1:] = np.clip(new_state[:, 1:], -3.0, 3.0)
        return new_state

    def run(self, initial_state, steps=100, stochastic=False):
        state = initial_state.copy()
        for _ in range(steps):
            state = self.step(state, stochastic=stochastic)
        return state

    def create_seed(self, grid_size=40):
        state = np.zeros((1, self.channels, grid_size, grid_size), dtype=np.float32)
        state[0, 0, grid_size//2, grid_size//2] = 1.0
        return state

    def get_alive_mask(self, state, threshold=0.1):
        return (state[:, 0:1] > threshold).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nca = MBlockNCA("weights.npz")
    state = nca.create_seed(grid_size=40)
    print(f"Initial state shape: {state.shape}")
    print(f"Initial active blocks: {(state[0,0] > 0.1).sum()}")

    final_state = nca.run(state, steps=100)
    alive = nca.get_alive_mask(final_state)
    print(f"Final active blocks: {(alive > 0.5).sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(state[0, 0], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Initial (Seed)")

    axes[1].imshow(final_state[0, 0], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Grown (100 steps)")

    damaged = final_state.copy()
    damaged[:, :, :, 20:] = 0
    repaired = nca.run(damaged, steps=100)

    axes[2].imshow(repaired[0, 0], cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title("Self-Repaired")

    plt.tight_layout()
    plt.savefig("nca_demo.png", dpi=150)
    plt.show()
    print("Demo complete! See nca_demo.png")
'''

    with open(f"{export_path}/mblock_nca.py", "w") as f:
        f.write(standalone_code)
    print(f"Saved: {export_path}/mblock_nca.py")

    # README
    readme = """# Neural CA for M-Block Self-Assembly

A trained Neural Cellular Automaton that enables decentralized self-organization into a circle pattern.

Each M-Block runs the exact same neural network using only local neighbor information.
No central controller. No global communication. Just local rules producing global shape.

## Files

- model.pt: PyTorch checkpoint
- weights.npz: NumPy weights (universal)
- config.json: Architecture and hyperparameters
- mblock_nca.py: Standalone inference code (pure NumPy)

## Quick Start

    from mblock_nca import MBlockNCA

    nca = MBlockNCA("weights.npz")
    state = nca.create_seed(grid_size=40)
    final = nca.run(state, steps=100)
    alive_mask = nca.get_alive_mask(final)

## How It Works

Each block maintains a 16-dimensional state vector:
- Channel 0: "Alive" signal (0-1) determines if block should be present
- Channels 1-15: Hidden state for communication with neighbors

Each simulation step:
1. Each block gathers neighbor states (3x3 neighborhood)
2. Applies perception convolution (3x3)
3. Applies ReLU activation
4. Applies update convolution (1x1)
5. Adds result to current state (residual)
6. Clamps values

## Key Properties

- Decentralized: No global controller
- Self-healing: Recovers from damage
- Scalable: Same rules work for any grid size
- Robust: Stochastic updates make it asynchronous-safe
"""

    with open(f"{export_path}/README.md", "w") as f:
        f.write(readme)
    print(f"Saved: {export_path}/README.md")

    print(f"\nExport complete! Files in: {export_path}/")
    return export_path


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = create_simple_shape(40)
    model = NeuralCA(16)
    trainer = CATrainer(model, target, device=DEVICE)

    model = trainer.train(epochs=2000)

    export_path = export_for_simulation(model, "mblock_circle_nca")

    print("\nVerification test...")
    model.eval()
    with torch.no_grad():
        seed = create_seed(40, 16).to(DEVICE)
        grown = model(seed.to(DEVICE), steps=100)

        damaged = grown.clone()
        damaged[:, :, :, 20:] = 0.0
        healed = model(damaged, steps=100)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(grown[0,0].cpu(), cmap='viridis')
        ax[0].set_title("Grown")
        ax[1].imshow(damaged[0,0].cpu(), cmap='viridis')
        ax[1].set_title("Damaged")
        ax[2].imshow(healed[0,0].cpu(), cmap='viridis')
        ax[2].set_title("Healed")
        plt.savefig(f"{export_path}/verification.png", dpi=150)
        plt.show()
        print(f"Saved: {export_path}/verification.png")


if __name__ == "__main__":
    main()