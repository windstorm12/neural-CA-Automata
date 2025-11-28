"""
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
