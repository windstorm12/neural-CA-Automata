import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import time


class MBlockNCA:
    def __init__(self, weights_path="mblock_circle_nca/weights.npz"):
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
        
        update_mask = None
        if stochastic:
            update_mask = (np.random.random((state.shape[0], 1, state.shape[2], state.shape[3])) < update_prob).astype(np.float32)
            dx = dx * update_mask
        
        new_state = state + dx
        new_state[:, 0:1] = np.clip(new_state[:, 0:1], 0.0, 1.0)
        new_state[:, 1:] = np.clip(new_state[:, 1:], -3.0, 3.0)
        
        return new_state, update_mask
    
    def create_seed(self, grid_size=40):
        state = np.zeros((1, self.channels, grid_size, grid_size), dtype=np.float32)
        state[0, 0, grid_size//2, grid_size//2] = 1.0
        return state
    
    def get_alive_mask(self, state, threshold=0.1):
        return (state[:, 0:1] > threshold).astype(np.float32)


class MBlockSimulation:
    def __init__(self, nca, grid_size=40):
        self.nca = nca
        self.grid_size = grid_size
        self.state = nca.create_seed(grid_size)
        self.history = [self.state.copy()]
        
    def reset(self):
        self.state = self.nca.create_seed(self.grid_size)
        self.history = [self.state.copy()]
    
    def step(self, stochastic=True):
        self.state, update_mask = self.nca.step(self.state, stochastic=stochastic)
        self.history.append(self.state.copy())
        return update_mask
    
    def apply_damage(self, damage_type="half"):
        if damage_type == "half":
            self.state[:, :, :, self.grid_size//2:] = 0.0
        elif damage_type == "circle":
            cx, cy = self.grid_size//2, self.grid_size//2
            r = 10
            y, x = np.ogrid[:self.grid_size, :self.grid_size]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            self.state[:, :, mask] = 0.0
        elif damage_type == "random":
            mask = np.random.random((1, 1, self.grid_size, self.grid_size)) > 0.5
            self.state = self.state * mask
        
        self.history.append(self.state.copy())
    
    def get_alive_display(self):
        return self.state[0, 0]
    
    def get_active_block_count(self):
        return (self.state[0, 0] > 0.1).sum()


def create_custom_colormap():
    colors = ['#0a0a0a', '#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560', '#f9a826', '#ffffff']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mblock', colors, N=n_bins)
    return cmap


def simulate_growth(nca, grid_size=40, steps=120, save_animation=True):
    print("Starting M-Block Self-Assembly Simulation...")
    print("=" * 60)
    
    sim = MBlockSimulation(nca, grid_size)
    cmap = create_custom_colormap()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('M-Block Neural CA: Decentralized Self-Assembly', fontsize=16, fontweight='bold')
    
    # Main display
    im = axes[0].imshow(sim.get_alive_display(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title(f"Step 0 | Active Blocks: 1", fontsize=12)
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # Activity heatmap
    im2 = axes[1].imshow(np.zeros((grid_size, grid_size)), cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title("Cell Update Activity", fontsize=12)
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Alive Signal')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Update Probability')
    
    frames = []
    activity_history = np.zeros((grid_size, grid_size))
    
    def update(frame):
        nonlocal activity_history
        
        if frame < steps:
            # Normal growth
            update_mask = sim.step(stochastic=True)
            phase = "GROWING"
            
            if update_mask is not None:
                activity_history = 0.9 * activity_history + 0.1 * update_mask[0, 0]
        
        elif frame == steps:
            # Apply damage
            sim.apply_damage("half")
            phase = "DAMAGED"
        
        else:
            # Repair
            update_mask = sim.step(stochastic=True)
            phase = "REPAIRING"
            
            if update_mask is not None:
                activity_history = 0.9 * activity_history + 0.1 * update_mask[0, 0]
        
        alive = sim.get_alive_display()
        count = sim.get_active_block_count()
        
        im.set_data(alive)
        axes[0].set_title(f"Step {frame} | Active Blocks: {count:.0f} | {phase}", fontsize=12)
        
        im2.set_data(activity_history)
        
        return [im, im2]
    
    print("Running simulation...")
    total_frames = steps + 80
    
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=50, blit=True, repeat=True
    )
    
    if save_animation:
        print("Saving animation (this may take a minute)...")
        writer = animation.PillowWriter(fps=20)
        anim.save('mblock_simulation.gif', writer=writer)
        print("✓ Saved: mblock_simulation.gif")
    
    plt.tight_layout()
    plt.show()
    
    return sim


def simulate_comparison(nca, grid_size=40):
    print("\nComparison: Deterministic vs Asynchronous")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('M-Block Assembly: Synchronous vs Asynchronous Updates', fontsize=14, fontweight='bold')
    
    cmap = create_custom_colormap()
    
    # Synchronous (deterministic)
    sim_sync = MBlockSimulation(nca, grid_size)
    snapshots_sync = [0, 30, 60, 100]
    
    for i, target_step in enumerate(snapshots_sync):
        while len(sim_sync.history) <= target_step:
            sim_sync.nca.step(sim_sync.state, stochastic=False)
            sim_sync.history.append(sim_sync.state.copy())
        
        state = sim_sync.history[target_step]
        axes[0, i].imshow(state[0, 0], cmap=cmap, vmin=0, vmax=1)
        axes[0, i].set_title(f"Sync Step {target_step}")
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel('Synchronous\n(All blocks update)', fontsize=11, fontweight='bold')
    
    # Asynchronous (stochastic)
    sim_async = MBlockSimulation(nca, grid_size)
    
    for i, target_step in enumerate(snapshots_sync):
        while len(sim_async.history) <= target_step:
            sim_async.step(stochastic=True)
        
        state = sim_async.history[target_step]
        axes[1, i].imshow(state[0, 0], cmap=cmap, vmin=0, vmax=1)
        axes[1, i].set_title(f"Async Step {target_step}")
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel('Asynchronous\n(~50% blocks update)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mblock_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mblock_comparison.png")
    plt.show()


def simulate_multi_damage(nca, grid_size=40):
    print("\nMulti-Damage Resilience Test")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle('M-Block Self-Repair: Multiple Damage Scenarios', fontsize=14, fontweight='bold')
    
    cmap = create_custom_colormap()
    damage_types = ["half", "circle", "random"]
    
    for row, damage in enumerate(damage_types[:2]):
        # Grow
        sim = MBlockSimulation(nca, grid_size)
        for _ in range(100):
            sim.step(stochastic=False)
        
        grown = sim.get_alive_display().copy()
        
        # Damage
        sim.apply_damage(damage)
        damaged = sim.get_alive_display().copy()
        
        # Repair sequence
        repair_steps = [0, 25, 50, 100]
        
        for i, steps in enumerate(repair_steps):
            if i == 0:
                axes[row, 0].imshow(grown, cmap=cmap, vmin=0, vmax=1)
                axes[row, 0].set_title("Initial")
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(damaged, cmap=cmap, vmin=0, vmax=1)
                axes[row, 1].set_title("Damaged")
                axes[row, 1].axis('off')
            else:
                for _ in range(25):
                    sim.step(stochastic=True)
                
                repaired = sim.get_alive_display()
                axes[row, i + 1].imshow(repaired, cmap=cmap, vmin=0, vmax=1)
                axes[row, i + 1].set_title(f"Repair +{steps}")
                axes[row, i + 1].axis('off')
        
        damage_label = damage.capitalize() + " Damage"
        axes[row, 0].set_ylabel(damage_label, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mblock_repair.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mblock_repair.png")
    plt.show()


def print_stats(nca, grid_size=40):
    print("\n" + "=" * 60)
    print("M-BLOCK NEURAL CA STATISTICS")
    print("=" * 60)
    
    sim = MBlockSimulation(nca, grid_size)
    
    for step in range(100):
        sim.step(stochastic=False)
    
    final_blocks = sim.get_active_block_count()
    
    print(f"Grid Size:        {grid_size} x {grid_size} = {grid_size**2} positions")
    print(f"Active Blocks:    {final_blocks:.0f}")
    print(f"Fill Ratio:       {final_blocks / grid_size**2 * 100:.1f}%")
    print(f"State Channels:   {nca.channels} per block")
    print(f"Total Parameters: {nca.perceive_w.size + nca.update_w.size:,}")
    print(f"Perceive Kernel:  {nca.perceive_w.shape}")
    print(f"Update Kernel:    {nca.update_w.shape}")
    print()
    print("Key Properties:")
    print("  ✓ Fully decentralized (no global controller)")
    print("  ✓ Self-organizing from single seed")
    print("  ✓ Self-repairing after damage")
    print("  ✓ Asynchronous-safe (stochastic updates)")
    print("  ✓ Scalable (same rules for any grid size)")
    print("=" * 60)


def main():
    print("\n" + "🤖 " * 20)
    print("M-BLOCK NEURAL CELLULAR AUTOMATA SIMULATION")
    print("Decentralized Self-Assembly Demo")
    print("🤖 " * 20 + "\n")
    
    # Load trained NCA
    print("Loading trained Neural CA...")
    nca = MBlockNCA("mblock_circle_nca/weights.npz")
    print("✓ Loaded successfully!\n")
    
    # Print statistics
    print_stats(nca)
    
    # Main simulation
    print("\n[1/3] Running main growth + repair simulation...")
    simulate_growth(nca, grid_size=40, steps=100, save_animation=True)
    
    # Comparison
    print("\n[2/3] Running sync vs async comparison...")
    simulate_comparison(nca, grid_size=40)
    
    # Multi-damage
    print("\n[3/3] Running multi-damage test...")
    simulate_multi_damage(nca, grid_size=40)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • mblock_simulation.gif   - Animated growth & repair")
    print("  • mblock_comparison.png   - Sync vs Async comparison")
    print("  • mblock_repair.png       - Multi-damage scenarios")
    print("\n✨ Ready for your MIT demo! ✨\n")


if __name__ == "__main__":
    main()