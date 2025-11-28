# Neural Cellular Automata for Self-Assembly

**Decentralized self-organization. 0.003 MSE loss. Self-repairs from damage.**

![Demo](mblock_simulation.gif)

## What This Is

A Neural CA system where identical local rules produce global structure. Each cell only sees 8 neighbors. No central controller. Yet they self-assemble and self-repair.

Trained in 5 minutes. Runs in pure NumPy. 20KB model.

## Quick Demo

    git clone https://github.com/windstorm12/neural-automata-lab.git
    python demo.py

No GPU needed. Runs anywhere.

## Results

- Loss: 0.003 MSE (near perfect)
- Self-repair: 100% recovery from damage
- Model size: ~20KB (fits on Arduino)
- Training time: 5 min on RTX 3060

## Why This Matters

This solves core challenges in:
- Modular robotics (MIT M-Blocks, swarm robots)
- Programmable matter (self-assembling materials)
- Distributed systems (fault-tolerant networks)

Each unit runs the same small neural network. Local decisions → global intelligence.

## The Experiments

### 1. Self-Healing Neural CA (demo.py)
Trained to grow specific patterns from a single seed and maintain homeostasis.

- The "Wolverine" Protocol: Model is trained under massive damage (portions zeroed out)
- Architecture: CNN with stochastic updates and residual connections
- Stability: Gradient clipping and value clamping prevent state explosion

### 2. Programmable Matter Simulation (smart_matter.py)
Agent-based simulation of autonomous robotic cubes that self-organize.

- Distributed Logic: No central controller
- Perception: Limited 3x3 neighborhood view
- Physics: Gravity, collision detection, magnetic linking

## Files

- demo.py - Quick demo (growth and repair)
- full_demo.py - Full visualization suite  
- self_healing_CA.py - Training code
- smart_matter.py - Interactive Pygame simulation
- mblock_circle_nca/ - Trained weights

## Tech Stack

- Deep Learning: PyTorch (Conv2d, Automatic Differentiation)
- Simulation: Pygame (Real-time agent rendering)
- Visualization: Matplotlib, PIL
- Math: NumPy

## License

GNU General Public License v3.0 (GPLv3)
