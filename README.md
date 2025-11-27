# Neural Automata Lab

A research framework exploring Differentiable Self-Organization and Programmable Matter using PyTorch. This repository bridges the gap between Deep Learning and Complex Systems, demonstrating how local neural policies can give rise to global emergent structures.

## The Experiments

### 1. Neural Morphogenesis (growing_ca.py)
A implementation of Neural Cellular Automata (NCA) trained to grow specific patterns and maintain homeostasis. This model treats a grid of pixels as a colony of independent cells, each running the same Neural Network.

* **The "Wolverine" Protocol:** The model is trained for robustness. During the learning process, the canvas is subjected to massive damage (portions of the grid are zeroed out). The network minimizes loss by learning to regenerate lost tissue based on local neighbor communication.
* **Architecture:** A Convolutional Neural Network (CNN) with stochastic updates and residual connections.
* **Stability:** Implements gradient clipping and value clamping to prevent state explosion, allowing for stable long-term growth.

### 2. Programmable Matter (smart_matter.py)
An agent-based simulation of "M-Blocks"—autonomous robotic cubes that organize themselves without central control.

* **Distributed Logic:** There is no central server telling the blocks where to move. Each block moves based solely on a local policy network.
* **Perception:** Agents have a limited field of view (3x3 Moore neighborhood) and a global target vector.
* **Physics:** The simulation implements gravity, collision detection, and magnetic neighbor linking.

## How to Run

1. Install Dependencies:

    pip install -r requirements.txt

2. Run the Self-Healing Demo:
    
    python growing_ca.py

    (Watch the terminal for training progress. A visualization window will appear showing the Growth -> Damage -> Regeneration cycle.)

3. Run the Swarm Simulation:

    python smart_matter.py

    (A Pygame window will open. Click anywhere to spawn agents. Press 'R' to reset the simulation.)

## Tech Stack

* **Deep Learning:** PyTorch (Conv2d, Automatic Differentiation)
* **Simulation:** Pygame (Real-time agent rendering)
* **Visualization:** Matplotlib, PIL
* **Math:** NumPy

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). This ensures that this research remains open and free for the academic community.
