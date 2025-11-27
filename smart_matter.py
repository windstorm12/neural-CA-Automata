import pygame
import torch
import torch.nn as nn
import numpy as np
import random
import math

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
GRID_W = WIDTH // CELL_SIZE
GRID_H = HEIGHT // CELL_SIZE
FPS = 30

# Colors
BG_COLOR = (15, 15, 20)
GRID_COLOR = (30, 30, 40)
BLOCK_COLOR = (0, 220, 150)  # MIT Cyber Green
LINK_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 50, 50)

# ═══════════════════════════════════════════════════════════════
# 2. THE NEURAL BRAIN (1x1 Conv Logic / Local Brain)
# ═══════════════════════════════════════════════════════════════
class SwarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: 
        # 8 Neighbors (Binary) + 
        # 2 Target Vector (dx, dy) + 
        # 1 Gravity Sense (Is ground below?)
        # = 11 Inputs
        self.fc1 = nn.Linear(11, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 5) # Stay, Left, Right, Climb_L, Climb_R
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# ═══════════════════════════════════════════════════════════════
# 3. THE AGENT (M-BLOCK)
# ═══════════════════════════════════════════════════════════════
class MBlock:
    def __init__(self, x, y, brain):
        self.x = x
        self.y = y
        self.brain = brain
        self.id = random.randint(0, 99999)
        
    def get_state(self, grid, target):
        """
        This is the 'Perception' step (Convolution equivalent).
        It looks at the 3x3 neighborhood.
        """
        state = []
        
        # 1. Check 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                
                nx, ny = self.x + dx, self.y + dy
                # Walls count as 'blocks' for stability
                if nx < 0 or nx >= GRID_W or ny >= GRID_H:
                    state.append(1.0)
                elif ny < 0:
                    state.append(0.0)
                else:
                    state.append(1.0 if grid[nx][ny] != 0 else 0.0)
                    
        # 2. Target Vector (Where is the goal?)
        tx, ty = target
        vec_x = (tx - self.x) / GRID_W
        vec_y = (ty - self.y) / GRID_H
        state.append(vec_x)
        state.append(vec_y)
        
        # 3. Ground Sensor
        ground_below = 1.0 if (self.y + 1 >= GRID_H or grid[self.x][min(GRID_H-1, self.y+1)] != 0) else 0.0
        state.append(ground_below)
        
        return torch.tensor(state).float()

    def decide(self, grid, target):
        # Run Neural Network
        state = self.get_state(grid, target)
        with torch.no_grad():
            probs = self.brain(state)
            
        # Stochastic Policy (Sample from distribution)
        # This mimics biological noise/robustness
        action = torch.multinomial(probs, 1).item()
        return action

# ═══════════════════════════════════════════════════════════════
# 4. THE SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural CA Programmable Matter Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Brain initialization (Pre-seeded for demo behavior)
    # We nudge weights slightly so untrained agents seek the center
    brain = SwarmNet()
    with torch.no_grad():
        brain.fc1.weight.normal_(0.0, 0.5)
    
    # World Data
    grid = np.zeros((GRID_W, GRID_H), dtype=int)
    blocks = []
    target_pos = (GRID_W // 2, GRID_H // 2)

    # Helper to move blocks
    def try_move(b, dx, dy):
        nx, ny = b.x + dx, b.y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            if grid[nx][ny] == 0:
                grid[b.x][b.y] = 0
                b.x, b.y = nx, ny
                grid[b.x][b.y] = b.id
                return True
        return False

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    b = MBlock(gx, gy, brain)
                    blocks.append(b)
                    grid[gx][gy] = b.id
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    grid = np.zeros((GRID_W, GRID_H), dtype=int)
                    blocks = []

        # 2. Update Loop
        random.shuffle(blocks) # Async update simulation
        
        for b in blocks:
            # A. Gravity
            if b.y < GRID_H - 1 and grid[b.x][b.y+1] == 0:
                try_move(b, 0, 1)
                continue # If falling, don't think
            
            # B. Neural Decision
            action = b.decide(grid, target_pos)
            
            # Actions: 0:Stay, 1:Left, 2:Right, 3:ClimbL, 4:ClimbR
            if action == 1: 
                if b.y < GRID_H-1 and grid[b.x][b.y+1] != 0: # Must be on ground to walk
                    try_move(b, -1, 0)
            elif action == 2: 
                if b.y < GRID_H-1 and grid[b.x][b.y+1] != 0:
                    try_move(b, 1, 0)
            elif action == 3: # Climb Left
                # Check support
                if b.x > 0 and grid[b.x-1][b.y] != 0: # Wall to left
                    if grid[b.x-1][b.y-1] == 0: # Space above wall empty
                        try_move(b, -1, -1)
            elif action == 4: # Climb Right
                if b.x < GRID_W-1 and grid[b.x+1][b.y] != 0:
                    if grid[b.x+1][b.y-1] == 0:
                        try_move(b, 1, -1)

        # 3. Drawing
        screen.fill(BG_COLOR)
        
        # Draw Target
        tx, ty = target_pos
        pygame.draw.circle(screen, (30, 30, 50), (tx*CELL_SIZE, ty*CELL_SIZE), 60)
        pygame.draw.circle(screen, (50, 50, 80), (tx*CELL_SIZE, ty*CELL_SIZE), 60, 2)

        # Draw Grid (Subtle)
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

        # Draw Blocks & Connections
        for b in blocks:
            # Draw Block
            rect = (b.x * CELL_SIZE + 2, b.y * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, BLOCK_COLOR, rect, border_radius=4)
            
            # Draw Magnetic Links (To Neighbors)
            cx, cy = b.x * CELL_SIZE + CELL_SIZE//2, b.y * CELL_SIZE + CELL_SIZE//2
            for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
                nx, ny = b.x + dx, b.y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and grid[nx][ny] != 0:
                    ncx, ncy = nx * CELL_SIZE + CELL_SIZE//2, ny * CELL_SIZE + CELL_SIZE//2
                    pygame.draw.line(screen, LINK_COLOR, (cx, cy), (ncx, ncy), 2)

        # UI Overlay
        ui_text = f"Agents: {len(blocks)} | Logic: Neural Swarm (PyTorch)"
        screen.blit(font.render(ui_text, True, (200, 200, 200)), (10, 10))
        screen.blit(font.render("Click to Spawn | 'R' to Reset", True, (150, 150, 150)), (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()