import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
import numpy as np
import pygame
import sys

# Инициализация Pygame
pygame.init()

# Размеры окна
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 500
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FINISH = (255, 64, 255)

# Размеры блоков
BLOCK_SIZE = 40

# Уровни игры
levels = [
    [
        "WWWWWWWWWWWWWWW",
        "W.............W",
        "WWWWWWWWWWWWWWW",
        "W...P.B...G...W",
        "WWWWWWWWWWWWWWW",
        "W.............W",
        "W.............W",
        "W.............W",
        "W.............W",
        "WWWWWWWWWWWWWWW",
    ],
    [
        "WWWWWWWWWWWWWWW",
        "W.............W",
        "W.............W",
        "W...P.G...B...W",
        "W.WWWWWWWWWWW.W",
        "W.............W",
        "W.............W",
        "W.............W",
        "W.............W",
        "WWWWWWWWWWWWWWW",
    ],
    [
        "WWWWWWWWWWWWWWW",
        "W.............W",
        "WWWWWWWWWWWWWWW",
        "W...P.........W",
        "W.WWWWWWWWWWW.W",
        "W.............W",
        "W.............W",
        "W.............W",
        "W....B....G...W",
        "WWWWWWWWWWWWWWW",
    ],
    [
        "WWWWWWWWWWWWWWW",
        "W.............W",
        "WWWWWWWWWWWWWWW",
        "W...P.........W",
        "W.............W",
        "W....B....G...W",
        "W.............W",
        "W.............W",
        "W.............W",
        "WWWWWWWWWWWWWWW",
    ],
    [
        "WWWWWWWWWWWWWWW",
        "W.....W.......W",
        "W.....W..W....W",
        "W..P..W.WWW...W",
        "W.....W...W...W",
        "W.WWWWW...W...W",
        "W.........W...W",
        "W......W..W.BGW",
        "W........W....W",
        "WWWWWWWWWWWWWWW",
    ],
]


def getFloorTiles(level):
    # Создаем копию уровня, где будут только стены и пустые плитки
    floor_tiles = []
    for row in level:
        new_row = ''
        for cell in row:
            if cell in "WG":
                new_row += cell  # Сохраняем стены
            else:
                new_row += "."  # Заменяем все остальные символы на пустые плитки
        floor_tiles.append(new_row)
    return floor_tiles


# Находим начальное положение игрока
def find_player(level):
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell == "P":
                return [x, y]


def restart_level():
    global level, level_floor_orig, player_pos, current_level
    print("Restart level index: ")
    print(current_level)
    level = copy.deepcopy(levels[current_level])
    level_floor_orig = getFloorTiles(level)
    player_pos = find_player(level)


current_level = 0
restart_level()




def draw_level(level, last_reward, last_total_reward):
    # Font initialization (consider doing this outside the function if called multiple times)
    pygame.font.init()  # Initialize the font module
    font = pygame.font.SysFont('Arial', 24)  # Create a Font object

    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            if cell == "W":
                pygame.draw.rect(screen, BLACK, rect)
            elif cell == "B":
                pygame.draw.rect(screen, BLUE, rect)
            elif cell == "P":
                pygame.draw.rect(screen, RED, rect)
            elif cell == "G":
                pygame.draw.rect(screen, GREEN, rect)
            elif cell == "F":
                pygame.draw.rect(screen, FINISH, rect)

    # Render the last reward and total reward text
    last_reward_text = font.render(f"Last Reward: {last_reward}", True, BLACK)
    total_reward_text = font.render(f"Total Reward: {last_total_reward}", True, BLACK)

    # Calculate Y position to draw below the level
    text_y_position = len(level) * BLOCK_SIZE + 10  # Adjust the 10 to increase/decrease padding

    # Blit the text onto the screen
    screen.blit(last_reward_text, (10, text_y_position))  # Adjust the X position as needed
    screen.blit(total_reward_text, (10, text_y_position + 30))  # Adjust for padding between lines



def check_win_condition(level):
    for row in level:
        if "B" in row:
            return False
    return True


def check_lost_condition(level):
    """Check if the level is in an unwinnable state."""
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell == 'B' and is_box_stuck(x, y, level):
                print("Lost box is stuck")
                return True
    return False


def is_box_stuck(x, y, level):
    """Check if a box at (x, y) is stuck against walls or corners."""
    # Check if box is at a border (simplified check, assumes outermost layer is all walls)
    if x == 0 or y == 0 or x == len(level[0]) - 1 or y == len(level) - 1:
        return True
    # Check for corners and walls directly adjacent to the box
    walls = [
        level[y - 1][x] == 'W',  # Up
        level[y + 1][x] == 'W',  # Down
        level[y][x - 1] == 'W',  # Left
        level[y][x + 1] == 'W'  # Right
    ]
    if sum(walls) >= 3:  # Box is in a corner or dead end
        return True
    if walls[0] and walls[3] or walls[0] and walls[2]:  # Stuck in upper corners
        return True
    if walls[1] and walls[3] or walls[1] and walls[2]:  # Stuck in lower corners
        return True
    return False


def set_cell(level, x, y, new_char):
    """Заменяет символ в уровне на заданных координатах на новый символ."""
    level[y] = level[y][:x] + new_char + level[y][x + 1:]

def detect_proximity_to_walls(x, y, level):
    """Detects if the player is close to walls."""
    proximity_threshold = 1  # Define how close to a wall is considered "close"
    wall_proximity = False
    level_height = len(level)
    level_width = len(level[0])

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            new_x, new_y = x + dx, y + dy
            # Check boundaries
            if 0 <= new_x < level_width and 0 <= new_y < level_height:
                if level[new_y][new_x] == 'W':
                    wall_proximity = True
                    return wall_proximity  # No need to check further
    return wall_proximity


def calculate_distance_to_nearest_box(x, y, level):
    """Calculates the Manhattan distance from the player to the nearest box."""
    box_positions = [(ix, iy) for iy, row in enumerate(level) for ix, cell in enumerate(row) if cell == 'B']
    distances = [abs(x - bx) + abs(y - by) for bx, by in box_positions]
    return min(distances) if distances else 0

def move_player(dx, dy):
    global player_pos, current_level, level, level_floor_orig

    reward = 0  # Initialize reward for this move
    level_completed = False  # Flag to indicate if the level is completed

    x, y = player_pos
    new_x, new_y = x + dx, y + dy
    target_cell = level[new_y][new_x]

    wall_proximity_before = detect_proximity_to_walls(x, y, level)
    distance_to_box_before = calculate_distance_to_nearest_box(x, y, level)

    if target_cell == "W":
        reward = -5  # Significantly penalize hitting a wall

    elif target_cell in ".G":
        set_cell(level, x, y, ".")
        set_cell(level, new_x, new_y, "P")
        player_pos = [new_x, new_y]
        reward = 0.1  # Reward for valid, non-box-pushing move

    elif target_cell == "B":
        beyond_x, beyond_y = new_x + dx, new_y + dy
        beyond_target_cell = level[beyond_y][beyond_x]
        if beyond_target_cell in ".G":
            set_cell(level, x, y, ".")
            set_cell(level, new_x, new_y, "P")
            set_cell(level, beyond_x, beyond_y, "B" if beyond_target_cell == "." else "F")
            player_pos = [new_x, new_y]
            reward = 5 if beyond_target_cell == "G" else 1  # Higher reward for moving box onto goal
        else:
            reward = -1  # Penalize if the box cannot be moved

    wall_proximity_after = detect_proximity_to_walls(new_x, new_y, level)
    distance_to_box_after = calculate_distance_to_nearest_box(new_x, new_y, level)

    # Adjust rewards based on wall proximity and moving closer to a box
    if wall_proximity_after:
        reward -= 0.5  # Penalize being close to a wall
    if distance_to_box_after < distance_to_box_before:
        reward += 0.5  # Reward for moving closer to a box

    # Check for win condition
    if check_win_condition(level):
        reward += 10
        level_completed = True
        current_level += 1
        if current_level < len(levels):
            print("Player win - next!")
            restart_level()
        else:
            print("You've completed all levels!")
            pygame.quit()
            sys.exit()

    # Check for lost condition
    if check_lost_condition(level):
        reward = -10  # Penalize for losing state
        restart_level()

    return reward, level_completed





class SokobanNet(nn.Module):
    def __init__(self):
        super(SokobanNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 15 * 10, 128)
        self.fc2 = nn.Linear(128, 4)  # Output layer: 4 actions (left, right, up, down)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 15 * 10)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def game_state_to_tensor(game_state):
    # Example function to convert game state to tensor
    mapping = {'W': 1, '.': 0, 'P': 2, 'B': 3, 'G': 4, 'F': 5}
    numeric_state = np.array([[mapping[cell] for cell in row] for row in game_state])
    tensor_state = torch.tensor(numeric_state, dtype=torch.float32)
    tensor_state = tensor_state.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return tensor_state



def select_action(state, model, epsilon):
    # Epsilon-greedy policy for exploration vs. exploitation
    if np.random.rand() < epsilon:  # Explore: choose a random action
        return np.random.randint(0, 4)
    else:  # Exploit: choose the best action predicted by the model
        with torch.no_grad():
            q_values = model(state).cpu().numpy()
        return np.argmax(q_values)


def simulate_step(action):
    """
    Simulate a step in the environment.

    Parameters:
    - action (int): The action to take, encoded as an integer (0: up, 1: right, 2: down, 3: left).

    Returns:
    - next_state (tensor): The next game state as a tensor.
    - reward (float): The reward obtained from taking the action.
    - done (bool): Whether the level has been completed.
    """
    if action == 4 or action not in [0, 1, 2, 3]:
        # Penalize the "do nothing" or invalid action
        reward = -0.1
        level_completed = check_win_condition(level)
        next_state = game_state_to_tensor(level)
        return next_state, reward, level_completed

    # Mapping from action integer to dx, dy
    action_mapping = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    dx, dy = action_mapping[action]

    # Execute the action using the move_player function
    reward, level_completed = move_player(dx, dy)

    # Convert the updated game state to a tensor
    next_state = game_state_to_tensor(level)

    return next_state, reward, level_completed





def ai_train(model, episodes, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epsilon = epsilon_start
    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Apply epsilon decay
        current_level = 0
        restart_level()
        state = game_state_to_tensor(level)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, model, epsilon)
            next_state, reward, done = simulate_step(action)

            total_reward += reward

            # Prepare for model update
            with torch.no_grad():
                future_rewards = model(next_state).max(1)[0].unsqueeze(0)
                target = reward + gamma * future_rewards * (1 - done)

            current_q_values = model(state).gather(1, torch.tensor([[action]]))
            loss = F.smooth_l1_loss(current_q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            # Rendering the game state
            screen.fill(WHITE)
            draw_level(level, reward, total_reward)
            pygame.display.flip()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        torch.save(model.state_dict(), f'sokoban_model_state_dict_{episode}.pth')




def play(ai):
    global level, level_floor_orig, player_pos, current_level
    current_level = 0  # Reset to the first level or any specific level you want the AI to play
    restart_level()

    state = game_state_to_tensor(level)  # Convert the initial state to tensor
    total_reward = 0
    running = True
    while running:


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    ai = not ai  # Toggle AI mode
                    print(f"AI Mode: {'On' if ai else 'Off'}")


        print(f"AI Mode: {'On' if ai else 'Off'}")
        if(ai):
            action = select_action(state, model, epsilon)
        else:
            action = user_select_action(state, model, epsilon)

        state, reward, done = simulate_step(action)
        # Update total reward
        total_reward += reward

        # Render the game
        screen.fill(WHITE)
        draw_level(level, reward, total_reward)
        pygame.display.flip()

        # Check if the level is completed or the game is closed
        if done:
            current_level += 1
            if current_level >= len(levels):
                print("All levels completed!")
                pygame.quit()
                break
            else:
                restart_level()
                state = game_state_to_tensor(level)




def user_select_action(state, model, epsilon):
    # Wait for the user to press a key and determine the action
    action = None
    while action is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                action = 4
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 2
    return action;






model = SokobanNet()  # Create a model instance with the same architecture
model.load_state_dict(torch.load('sokoban_model_state_dict_2.pth'))
model.eval()  # Set the model to evaluation mode

print(model)
# Example usage
tensor_state = game_state_to_tensor(level)
print(tensor_state.size())  # Should show torch.Size([1, 1, 15, 10])

optimizer = Adam(model.parameters(), lr=0.001)
epsilon = 0.05  # Set to a low value to make the bot exploit its learned behavior





ai_train(model, episodes=100)
#play(True)
#play(False)
