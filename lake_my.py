import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import pygame


class FrozenLakeGame:
    def __init__(self, map_name="4x4", is_dizzy=False, render_mode="human", game_skin ="default"):
        self.skins = ['default', 'portal', 'bombs', 'forest']
        self.skin = game_skin
        self.last_action = None
        global win, win_size;
        self.map_name = map_name
        self.is_dizzy = is_dizzy
        self.map_layout = self.load_map(map_name)
        self.state = 0
        self.done = False
        # Define action mapping: 0=Left, 1=Down, 2=Right, 3=Up
        # self.action_space = np.arange(4)
        # self.observation_space = np.arange(len(self.map_layout) * len(self.map_layout[0]))
        self.render_mode = render_mode

        self.observation_space = len(self.map_layout) * len(self.map_layout[0])
        self.action_space = 4  # Assuming Left, Down, Right, Up

        pygame.init()
        win_size = (600, 600)
        win = pygame.display.set_mode(win_size)
        pygame.display.set_caption("FrozenLake Play")

    def action_space_sample(self):
        # Returns a random action from 0, 1, 2, 3
        return random.randint(0, 3)

    def generate_solvable_map(self, rows, cols):
        # Initialize map with 'F' (free) cells
        game_map = [['F' for _ in range(cols)] for _ in range(rows)]

        # Starting point
        game_map[0][0] = 'S'

        # Ensure 'G' is placed at the bottom-right corner
        game_map[-1][-1] = 'G'

        # Create a guaranteed solvable path from S to G
        cur_row, cur_col = 0, 0

        # Randomly add holes, ensuring 'G' remains at its location
        for _ in range(int(rows * cols * 0.6)):  # Adjust density of 'H' as needed
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            if game_map[r][c] == 'F':
                game_map[r][c] = 'H'

        while cur_row < rows - 1 or cur_col < cols - 1:
            if cur_row < rows - 1 and (random.choice([True, False]) or cur_col == cols - 1):
                cur_row += 1  # Move down
            else:
                cur_col += 1  # Move right
            if cur_row < rows and cur_col < cols:
                game_map[cur_row][cur_col] = 'U'  # Mark path as 'F'

        # Reaffirm 'G' placement in case of overwrites
        game_map[-1][-1] = 'G'

        return ["".join(row) for row in game_map]

    def load_map(self, map_name):
        # Simplified map loader
        if map_name == "4x4":
            return [
                "SFFM",
                "FHFH",
                "FFFH",
                "HKFC"
            ]
        elif map_name == "8x8":
            return [
                "SFFFFFKF",
                "FFFFFFMF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFC"
            ]
        else:
            # Extract rows and cols from the map_name
            try:
                dimensions = map_name.split('x')
                rows, cols = int(dimensions[0]), int(dimensions[1])
                return self.generate_solvable_map(rows, cols)
            except ValueError:
                raise ValueError("Invalid map size format. Use 'NxN' format.")

    def step(self, action):

        if self.is_dizzy:
            # With some probability, choose a random action instead of the intended one
            if random.random() < 0.33:  # Adjust this probability as needed
                action = self.action_space_sample()

        # Simplified step function with directional movement.
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        reward = 0
        terminated = False
        truncated = False  # Assuming this version doesn't use truncation.
        info = {}  # Placeholder for additional info.

        # Calculate current position
        row = self.state // cols
        col = self.state % cols
        row_prev = row;
        col_prev = col;

        self.last_action = action;
        # Determine new position based on action
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(rows - 1, row + 1)
        elif action == 2:  # Right
            col = min(cols - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)

        # Update state
        self.state = row * cols + col

        self.moveAllMobs("M", "F")

        # Check for game termination conditions
        cell_prev= self.map_layout[row_prev][col_prev]
        if cell_prev == 'U':
            self.replaceThisCell(row_prev, col_prev, "H")

        cell = self.map_layout[row][col]
        if cell == 'G':
            terminated = True
            reward = 1
        elif cell == 'H':
            terminated = True
        elif cell == 'M':
            terminated = True
        elif cell == 'K':
            reward = 0.5
            self.replaceThisCell(row, col, "F")
            self.replaceAllCells("C", "G")


        self.render();

        return self.state, reward, terminated, truncated, info

    def replaceThisCell(self, row, col, new_type):
        """
        Replace the cell at the specified row and column with the new type.
        """
        if 0 <= row < len(self.map_layout) and 0 <= col < len(self.map_layout[0]):
            self.map_layout[row] = self.map_layout[row][:col] + new_type + self.map_layout[row][col + 1:]

    def moveAllMobs(self, mob_type, allowed_tile):
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        mobs_positions = [(r, c) for r in range(rows) for c in range(cols) if self.map_layout[r][c] == mob_type]

        for r, c in mobs_positions:
            random.shuffle(directions)  # Shuffle directions to randomize mob movement
            moved = False
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols and self.map_layout[new_r][new_c] == allowed_tile:
                    # Move mob to new position
                    self.map_layout[r] = self.map_layout[r][:c] + allowed_tile + self.map_layout[r][c + 1:]
                    self.map_layout[new_r] = self.map_layout[new_r][:new_c] + mob_type + self.map_layout[new_r][
                                                                                         new_c + 1:]
                    moved = True
                    break  # Break after moving to avoid trying other directions

            if not moved:
                # If the mob cannot move (all adjacent tiles are not allowed), it stays in its current position
                continue

    def replaceAllCells(self, from_type, to_type):
        """
        Replace all instances of one cell type with another throughout the map.
        """
        for row in range(len(self.map_layout)):
            self.map_layout[row] = self.map_layout[row].replace(from_type, to_type)

    def reset(self):
        self.state = 0
        self.done = False

        self.render();

        return self.state, {}

    def render(self):
        global win, win_size
        if self.render_mode != 'human':
            return

        def load_image(skin, name):
            """Loads an image from the 'tiles/' directory."""
            return pygame.image.load(f'tiles/{skin}/{name}.png')

        scale_factor = 2;

        def load_image_scaled(skin, image_name, scale_factor=2):
            # Load the image using your existing load_image function
            image = load_image(skin, image_name)
            # Get the current size of the image
            original_size = image.get_size()
            # Calculate the new size based on the scale factor
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            # Scale the image to the new size
            scaled_image = pygame.transform.scale(image, new_size)
            return scaled_image


        # Load tiles with added default and wall tiles
        tiles = {
            'CL': load_image_scaled('char', 'char_left', scale_factor),
            'CB': load_image_scaled('char', 'char_bottom', scale_factor),
            'CR': load_image_scaled('char', 'char_right', scale_factor),
            'CT': load_image_scaled('char', 'char_top', scale_factor),
            'S': load_image_scaled(self.skin, 'start', scale_factor),
            'F': load_image_scaled(self.skin, 'free', scale_factor),
            'H': load_image_scaled(self.skin, 'hole', scale_factor),
            'G': load_image_scaled(self.skin, 'goal', scale_factor),
            'C': load_image_scaled(self.skin, 'goalClosed', scale_factor),
            'K': load_image_scaled(self.skin, 'key', scale_factor),
            'M': load_image_scaled(self.skin, 'mob', scale_factor),
            'U': load_image_scaled(self.skin, 'unstable', scale_factor),
            'WTL': load_image_scaled(self.skin, 'wall', scale_factor),  # Wall Top Left
            'WTR': load_image_scaled(self.skin, 'wall', scale_factor),  # Wall Top Right
        }

        state = self.state
        map_layout = self.map_layout
        grid_size = len(self.map_layout)
        # Assume each tile has fixed size for simplicity
        tile_width, tile_height = 32 * scale_factor, 16 * scale_factor

        # This time, calculate offset to center the (-1, -1) tile
        # First, find the center of the window
        center_x = win_size[0] / 2
        center_y = win_size[1] / 2

        # Calculate offset_x such that (-1, -1) tile's center is at window's center
        # We adjust for half a tile width because (-1, -1) is off-center to the left
        # and we move it up by half the total map height in isometric projection to align top
        offset_x = center_x - (tile_width // 2)

        # Calculate offset_y to place (-1, -1) at the top of the screen
        # We take into account the entire height of the map in isometric projection
        # and adjust it so the top is at the center_y, moving it up by half the height of one tile
        total_height_iso = (grid_size * tile_height // 2) * 2  # Total height in isometric view
        offset_y = tile_height*2+tile_height*3

        win.fill((0, 0, 0))  # Clear the screen

        def drawChar(row, col):

            tile_type = "CR"
            if (self.last_action == 0):
                tile_type = "CL"
            elif (self.last_action == 1):
                tile_type = "CB"
            elif (self.last_action == 2):
                tile_type = "CR"
            elif (self.last_action == 3):
                tile_type = "CT"
            this_tile_offset_y = tiles[tile_type].get_size()[1] + 16 * scale_factor
            this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
            # Convert grid coordinates to isometric, including walls
            iso_x = (col - row) * (tile_width // 2) + offset_x + tile_width / 2 - this_tile_offset_x
            iso_y = (col + row) * (tile_height // 2) + offset_y - this_tile_offset_y
            win.blit(tiles[tile_type], (iso_x, iso_y))

        # Render tiles in isometric view, including boundary for walls
        for i in range(-1, grid_size):
            for j in range(-1, grid_size):
                # Determine the type of tile
                if i == -1 or j == -1:
                    if (j) == (-1):  # this must me in minus
                        tile_type = 'WTL'  # Top left wall for the first cell
                    elif (i) == (-1):  # this must be in minus
                        tile_type = 'WTR'  # Top right wall for the last cell in the first row

                else:
                    tile_type = map_layout[i][j]
                    if state == i * grid_size + j:
                        # Logic to change the tile based on the state, if needed
                        # For example, marking the current position differently
                        pass  # Implement state-based logic here

                this_tile_offset_y = tiles[tile_type].get_size()[1]
                this_tile_offset_x = tiles[tile_type].get_size()[0] / 2

                # Convert grid coordinates to isometric, including walls
                iso_x = (j - i) * (tile_width // 2) + offset_x + tile_width / 2 - this_tile_offset_x
                iso_y = (j + i) * (tile_height // 2) + offset_y - this_tile_offset_y

                win.blit(tiles[tile_type], (iso_x, iso_y))

                row = self.state // grid_size
                col = self.state % grid_size
                if (col == j and row == i):
                    drawChar(i, j);

            # Define your colors
        top_color = (193, 223, 254)
        bottom_color = (29, 99, 214)

        # Example usage
        self.draw_text_with_gradient(
            "Your Level Name, Rewards, Game Name", (50, 450), "font/solstice-nes.ttf", 18,
                                top_color,
                                bottom_color)

        pygame.display.flip()


    def draw_text_with_gradient(self, text, position, font_path, font_size, top_color, bottom_color):
        global win, win_size

        # Load the custom font
        font = pygame.font.Font(font_path, font_size)

        # Render the text to a new Surface
        text_surface = font.render(text, True, pygame.Color('white'))

        # Create a vertical gradient
        gradient_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
        for y in range(text_surface.get_height()):
            # Calculate the color for the current position
            alpha = y / text_surface.get_height()
            color = [top_color[j] * (1 - alpha) + bottom_color[j] * alpha for j in range(3)]
            pygame.draw.line(gradient_surface, color, (0, y), (text_surface.get_width(), y))

        # Blit the text surface onto the gradient surface
        gradient_surface.blit(text_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Blit the gradient surface to the target surface
        win.blit(gradient_surface, position)




    def renderOld(self):
        global win, win_size;

        state = self.state;
        # Determine grid size from the map layout
        map_layout = self.map_layout;
        grid_size = len(self.map_layout)
        cell_size = win_size[0] // grid_size
        clock = pygame.time.Clock()

        win.fill((0, 0, 0))  # Clear the screen

        # Draw grid and highlight tiles based on the dynamic map layout
        for i, row in enumerate(map_layout):
            for j, cell in enumerate(row):
                color = (255, 255, 255)  # Default to white for ice
                if cell == 'H':
                    color = (255, 0, 0)  # Red for holes
                elif cell == 'G':
                    color = (0, 255, 255)  # Cyan for goal

                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(win, color, rect)  # Fill cell
                pygame.draw.rect(win, (255, 255, 255), rect, 1)  # Cell border

        # Highlight current position
        row = state // grid_size
        col = state % grid_size
        highlight_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
        pygame.draw.rect(win, (0, 255, 0), highlight_rect)  # Green for player

        pygame.display.flip()  # Update the display
        clock.tick(30)  # Cap the frame rate

    def close(self):
        global pygame, win, win_size;
        pygame.quit()
        # Placeholder for any cleanup tasks.
        # For example, if using Pygame for rendering:
        # pygame.quit()
        pass

    def DisableDisplay(self):
        self.render_mode = None

    def EnableDisplay(self):
        self.render_mode = 'human'

    def SetTitle(self, param):
        print(param)
        pygame.display.set_caption("FrozenLake: " + param)
        pass


# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)  # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# FrozeLake Deep Q-Learning
class FrozenLakeDQL:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    ACTIONS = ['L', 'D', 'R', 'U']  # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, game: FrozenLakeGame, episodes):

        num_states = game.observation_space
        num_actions = game.action_space

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(episodes):
            game.SetTitle("Train episode {}/{}".format(i + 1, episodes))

            state = game.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = game.action_space_sample()  # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = game.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            if reward >= 1:
                rewards_per_episode[i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql_" + str(game.map_name) + ".pt")

        # Close environment
        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig("frozen_lake_dql_" + str(game.map_name) + ".png")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(
                            self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, env: FrozenLakeGame, episodes):
        # Create FrozenLake instance
        num_states = env.observation_space
        num_actions = env.action_space

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql_" + str(env.map_name) + ".pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            env.SetTitle("Test episode {}/{}".format(i + 1, episodes))
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

    def play(self, game: FrozenLakeGame):

        game.SetTitle("Play the game")
        game.reset()

        # Correctly decode the env.desc to get the map layout
        map_layout = game.map_layout

        action_mapping = {
            pygame.K_LEFT: 0,
            pygame.K_DOWN: 1,
            pygame.K_RIGHT: 2,
            pygame.K_UP: 3,
            pygame.K_r: 'reset',
            pygame.K_s: 'skin'
        }

        state, info = game.reset()
        terminated = False

        while not terminated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in action_mapping:
                        if action_mapping[event.key] == 'reset':
                            # Regenerate the map and reset position
                            dimensions = game.map_name.split('x')
                            rows, cols = int(dimensions[0]), int(dimensions[1])
                            game.map_layout = game.generate_solvable_map(rows, cols)
                            state, info = game.reset()
                            print("Map regenerated and position reset.")
                        elif action_mapping[event.key] == 'skin':
                            game.skin = random.choice(game.skins)
                            game.reset()
                            print(f"Skin changed to {game.skin}.")
                        else:
                            action = action_mapping[event.key]
                            new_state, reward, terminated, truncated, _ = game.step(action)
                            print(
                                f"Action: {['Left', 'Down', 'Right', 'Up'][action]}, New State: {new_state}, Reward: {reward}")
                            if terminated:
                                if reward == 1:
                                    print("Congratulations, you've reached the goal!")
                                else:
                                    print("Oops, you fell into a hole or died!")
                            state = new_state

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q) + ' '  # Concatenate q values, format to 2 decimals
            q_values = q_values.rstrip()  # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()  # Print a newline every 4 states


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    is_dizzy = False
    map_size = "8x8"

    env = FrozenLakeGame(map_name=map_size, is_dizzy=is_dizzy, render_mode='human')
    # http://www.1up-games.com/nes/solstice/map.html

    env.DisableDisplay()
    # frozen_lake.train(env, 1000)
    env.EnableDisplay()
    # frozen_lake.test(env, 10)
    frozen_lake.play(env)

    env.close()
