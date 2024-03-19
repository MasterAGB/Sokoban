import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import pygame

from SolsticeGame import SolsticeGame


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


# Solstice Deep Q-Learning
class SolsticeDQL:
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.

    optimizer = None  # NN Optimizer. Initialize later.


    # Train the Solstice environment
    def train(self, game: SolsticeGame, episodes):
        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size) #Size is 1000

        # Create policy and target network.
        policy_dqn = DQN(in_states=game.level_size, h1_nodes=game.level_size, out_actions=game.action_size)
        target_dqn = DQN(in_states=game.level_size, h1_nodes=game.level_size, out_actions=game.action_size)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy network - random, before training:')
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
            is_terminated = False  # True when agent falls in hole or reached goal
            is_truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while (not is_terminated and not is_truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = game.action_space_sample()  # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, game.level_size)).argmax().item()

                # Execute action
                new_state, reward, is_terminated, is_truncated, _ = game.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, is_terminated))

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
        torch.save(policy_dqn.state_dict(), "solstice_dql_" + str(game.level_index) + ".pt")

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
        plt.savefig("solstice_dql_" + str(game.level_index) + ".png")

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
    For example, the Solstice 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the Solstice environment with the learned policy
    def test(self, game: SolsticeGame, episodes):
        # Create Solstice instance
        num_states = game.level_size
        num_actions = game.action_size

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("solstice_dql_" + str(game.level_index) + ".pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            game.SetTitle("Test episode {}/{}".format(i + 1, episodes))
            state = game.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = game.step(action)

    def play(self, game: SolsticeGame):

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
            pygame.K_s: 'skin',
            pygame.K_t: 'train',
            pygame.K_e: 'test',
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
                            game.map_layout = game.generate_solvable_map(8, 8)
                            state, info = game.reset()
                            print("Map regenerated and position reset.")
                        elif action_mapping[event.key] == 'skin':
                            game.skin = random.choice(game.skins)
                            game.render()
                            print(f"Skin changed to {game.skin}.")
                        elif action_mapping[event.key] == 'train':
                            game.DisableDisplay()
                            solstice.train(game, 1500)
                            game.EnableDisplay()
                            print(f"Training the game.")
                        elif action_mapping[event.key] == 'test':
                            solstice.test(game, 10)
                            print(f"Testing the game.")
                        else:
                            action = action_mapping[event.key]
                            new_state, reward, terminated, truncated, _ = game.step(action)
                            state = new_state
                            print(
                                f"Action: {['Left', 'Down', 'Right', 'Up'][action]}, New State: {new_state}, Reward: {reward}")
                            if terminated:
                                if reward >= 1:
                                    state, info = game.Won();
                                    terminated = False
                                else:
                                    state, info = game.Lost();
                                    terminated = False

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
            best_action = ['Left', 'Down', 'Right', 'Up'][dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the Solstice map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()  # Print a newline every 4 states


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    solstice = SolsticeDQL()
    game = SolsticeGame(level_index=1)
    # http://www.1up-games.com/nes/solstice/map.html

    solstice.play(game)

    game.close()
