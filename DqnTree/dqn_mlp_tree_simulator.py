# ===========================================================================
# File: dqn_mlp_tree_simulator.py
# Description: Implements a Deep Q-Network (DQN) agent using an MLP
#              to learn how to tree (StructuredTree).
# Author:      Aditya Jindal
# Date:        2025-03-29
# ===========================================================================


# Required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time  # For timestamp

# Define the structure for storing experiences in the replay buffer
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

# --- Environment Simulation ---
class StructuredTreeEnv:
    """
    Simulates a tree whose growth and health are affected by agent actions.
    """
    def __init__(self, state_size=5, action_size=4):
        """
        Initializes the tree environment.

        Args:
            state_size (int): Number of features in the state vector.
            action_size (int): Number of possible actions.
        """
        self.state_size = state_size  # [Height, Branches, Leaves, Health, Age]
        self.action_size = action_size # 0: Water, 1: Prune, 2: Fertilize, 3: Wait
        self._state = None
        self.action_names = {0: "Water", 1: "Prune", 2: "Fertilize", 3: "Wait"}
        print(f"Initialized StructuredTreeEnv Environment (State Size: {state_size}, Action Size: {action_size})")

    def reset(self):
        """
        Resets the tree to a random initial state.

        Returns:
            np.ndarray: The initial state vector.
        """
        # Initialize with somewhat realistic starting values
        self._state = np.array([
            np.random.uniform(0.5, 2.0), # Height (m)
            np.random.randint(1, 5),     # Branches
            np.random.randint(20, 50),   # Leaves
            np.random.uniform(0.7, 0.9), # Health (0-1)
            np.random.randint(0, 3)      # Age (years)
        ], dtype=np.float32) # Use float32 for consistency with PyTorch
        # print(f"TreeEnv Reset. Initial state: {self._state}")
        return self._state

    def step(self, action):
        """
        Simulates one time step in the environment based on the agent's action.

        Args:
            action (int): The action chosen by the agent (0-3).

        Returns:
            tuple: A tuple containing (next_state, reward, done, info).
                   - next_state (np.ndarray): The state after taking the action.
                   - reward (float): The reward received for the action.
                   - done (bool): Whether the episode has ended (tree died).
                   - info (dict): Auxiliary information (empty here).
        """
        if not (0 <= action < self.action_size):
            print(f"Warning: Invalid action {action} received!")
            action = 3

        height, branches, leaves, health, age = self._state
        reward = 0.0
        action_name = self.action_names.get(action, "Unknown")

        # --- Action Effects & State Updates ---
        if action == 0:  # Water
            # Increases health, slightly more effective if health is lower
            health_increase = np.random.uniform(0.05, 0.15) * (1.1 - health)
            health = min(1.0, health + health_increase)
            reward = 0.5 + health # Reward based on maintaining health

        elif action == 1:  # Prune
            if branches > 1 and leaves > 15:
                branches_removed = 1 # Could be random amount
                leaves_lost = random.randint(int(leaves * 0.1), int(leaves * 0.3))
                branches -= branches_removed
                leaves = max(10, leaves - leaves_lost)
                # Pruning might slightly stress the tree short-term but help long-term
                health = max(0.0, health - 0.02)
                # Reward for successful pruning, less if health is very low
                reward = 0.3 * health

            else:
                # Penalize trying to prune too much
                reward = -0.5

        elif action == 2:  # Fertilize
            # Boosts growth (height, leaves), small health cost if overused?
            height_increase = random.uniform(0.1, 0.3) * health # Growth depends on health
            leaves_increase = random.randint(5, 10) * health
            height += height_increase
            leaves += leaves_increase
            # health = max(0.0, health - 0.01) # Optional small cost
            # Reward significant growth
            reward = 0.6 + height_increase * 2

        elif action == 3:  # Wait
            # Natural processes: aging, slow growth, health decline without care
            age += 0.1 # Simulate passage of time (e.g., 0.1 years per step)
            natural_growth = 0.05 * health * (1 - age/20 if age < 20 else 0) # Growth slows with age
            height += natural_growth
            branches += int(random.random() < 0.1 * health) # Small chance of new branch if healthy
            leaves += int(random.uniform(1, 5) * health) # Natural leaf growth/loss balance
            health = max(0.0, health - 0.03) # Natural health decline without intervention
            # Small negative reward for inaction, less penalty if tree is healthy
            reward = -0.2 * (1.0 - health)

        # Apply state updates
        self._state = np.array([height, branches, leaves, health, age], dtype=np.float32)

        # --- Termination Condition ---
        # Episode ends if the tree's health drops to zero or below
        done = health <= 0
        if done:
            reward = -10.0

        return self._state, reward, done, {} # info dict is empty

    def close(self):
        """Performs any necessary cleanup."""
        print("StructuredTree environment closed.")

# --- Neural Network Definition (MLP) ---
class QNetwork(nn.Module):
    """Simple MLP Q-Network Model."""

    def __init__(self, state_size, action_size, hidden_dim=64):
        """Initializes parameters and builds model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_dim (int): Number of nodes in hidden layers
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (torch.device): device to send tensors to (CPU or CUDA)
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Ensure states are stored as numpy arrays for consistent handling before batching
        e = Experience(np.array(state, dtype=np.float32),
                       action, reward,
                       np.array(next_state, dtype=np.float32),
                       done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert batch of Experiences to tensors and move to the designated device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class StructuredTree:
    """Interacts with and learns from the environment using DQN with an MLP."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config['epsilon_start']
        self.time_step = 0  # To track when to update network

        # Determine compute device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-Network (Using MLP architecture)
        self.qnetwork_local = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['lr'])

        # Initialize target network weights the same as local network
        self._hard_update_target_network()

        # Replay memory
        self.memory = ReplayBuffer(config['buffer_size'], config['batch_size'], self.device)

        print(f"Initialized Tree (MLP based).")

    def _preprocess_state(self, state):
        """Convert numpy state to a PyTorch tensor and move to device."""
        # Add batch dimension (unsqueeze(0)) as network expects batches
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return state_tensor

    def select_action(self, state, use_epsilon=True):
        """Uses the current policy (Q-network) to select an action."""
    StopAsyncIteration    state_tensor = self._preprocess_state(state) # Prepare state for network

        # Set network to evaluation mode for inference
        self.qnetwork_local.eval()
        with torch.no_grad():   # Disable gradient calculations
            action_values = self.qnetwork_local(state_tensor)
        # Set network back to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if use_epsilon and random.random() < self.epsilon:
            # Exploration: choose a random action
            action = random.choice(np.arange(self.action_size))
            # print(f"  (Action: Random - {action})") # Debug print
        else:
            # Exploitation: choose the action with the highest Q-value
            # Move action values to CPU to convert to numpy for argmax
            action = np.argmax(action_values.cpu().data.numpy())
            # print(f"  (Action: Greedy - {action})") # Debug print

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and trigger learning step."""
        # State and next_state should be numpy arrays here
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % self.config['update_every']
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.config['batch_size']:
                experiences = self.memory.sample() # Samples are already on self.device
                self._learn(experiences)

    def _learn(self, experiences):
        """Update Q-network parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences # Unpack batch

        # --- Calculate Target Q-values (using target network) ---
        # Get Q-values for next_states from target network
        q_values_next_target = self.qnetwork_target(next_states).detach()
        # Select the maximum Q-value for each next_state (greedy policy)
        q_targets_next = q_values_next_target.max(1)[0].unsqueeze(1) # max returns (values, indices)

        # Compute Q targets for current states: R + gamma * max_a' Q_target(s', a')
        # Target is simply R if the episode finished (done=1)
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))

        # --- Calculate Expected Q-values (using local network) ---
        # Get Q-values from local model for the states in the batch
        q_values_local = self.qnetwork_local(states)
        # Select the Q-values corresponding to the *actions actually taken* in the batch
        q_expected = q_values_local.gather(1, actions)

        # --- Compute Loss (Mean Squared Error) ---
        loss = F.mse_loss(q_expected, q_targets)

        # --- Minimize the loss ---
        self.optimizer.zero_grad() # Reset gradients
        loss.backward()           # Backpropagate error
        # Optional: Clip gradients to prevent large updates
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()      # Update local network weights

        # --- Update target network (soft update) ---
        self._soft_update_target_network()


    def _hard_update_target_network(self):
        """Copy weights from local network to target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def _soft_update_target_network(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        tau = self.config['tau']
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """Linearly decay exploration rate epsilon."""
        self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)

    def save_model(self, filepath):
        """Save the trained local Q-network weights."""
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load weights into the local Q-network."""
        # Load weights onto the agent's current device
        map_location = self.device
        self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_location))
        self._hard_update_target_network()  # Ensure target network also matches loaded weights
        print(f"Model loaded from {filepath} onto {self.device}")

# --- Main Execution Block ---
if __name__ == '__main__':
    print("==============================================")
    print(" Starting DQN Agent Training for Tree Simulator ")
    print("==============================================")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Configuration ---
    config = {
        'state_size': 5,          # Env state: [Height, Branches, Leaves, Health, Age]
        'action_size': 4,         # Env actions: [Water, Prune, Fertilize, Wait]
        'hidden_dim': 64,         # Neurons in hidden layers
        'buffer_size': int(5e4),  # Replay buffer size (e.g., 50,000)
        'batch_size': 64,         # Minibatch size for learning
        'gamma': 0.99,            # Discount factor for future rewards
        'lr': 5e-4,               # Learning rate for Adam optimizer
        'tau': 1e-3,              # For soft update of target network parameters
        'update_every': 4,        # How often to update the network (steps)
        'epsilon_start': 1.0,     # Starting exploration rate
        'epsilon_end': 0.01,      # Minimum exploration rate
        'epsilon_decay': 0.995    # Multiplicative decay factor per episode
    }

    # --- Initialization ---
    env = StructuredTreeEnv(config['state_size'], config['action_size'])
    agent = StructuredTree(state_size=config['state_size'], action_size=config['action_size'], config=config)

    # --- Pre-populate Replay Buffer ---
    print("\n--- Pre-populating Replay Buffer ---")
    pre_populate_steps = config['batch_size'] * 10 # Ensure enough unique samples
    state = env.reset()
    for _ in range(pre_populate_steps):
        action = random.choice(np.arange(config['action_size'])) # Random actions
        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done) # Add to agent's buffer
        state = next_state
        if done:
            state = env.reset()
    print(f"Replay buffer pre-populated with {len(agent.memory)} experiences.")


    # --- Training Loop ---
    num_episodes = 200  # Number of episodes to train for
    max_steps_per_episode = 150 # Max steps per episode (prevent infinite loops)
    scores = []       # List to store total reward per episode
    scores_window = deque(maxlen=100) # Store last 100 episode scores for moving average

    print("\n--- Starting Training Loop ---")
    training_start_time = time.time()

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        # print(f"\n--- Episode {i_episode} ---") # Verbose episode start

        for t in range(max_steps_per_episode):
            # 1. Agent selects action
            action = agent.select_action(state, use_epsilon=True)

            # 2. Environment executes action
            next_state, reward, done, _ = env.step(action)

            # 3. Agent stores experience and learns
            agent.store_experience(state, action, reward, next_state, done)

            # Update state and total reward
            state = next_state
            episode_reward += reward

            if done:
                # print(f"  Episode terminated at step {t + 1}")
                break

        # --- End of Episode ---
        scores.append(episode_reward)
        scores_window.append(episode_reward)
        agent.decay_epsilon() # Decay exploration rate

        avg_score = np.mean(scores_window)

        # Print progress update
        print(f"\rEpisode {i_episode}\tAvg Reward (Last {len(scores_window)}): {avg_score:.2f}\tReward: {episode_reward:.2f}\tEpsilon: {agent.epsilon:.3f}   ", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score (Last 100): {avg_score:.2f}\tCurrent Reward: {episode_reward:.2f}\tEpsilon: {agent.epsilon:.3f}")

    # --- End of Training ---
    training_end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds")
    # print(f"Final Average Score (Last 100 episodes): {np.mean(scores_window):.2f}")

    # Clean up environment
    env.close()

    print("\n==============================================")
    print(" DQN Agent Training for Tree Simulator Finished ")
    print("==============================================")