# ===========================================================================
# File: dqn_tree_agent.py
# Description: Implements a Deep Q-Network (DQN) agent to learn node
#              prioritization in a structured tree environment.
# Author:      ADITYA JINDAL
# Date:        2025-03-29
# ===========================================================================

# --- How to Run This Code ---
# 1. Prerequisites:
#    - Python 3.6 or later installed.
#    - PyTorch and NumPy libraries installed.
#      Open your terminal or command prompt and run:
#      pip install torch numpy
#
# 2. Save the Code:
#    - Save this entire code block as a Python file (e.g., `dqn_tree_agent.py`).
#
# 3. Execute from Terminal:
#    - Navigate to the directory where you saved the file in your terminal.
#    - Run the script using:
#      python dqn_tree_agent.py
#
# 4. Expected Output:
#    - Timestamp and location context message.
#    - Initialization messages for the environment, agent, network, and buffer.
#    - Messages indicating the pre-population of the replay buffer.
#    - Output for each training episode, showing the total episode reward,
#      average reward over the last 10 episodes, and the decaying epsilon value.
#    - A final message when the training completes.
# ==============================================================================


# Required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time

# Define the structure for storing experiences in the replay buffer
# state: The state observed by the agent.
# action: The action taken by the agent.
# reward: The reward received after taking the action.
# next_state: The resulting state after taking the action.
# done: A boolean indicating if the episode terminated after this transition.
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])


# --- Neural Network Definition ---
class QNetwork(nn.Module):
    """
    Neural Network Model for approximating the Q-function.
    Uses a simple Multi-Layer Perceptron (MLP) architecture.
    """
    def __init__(self, state_size, action_size, hidden_dim=64):
        """
        Initializes the QNetwork.

        Args:
            state_size (int): Dimension of the input state space.
            action_size (int): Dimension of the output action space (number of possible actions).
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        """
        Performs the forward pass of the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Replay Buffer ---
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for Experience Replay.
    Helps break correlations between consecutive experiences and stabilize training.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initializes the ReplayBuffer.

        Args:
            buffer_size (int): Maximum number of experiences to store.
            batch_size (int): Number of experiences to sample during learning.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # print(f"Initialized ReplayBuffer with size={buffer_size}, batch_size={batch_size}")

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.

        Args:
            state: The state observed.
            action: The action taken.
            reward: The reward received.
            next_state: The next state observed.
            done (bool): Whether the episode terminated.
        """
        # Store states as numpy arrays for consistency before batching
        e = Experience(np.array(state), action, reward, np.array(next_state), done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly samples a batch of experiences from the buffer.

        Returns:
            tuple: A tuple of tensors (states, actions, rewards, next_states, dones).
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert lists of numpy arrays/values into batched PyTorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.

        Returns:
            int: The size of the memory.
        """
        return len(self.memory)


# --- AI Agent Definition ---
class Tree:
    """
    Deep Q-Network (DQN) agent that interacts with and learns from the environment.
    Implements epsilon-greedy exploration, experience replay, and target network updates.
    """
    def __init__(self, state_size, action_size, config):
        """
        Initializes the Tree.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            config (dict): Dictionary containing hyperparameters and settings.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config['epsilon_start']  # Exploration rate
        self.time_step = 0 # Counter for triggering network updates

        # Determine compute device (GPU if available, else CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-Networks: Local network for action selection/learning, Target network for stable target values
        self.qnetwork_local = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['lr'])

        # Initialize target network weights to be the same as the local network
        self._hard_update_target_network()

        # Replay memory
        self.memory = ReplayBuffer(config['buffer_size'], config['batch_size'])

        print(f"  State Size: {state_size}")
        print(f"  Action Size: {action_size}")
        # print(f"  Config: {config}") # Config can be long, optional print

    def _preprocess_state(self, state):
        """
        Converts the environment state (expected as numpy array) into a PyTorch tensor
        suitable for the network, adds a batch dimension, and moves it to the correct device.

        Args:
            state (np.ndarray): The input state from the environment.

        Returns:
            torch.Tensor: Processed state tensor ready for the QNetwork.
        """
        # Convert state to a PyTorch tensor, add batch dimension, and send to device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return state_tensor

    def select_action(self, state, use_epsilon=True):
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): The current state.
            use_epsilon (bool): If True, uses epsilon-greedy exploration. If False, always acts greedily.

        Returns:
            int: The selected action.
        """
        state_tensor = self._preprocess_state(state)

        # Set network to evaluation mode for inference
        self.qnetwork_local.eval()
        with torch.no_grad(): # Disable gradient calculations during inference
            action_values = self.qnetwork_local(state_tensor)
        # Set network back to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if use_epsilon and random.random() < self.epsilon:
            # Explore: choose a random action
            # action_type = "Random" # Optional debug print
            action = random.choice(np.arange(self.action_size))
        else:
            # Exploit: choose the action with the highest Q-value
            # action_type = "Greedy" # Optional debug print
            action = np.argmax(action_values.cpu().data.numpy()) # Move tensor to CPU before converting to numpy

        # print(f"    Action Selected: {action} ({action_type})") # Verbose action selection
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer and triggers learning
        at regular intervals.

        Args:
            state: The state observed.
            action: The action taken.
            reward: The reward received.
            next_state: The next state observed.
            done (bool): Whether the episode terminated.
        """
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % self.config['update_every']
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.config['batch_size']:
                experiences = self.memory.sample()
                # Move experience tensors to the correct device before learning
                experiences_on_device = tuple(exp.to(self.device) for exp in experiences)
                self._learn(experiences_on_device)

    def _learn(self, experiences):
        """
        Updates the Q-network parameters using a batch of experiences.
        Calculates TD targets and minimizes the MSE loss between expected Q-values
        and target Q-values.

        Args:
            experiences (tuple): A tuple of tensors (states, actions, rewards, next_states, dones)
                                 already moved to the agent's device.
        """
        states, actions, rewards, next_states, dones = experiences

        # --- Calculate Target Q-values ---
        # Get max predicted Q values for next states from target model
        # .detach() prevents gradients from flowing into the target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states: R + gamma * max_a' Q_target(s', a')
        # Target is 0 if the episode finished (done=1)
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))

        # --- Calculate Expected Q-values ---
        # Get expected Q values from local model for the actions that were taken
        # .gather(1, actions) selects the Q-value corresponding to the action taken in each state
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- Compute Loss ---
        # Mean Squared Error (MSE) loss between expected and target Q-values
        loss = F.mse_loss(q_expected, q_targets)

        # --- Optimize the Model ---
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()           # Calculate gradients
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()      # Update network weights

        # --- Update Target Network ---
        # Soft update the target network weights towards the local network weights
        self._soft_update_target_network()

        # print(f"      Learned - Loss: {loss.item():.4f}") # Verbose learning step

    def _hard_update_target_network(self):
        """Copies weights from the local network to the target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        # print("Hard updated target network.") # Optional print

    def _soft_update_target_network(self):
        """
        Performs a soft update of the target network's weights:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = self.config['tau']
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """Decays the epsilon value for exploration-exploitation tradeoff."""
        self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)

    def save_model(self, filepath):
        """Saves the trained local Q-network weights to a file."""
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads pre-trained Q-network weights from a file."""
        # Ensure loading maps weights to the correct device (CPU or GPU)
        map_location = self.device
        self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_location))
        # Important: Also update the target network to match the loaded weights
        self._hard_update_target_network()
        print(f"Model loaded from {filepath} onto device {self.device}")


# --- Environment Simulation ---
class StructuredTree:
    """
    Simulates a simple environment where the agent needs to choose a node (action)
    based on a state vector. Generates structured states and rewards.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the simulated environment.

        Args:
            state_size (int): The dimension of the state vector.
            action_size (int): The number of possible actions (nodes).
        """
        self.state_size = state_size
        self.action_size = action_size # Number of nodes to choose from
        self._state = np.zeros(state_size) # Internal representation of the current state
        self.steps_taken = 0
        self.max_steps = 50 # Maximum steps per episode before forced termination
        print(f"Initialized StructuredTree (State Size: {state_size}, Action Size: {action_size}, Max Steps: {self.max_steps})")

    def _generate_state(self):
        """
        Generates the next state based on the current internal state, simulating dynamics.
        State features are hypothetical: [complexity, urgency, interaction_flag, noise...]

        Returns:
            np.ndarray: The next state vector.
        """
        state = np.zeros(self.state_size)
        # Feature 0: Complexity (e.g., slowly decays but can be influenced by actions)
        state[0] = np.clip(self._state[0] * 0.9 + np.random.rand() * 0.1, 0, 1)
        # Feature 1: Urgency (e.g., changes somewhat randomly, influenced by actions)
        state[1] = np.random.rand()
        # Feature 2: Interaction Flag (e.g., binary flag, sometimes active, reset by actions)
        state[2] = 1.0 if np.random.rand() > 0.8 else 0.0
        # Remaining features (if any): Noise
        if self.state_size > 3:
            state[3:] = np.random.rand(self.state_size - 3) * 0.1
        return state

    def reset(self):
        """
        Resets the environment to a new initial state for the start of an episode.

        Returns:
            np.ndarray: The initial state vector.
        """
        # Reset to a random initial state
        self._state = np.random.rand(self.state_size)
        # Ensure urgency and interaction have clear starting points for demo consistency
        self._state[1] = np.random.rand() * 0.5  # Start with lower urgency
        self._state[2] = 0.0                     # Start with no interaction flag
        self.steps_taken = 0
        # print("Environment Reset") # Optional print
        return self._state

    def step(self, action):
        """
        Executes one time step in the environment based on the agent's action.

        Args:
            action (int): The action chosen by the agent (index of the node).

        Returns:
            tuple: A tuple containing (next_state, reward, done, info).
                   - next_state (np.ndarray): The state after taking the action.
                   - reward (float): The reward received for the action.
                   - done (bool): Whether the episode has ended.
                   - info (dict): Auxiliary information (empty in this case).
        """
        if not (0 <= action < self.action_size):
            raise ValueError(f"Invalid action {action} for action_size {self.action_size}")

        self.steps_taken += 1
        # Simulate reward based on current state and chosen action
        reward = -0.05 # Small cost for taking any step (encourages efficiency)

        # --- Hypothetical Reward Logic ---
        # Agent should learn to prefer certain actions based on state features:
        # Reward choosing node 0 if urgency (state[1]) is high
        if self._state[1] > 0.7 and action == 0:
            reward += 0.8
            # print("  -> Rewarding action 0 (high urgency)")
        # Reward choosing node 1 if interaction flag (state[2]) is set
        elif self._state[2] > 0.5 and action == 1:
            reward += 0.6
            # print("  -> Rewarding action 1 (interaction flag)")
        # Small penalty for choosing node 2 frequently (discourage overuse)
        elif action == 2:
            reward -= 0.1
        else:
            # Small random positive reward for other valid actions
            reward += np.random.rand() * 0.1

        # Generate the base next state
        next_state_base = self._generate_state()

        # --- Simulate State Transition Dynamics Based on Action ---
        # Action 0 might reduce urgency
        if action == 0:
            next_state_base[1] *= 0.5
        # Action 1 might reset the interaction flag
        if action == 1:
            next_state_base[2] = 0.0
        # Action might slightly influence the next complexity
        next_state_base[0] *= (0.95 + action * 0.01)

        # Update the internal state, ensuring values stay within a valid range [0, 1]
        self._state = np.clip(next_state_base, 0, 1)

        # --- Determine if the Episode is Done ---
        done = False
        # End episode if a high reward is achieved (task potentially completed)
        if reward > 0.7:
            done = True
            # print("  -> High reward achieved, ending episode.")
        # End episode if maximum number of steps is reached
        elif self.steps_taken >= self.max_steps:
            done = True
            # print("  -> Max steps reached, ending episode.")

        # print(f"  Step {self.steps_taken}: Action={action}, Reward={reward:.2f}, Done={done}")
        # print(f"    Next State: {np.round(self._state, 2)}")
        # time.sleep(0.01) # Optional small delay for visualization/realism

        # Return standard OpenAI Gym-like tuple
        return self._state, reward, done, {} # info dict is usually empty for simple envs

    def close(self):
        """Performs any necessary cleanup for the environment."""
        # In this simple simulation, there's nothing to clean up.
        print("StructuredTree environment closed.")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("==============================================")
    print(" Starting DQN Agent Training ")
    print("==============================================")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Location Context: Bengaluru, Karnataka, India") # Example context

    # --- Configuration ---
    config = {
        'state_size': 4,        # Dimension of the state vector [complexity, urgency, interaction, noise]
        'action_size': 3,       # Number of nodes/actions the agent can choose
        'hidden_dim': 128,      # Number of neurons in hidden layers of the Q-network
        'buffer_size': int(5e4),# Max size of the replay buffer (e.g., 50,000 experiences)
        'batch_size': 128,      # Number of experiences sampled from buffer for each learning step
        'gamma': 0.99,          # Discount factor for future rewards
        'lr': 1e-4,             # Learning rate for the Adam optimizer
        'tau': 1e-3,            # Interpolation parameter for soft target network updates
        'update_every': 4,      # How often to update the network (every 4 steps)
        'epsilon_start': 1.0,   # Initial value of epsilon for epsilon-greedy exploration
        'epsilon_end': 0.01,    # Minimum value of epsilon
        'epsilon_decay': 0.996  # Multiplicative factor for decaying epsilon each episode
    }

    # --- Initialization ---
    env = StructuredTree(config['state_size'], config['action_size'])
    agent = Tree(state_size=config['state_size'], action_size=config['action_size'], config=config)

    # --- Pre-populate Replay Buffer ---
    # Fill the buffer with some initial experiences gained by taking random actions.
    # This helps stabilize the learning process at the beginning.
    print("\n--- Pre-populating Replay Buffer ---")
    pre_populate_steps = config['batch_size'] * 5 # Ensure enough samples for several initial learning steps
    state = env.reset()
    for _ in range(pre_populate_steps):
        action = random.choice(np.arange(config['action_size'])) # Take random actions
        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done) # Add directly to agent's buffer
        state = next_state
        if done:
            state = env.reset() # Reset if episode finishes during pre-population
    print(f"Replay buffer pre-populated with {len(agent.memory)} experiences.")

    # --- Training Loop ---
    num_episodes = 50 # Total number of training episodes
    max_steps_per_episode = env.max_steps # Max steps allowed within one episode
    scores = []       # List to store total reward per episode
    scores_window = deque(maxlen=10) # Store scores of the last 10 episodes for averaging

    print("\n--- Starting Training Loop ---")
    start_time = time.time()

    for i_episode in range(1, num_episodes + 1):
        state = env.reset() # Reset environment at the start of each episode
        episode_reward = 0  # Track reward accumulated in this episode

        # Loop within a single episode
        for t in range(max_steps_per_episode):
            # Agent selects action based on current state and epsilon-greedy policy
            action = agent.select_action(state, use_epsilon=True)

            # Environment performs the action and returns the outcome
            next_state, reward, done, _ = env.step(action)

            # Agent stores this experience and potentially learns from a batch
            agent.store_experience(state, action, reward, next_state, done)

            # Move to the next state and accumulate reward
            state = next_state
            episode_reward += reward

            # Check if the episode has finished
            if done:
                break

        # --- End of Episode ---
        scores.append(episode_reward)       # Save the total reward for this episode
        scores_window.append(episode_reward) # Add score to the rolling window
        agent.decay_epsilon()               # Decrease epsilon for less exploration over time

        avg_score = np.mean(scores_window) # Calculate average score over the window

        # Print progress: current episode, total reward, average reward, epsilon
        print(f"\rEpisode {i_episode}\tTotal Reward: {episode_reward:.2f}\tAverage Reward (last {len(scores_window)}): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}", end="")
        # Print a newline every 10 episodes to avoid overly long lines
        if i_episode % 10 == 0:
            print(f"\rEpisode {i_episode}\tTotal Reward: {episode_reward:.2f}\tAverage Reward (last {len(scores_window)}): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}")

        # --- Optional: Add condition to stop training early if desired performance is reached ---
        # if avg_score >= desired_score_threshold:
        #     print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}")
        #     agent.save_model("dqn_model_solved.pth")
        #     break

    # --- End of Training ---
    end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # --- Optional: Save the final model ---
    # agent.save_model("dqn_model_final.pth")

    # Clean up the environment
    env.close()

    print("\n==============================================")
    print(" DQN Agent Training Finished ")
    print("==============================================")