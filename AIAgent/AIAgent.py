# - Defines state, actions (Water, Prune, Fertilize, Wait), and rewards.
# - Issues & Improvements:
#   1. Ensure state updates reflect realistic tree growth dynamics.
#   2. Improve reward function to balance short-term and long-term growth.
#   3. Handle invalid actions gracefully.
#   4. Optimize termination conditions for meaningful episode completion.
#   5. Add logging/debugging to track agent behavior.
# - Enhancements: Consider environmental factors like seasons, pests, or weather.

# Required libraries (ensure you have them installed: pip install torch numpy)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Define the structure for storing experiences in the replay buffer
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

class GNNLayer(nn.Module):
    """A single layer of Graph Neural Network using GCN."""

    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features (N x F), edge_index: Graph connectivity
        return F.relu(self.conv(x, edge_index))


class QNetwork(nn.Module):
    """Graph Neural Network based Q-Network Model."""

    def __init__(self, state_size, action_size, hidden_dim=64):
        super(QNetwork, self).__init__()

        # Dummy values for the number of nodes and features in the graph
        num_nodes = 10  # Example number of nodes in the graph
        num_node_features = state_size  # Number of features per node

        # Define GNN layers
        self.gnn1 = GNNLayer(num_node_features, hidden_dim)  # First GNN layer
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)  # Second GNN layer

        # Fully connected layers after GNN processing
        self.fc1 = nn.Linear(hidden_dim * num_nodes, hidden_dim)  # Flattened output from GNN
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)  # Output layer for action values

        print(f"Initialized QNetwork with state_size={state_size}, action_size={action_size}")

    def forward(self, x, edge_index):
        """Forward pass through the network."""

        # Pass through GNN layers
        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)

        # Global pooling (e.g., mean pooling)
        x = F.relu(torch.mean(x, dim=0).unsqueeze(0))  # Assume batch size of 1 for simplicity

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)  # Output raw Q-values for each action

# Example usage:
if __name__ == '__main__':
    state_size = 5  # Number of features per node (example)
    action_size = 4  # Number of possible actions
    model = QNetwork(state_size=state_size, action_size=action_size)

    # Dummy input (e.g., node features and edge index for a simple graph)
    dummy_node_features = torch.rand((10, state_size))  # 10 nodes with 'state_size' features each
    dummy_edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], dtype=torch.long).t().contiguous()

    q_values = model(dummy_node_features, dummy_edge_index)
    print("Q-values:", q_values)


# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        print(f"Initialized ReplayBuffer with size={buffer_size}, batch_size={batch_size}")

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert batch of Experiences to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# --- AI Agent Definition ---
class AIAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action (number of nodes to choose from)
            config (dict): Configuration parameters
                - hidden_dim (int): Dimension of hidden layers
                - buffer_size (int): Replay buffer size
                - batch_size (int): Minibatch size
                - gamma (float): Discount factor
                - lr (float): Learning rate
                - tau (float): For soft update of target parameters
                - update_every (int): How often to update the network
                - epsilon_start (float): Starting value of epsilon (exploration)
                - epsilon_end (float): Minimum value of epsilon
                - epsilon_decay (float): Multiplicative factor (per episode) for decreasing epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config['epsilon_start']
        self.time_step = 0  # To track when to update network

        # Q-Network (Policy and Target)
        # TODO: Replace with appropriate network architecture (e.g., GNN) if needed
        self.qnetwork_local = QNetwork(state_size, action_size, config['hidden_dim'])
        self.qnetwork_target = QNetwork(state_size, action_size, config['hidden_dim'])
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['lr'])

        # Initialize target network weights the same as local network
        self._hard_update_target_network()

        # Replay memory
        self.memory = ReplayBuffer(config['buffer_size'], config['batch_size'])

        print(f"Initialized AIAgent.")
        print(f"  State Size: {state_size}")
        print(f"  Action Size: {action_size}")
        print(f"  Config: {config}")

    def _preprocess_state(self, state):
        """Convert state to a PyTorch tensor (add batch dimension if needed)."""
        # TODO: Implement actual state preprocessing (flattening, feature extraction, GNN input prep)
        # Assuming state is already a numpy array of correct size for now
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().unsqueeze(0)  # Add batch dim
        return state

    def select_action(self, state, use_epsilon=True):
        """Uses the current policy to select an action.

        Args:
            state (np.array/torch.tensor): current state from the environment
            use_epsilon (bool): whether to apply epsilon-greedy exploration

        Returns:
            int: chosen action
        """
        state_tensor = self._preprocess_state(state)

        # Put the network in evaluation mode (disables dropout, etc.)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        # Put the network back in training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if use_epsilon and random.random() < self.epsilon:
            # TODO: Ensure random action is valid for the *current* tree/state
            action = random.choice(np.arange(self.action_size))
            # print(f"  (Action: Random - {action})")
        else:
            # Select action with highest Q-value
            action = np.argmax(action_values.cpu().data.numpy())
            # print(f"  (Action: Greedy - {action}, Q-vals: {action_values.cpu().data.numpy()})")

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and trigger learning step."""
        # Save experience
        # TODO: Ensure state and next_state are stored appropriately (e.g., as numpy arrays)
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % self.config['update_every']
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.config['batch_size']:
                experiences = self.memory.sample()
                self._learn(experiences)

    def _learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Q_targets = r + γ * max_a Q_target(s', a)

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # --- Get max predicted Q values (for next states) from target model ---
        # Detach() prevents gradients from flowing into the target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # --- Compute Q targets for current states ---
        # If done is 1, the future reward is 0
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))

        # --- Get expected Q values from local model ---
        # We need Q(s,a) for the *actions actually taken* in the batch
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- Compute loss ---
        loss = F.mse_loss(q_expected, q_targets)

        # --- Minimize the loss ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # --- Update target network (soft update) ---
        self._soft_update_target_network()

        # print(f"    Learned - Loss: {loss.item():.4f}") # Optional: Log loss

    def _hard_update_target_network(self):
        """Copy weights from local network to target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        # print("    Hard updated target network.")

    def _soft_update_target_network(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = self.config['tau']
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)

    def save_model(self, filepath):
        """Save the local Q-network weights."""
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load weights into the local Q-network."""
        self.qnetwork_local.load_state_dict(torch.load(filepath))
        self._hard_update_target_network()  # Also update target network
        print(f"Model loaded from {filepath}")


# --- Example Usage ---
if __name__ == '__main__':
    print("Running Agent Stub Example...")


    class AIAgentTree:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self._state = None
            print("Initialized DummyTreeEnv.")

        def reset(self):
            """
            Initializes the tree with realistic parameters
            """
            self._state = np.array([
                np.random.uniform(1, 5),  # Height in meters
                np.random.randint(1, 10),  # Number of branches
                np.random.randint(10, 100),  # Number of leaves
                np.random.uniform(0.5, 1),  # Health (0 to 1)
                np.random.randint(1, 10)  # Age in years
            ])
            print("TreeEnv Reset. Initial state:", self._state)
            return self._state

    def step(self, action):
        """
        Simulates an action affecting the tree.
        Actions: 0 - Water, 1 - Prune, 2 - Fertilize, 3 - Wait
        """
        if action not in range(self.action_size):
            raise ValueError(f"Invalid action {action} for action_size {self.action_size}")

        height, branches, leaves, health, age = self._state
        reward = 0

        if action == 0:  # Water
            health = min(1.0, health + 0.1)  # Improve health
            reward = 1.0

        elif action == 1:  # Prune
            if branches > 1:
                branches -= 1
                leaves = max(10, leaves - random.randint(5, 20))  # Lose leaves
                health = min(1.0, health + 0.05)  # Slight health boost
                reward = 0.5
            else:
                reward = -0.5  # Can't prune further

        elif action == 2:  # Fertilize
            height += random.uniform(0.2, 0.5)  # Boost growth
            leaves += random.randint(5, 15)  # More leaves
            reward = 1.0

        elif action == 3:  # Wait
            age += 1  # Tree ages
            height += 0.1  # Natural growth
            health = max(0, health - 0.05)  # Health declines without care
            reward = -0.1  # Slight penalty for inaction

        # Determine if tree dies (health = 0)
        done = health <= 0

        self._state = np.array([height, branches, leaves, health, age])

        print(f"Action: {action} -> New State: {self._state}, Reward: {reward}, Done: {done}")
        return self._state, reward, done, {}


    # --- Configuration ---
    config = {
        'state_size': 1,  # TODO: Adjust based on actual state representation
        'action_size': 1,  # TODO: Adjust based on max number of nodes to choose from
        'hidden_dim': 64,
        'buffer_size': int(1e1),  # Replay buffer size
        'batch_size': 32,  # Minibatch size
        'gamma': 0.10,  # Discount factor
        'lr': 5e-4,  # Learning rate
        'tau': 1e-3,  # For soft update of target parameters
        'update_every': 2,  # How often to update the network
        'epsilon_start': 1,  # Starting exploration rate
        'epsilon_end': 0.01,  # Minimum exploration rate
        'epsilon_decay': 0.1  # Epsilon decay rate
    }

    # --- Initialization ---
    env = AIAgentTree(config['state_size'], config['action_size'])
    agent = AIAgent(state_size=config['state_size'], action_size=config['action_size'], config=config)

    # --- Basic Training Loop (Example) ---
    num_episodes = 2  # Run a few dummy episodes
    max_steps_per_episode = 1  # Max actions per episode in this dummy setup

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        print(f"\n--- Episode {i_episode} ---")

        for t in range(max_steps_per_episode):
            # 1. Agent selects action
            action = agent.select_action(state)  # Uses epsilon-greedy

            # 2. Agent performs action in environment
            next_state, reward, done, _ = env.step(action)

            # 3. Agent stores experience and potentially learns
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print(f"  Episode terminated early at step {t + 1}")
                break

        # Decay epsilon after each episode
        agent.decay_epsilon()

        print(f"Episode {i_episode} finished. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    print("\nAgent Stub Example Finished.")
