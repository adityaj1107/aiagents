# ===========================================================================
# This script provides a structural stub for a Deep Q-Network (DQN) agent
# designed to learn which node in a simulated tree environment to prioritize.
# ===========================================================================

# --- How to Run This Code ---
# 1. Prerequisites:
#    - Python 3.6 or later installed.
#    - PyTorch and NumPy libraries installed.
#      Open your terminal or command prompt and run:
#      pip install torch numpy
#
# 2. Save the Code:
#    - Save this entire code block as a Python file (e.g., `ai_agents.py`).
#
# 3. Execute from Terminal:
#    - Navigate to the directory where you saved the file in your terminal.
#    - Run the script using:
#      python ai_agents.py
#
# 4. Expected Output:
#    - You will see initialization messages for the environment, agent, network, and buffer.
#    - Messages indicating the pre-population of the replay buffer.
#    - Output for each training episode, showing steps taken, actions (random or greedy),
#      rewards received, total episode reward, and the decaying epsilon value.
#    - A final message when the example run completes.
#
# 5. Customization:
#    - Modify the `QNetwork` architecture (especially if using GNNs).
#    - Tune hyperparameters in the `config` dictionary.
#    - Adjust the state/action space sizes in `config`.
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

# To add slight delay for realism

# Define the structure for storing experiences in the replay buffer
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])


# --- Neural Network Definition (Placeholder) ---
class QNetwork(nn.Module):
    """Simple Placeholder Q-Network Model."""

    def __init__(self, state_size, action_size, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
        # print(f"Initialized QNetwork with state_size={state_size}, action_size={action_size}")

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # print(f"Initialized ReplayBuffer with size={buffer_size}, batch_size={batch_size}")

    def add(self, state, action, reward, next_state, done):
        e = Experience(np.array(state), action, reward, np.array(next_state), done)  # Store as numpy arrays
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# --- AI Agent Definition ---
class AIAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config['epsilon_start']
        self.time_step = 0

        # Use CUDA if available, otherwise CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qnetwork_local = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, config['hidden_dim']).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['lr'])

        self._hard_update_target_network()
        self.memory = ReplayBuffer(config['buffer_size'], config['batch_size'])

        print(f"Initialized AIAgent.")
        print(f"  State Size: {state_size}")
        print(f"  Action Size: {action_size}")
        # print(f"  Config: {config}") # Config can be long, optional print

    def _preprocess_state(self, state):
        # Convert state to a PyTorch tensor and send to the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)  # Add batch dim
        else:
            state = state.unsqueeze(0).to(self.device)  # Ensure it has batch dim and is on correct device
        return state

    def select_action(self, state, use_epsilon=True):
        state_tensor = self._preprocess_state(state)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if use_epsilon and random.random() < self.epsilon:
            action_type = "Random"
            action = random.choice(np.arange(self.action_size))
        else:
            action_type = "Greedy"
            action = np.argmax(action_values.cpu().data.numpy())  # Get action from CPU tensor

        # print(f"    Action Selected: {action} ({action_type})") # Verbose action selection
        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % self.config['update_every']
        if self.time_step == 0:
            if len(self.memory) >= self.config['batch_size']:
                experiences = self.memory.sample()
                # Move experiences to the correct device before learning
                experiences_on_device = tuple(exp.to(self.device) for exp in experiences)
                self._learn(experiences_on_device)

    def _learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update_target_network()

        # print(f"      Learned - Loss: {loss.item():.4f}") # Verbose learning step

    def _hard_update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def _soft_update_target_network(self):
        tau = self.config['tau']
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)

    def save_model(self, filepath):
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        # Ensure loading maps to the correct device (CPU or GPU)
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_location))
        self._hard_update_target_network()  # Also update target network
        print(f"Model loaded from {filepath} onto device {self.device}")


class StructuredTree:
    """Generates more structured states, rewards, etc."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # Number of nodes to choose from
        self._state = np.zeros(state_size)
        self.steps_taken = 0
        self.max_steps = 50  # End episode after this many steps if not done
        print(f"Initialized StructuredTree (State Size: {state_size}, Action Size: {action_size})")

    def _generate_state(self):
        # Simulate features: [complexity, urgency, interaction_flag, noise]
        state = np.zeros(self.state_size)
        state[0] = np.clip(self._state[0] * 0.9 + np.random.rand() * 0.1, 0, 1)  # Complexity decays slightly
        state[1] = np.random.rand()  # Urgency changes randomly
        state[2] = 1.0 if np.random.rand() > 0.8 else 0.0  # Interaction flag is sometimes active
        if self.state_size > 3:  # Add noise if state_size allows
            state[3:] = np.random.rand(self.state_size - 3) * 0.1
        return state

    def reset(self):
        # Reset to a random initial state
        self._state = np.random.rand(self.state_size)
        # Ensure urgency and interaction have clear starting points for demo
        self._state[1] = np.random.rand() * 0.5  # Start with lower urgency
        self._state[2] = 0.0  # Start with no interaction flag
        self.steps_taken = 0
        return self._state

    def step(self, action):
        if not (0 <= action < self.action_size):
            raise ValueError(f"Invalid action {action} for action_size {self.action_size}")

        self.steps_taken += 1
        # Simulate reward based on state and action
        reward = -0.05  # Small cost for taking a step

        # Hypothetical logic: Reward choosing node 0 if urgency is high
        if self._state[1] > 0.7 and action == 0:
            reward += 0.8
            # print("  -> Rewarding action 0 (high urgency)")
        # Hypothetical logic: Reward choosing node 1 if interaction flag is set
        elif self._state[2] > 0.5 and action == 1:
            reward += 0.6
            # print("  -> Rewarding action 1 (interaction flag)")
        # Hypothetical: Small penalty for choosing node 2 often
        elif action == 2:
            reward -= 0.1
        else:
            reward += np.random.rand() * 0.1  # Small random positive reward otherwise

        # Generate next state
        next_state = self._generate_state()

        # Update state based on action (simple simulation)
        if action == 0:  # Action 0 might reduce urgency
            next_state[1] *= 0.5
        if action == 1:  # Action 1 resets interaction flag
            next_state[2] = 0.0
        next_state[0] *= (0.95 + action * 0.01)  # Action slightly influences next complexity

        self._state = np.clip(next_state, 0, 1)  # Ensure state values stay in range

        # Determine if episode is done
        done = False
        if reward > 0.7:  # End episode on high reward
            done = True
            # print("  -> High reward achieved, ending episode.")
        elif self.steps_taken >= self.max_steps:  # End episode after max steps
            done = True
            # print("  -> Max steps reached, ending episode.")

        # print(f"  Step {self.steps_taken}: Action={action}, Reward={reward:.2f}, Done={done}")
        # print(f"    Next State: {np.round(self._state, 2)}")
        # time.sleep(0.01) # Optional small delay
        return self._state, reward, done, {}  # info dict


# --- Main Execution Block ---
if __name__ == '__main__':
    print("==============================================")
    print(" Starting AI Agent Stub Execution ")
    print("==============================================")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Location Context: Bengaluru, Karnataka, India")

    # --- Configuration ---
    config = {
        'state_size': 4,  # Size of the state vector
        'action_size': 3,  # Number of nodes/actions
        'hidden_dim': 128,  # Increased hidden layer size
        'buffer_size': int(5e4),  # Increased buffer size
        'batch_size': 128,  # Increased batch size
        'gamma': 0.99,
        'lr': 1e-4,  # Slightly lower learning rate
        'tau': 1e-3,
        'update_every': 4,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.996  # Slower decay
    }

    # --- Initialization ---
    env = StructeredTree(config['state_size'], config['action_size'])
    agent = AIAgent(state_size=config['state_size'], action_size=config['action_size'], config=config)

    # --- Pre-populate Replay Buffer ---
    print("\n--- Pre-populating Replay Buffer ---")
    pre_populate_steps = config['batch_size'] * 5  # Ensure enough samples for initial learning
    state = env.reset()
    for _ in range(pre_populate_steps):
        action = random.choice(np.arange(config['action_size']))  # Take random actions
        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)  # Add directly to buffer
        state = next_state
        if done:
            state = env.reset()
    print(f"Replay buffer pre-populated with {len(agent.memory)} experiences.")

    # --- Training Loop ---
    num_episodes = 50  # Number of training episodes
    max_steps_per_episode = env.max_steps  # Use max_steps from env
    scores = []  # List to keep track of episode scores
    scores_window = deque(maxlen=10)  # For calculating average score over last 10 episodes

    print("\n--- Starting Training Loop ---")
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0

        for t in range(max_steps_per_episode):
            # Agent selects action
            action = agent.select_action(state, use_epsilon=True)

            # Environment steps
            next_state, reward, done, _ = env.step(action)

            # Agent stores experience & learns
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # End of Episode
        scores.append(episode_reward)
        scores_window.append(episode_reward)
        agent.decay_epsilon()  # Decay exploration rate

        avg_score = np.mean(scores_window)
        print(
            f"\rEpisode {i_episode}\tTotal Reward: {episode_reward:.2f}\tAverage Reward (last 10): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}",
            end="")
        if i_episode % 10 == 0:
            print(
                f"\rEpisode {i_episode}\tTotal Reward: {episode_reward:.2f}\tAverage Reward (last 10): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}")  # Print newline every 10 episodes

    print("\n--- Training Finished ---")
    env.close()

    # Optional: Save the trained model
    # agent.save_model("dqn_agent_model.pth")

    print("\n==============================================")
    print(" AI Agent Stub Execution Finished ")
    print("==============================================")
