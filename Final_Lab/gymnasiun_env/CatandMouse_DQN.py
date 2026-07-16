import os
import gc
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Import your custom environment from grid_world.py
import gymnasium_envCatandMouse

# PyTorch device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Garbage collection and CUDA cache cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Seed everything for reproducible results
seed = 2024
random.seed(seed)
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GridWorldDQN(nn.Module):
    """
    Deep Q-Network MLP architecture mapping 6D coordinates to 8 action Q-values.
    No Softmax activation is applied to the output layer.
    """
    def __init__(self, input_size=6, n_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Circular FIFO experience replay memory buffer.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.asarray(state),
            np.asarray(action),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state),
            np.asarray(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)


def preprocess_state(state, grid_size=5):
    """
    Normalizes grid coordinates from [0, size-1] to [0.0, 1.0] for neural network processing.
    """
    normalized_state = torch.as_tensor(state, dtype=torch.float32, device=device)
    return normalized_state / (grid_size - 1.0)


def make_env(size=5):
    """
    Factory function creating wrapped GridWorldEnv with FlattenObservation applied.
    """
    def _init():
        env = gym.make("gymnasium_envCatandMouse/GridWorld-v0", size=size)
        env = gym.wrappers.FlattenObservation(env)
        return env
    return _init


def train():

    # Initialize Policy Net, Target Net, and Optimizer
    policy_net = GridWorldDQN(input_size=6, n_actions=8).to(device)
    target_net = GridWorldDQN(input_size=6, n_actions=8).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is frozen during gradient calculation

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(capacity=replay_capacity)
    writer = SummaryWriter(comment="-cat-mouse-dqn")

    # Initialize N parallel synchronous environments
    envs = gym.vector.SyncVectorEnv([make_env(size=grid_size) for _ in range(num_envs)])
    states, _ = envs.reset(seed=seed)

    global_step = 0
    completed_episodes = 0
    current_episode_rewards = np.zeros(num_envs)
    recent_rewards = []

    print(f"Starting DQN training across N={num_envs} parallel environments...")

    while global_step < total_steps:
        # 1. CALCULATE EPSILON FOR CURRENT STEP
        epsilon = np.interp(global_step, [0, eps_decay_steps], [eps_start, eps_end])

        # 2. SELECT ACTIONS USING EPSILON-GREEDY STRATEGY
        states_tensor = preprocess_state(states, grid_size=grid_size)
        actions = []
        with torch.no_grad():
            q_values = policy_net(states_tensor)
            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        for i in range(num_envs):
            if random.random() < epsilon:
                actions.append(random.randint(0, 7))  # Explore: Random action
            else:
                actions.append(greedy_actions[i])     # Exploit: Greedy action

        # 3. STEP PARALLEL ENVIRONMENTS
        next_states, rewards, terminations, truncations, _ = envs.step(np.array(actions))
        dones = np.logical_or(terminations, truncations)

        # 4. STORE TRANSITIONS IN REPLAY BUFFER
        for i in range(num_envs):
            buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            current_episode_rewards[i] += rewards[i]

            if dones[i]:
                completed_episodes += 1
                recent_rewards.append(current_episode_rewards[i])
                writer.add_scalar("Episode Reward", current_episode_rewards[i], completed_episodes)
                current_episode_rewards[i] = 0.0

        states = next_states
        global_step += num_envs

        # 5. PERFORM GRADIENT DESCENT UPDATE (Once buffer is sufficiently full)
        if len(buffer) >= min_replay_size:
            # Sample mini-batch from replay memory
            b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)

            # Convert batch to tensors and normalize coordinates
            states_v = preprocess_state(b_states, grid_size=grid_size)
            next_states_v = preprocess_state(b_next_states, grid_size=grid_size)
            actions_v = torch.tensor(b_actions, dtype=torch.int64, device=device).unsqueeze(-1)
            rewards_v = torch.tensor(b_rewards, dtype=torch.float32, device=device)
            dones_v = torch.tensor(b_dones, dtype=torch.float32, device=device)

            # Compute current Q-value predictions: Q(s, a; theta)
            current_q_values = policy_net(states_v).gather(1, actions_v).squeeze(-1)

            # Compute target Q-values using Target Net: r + gamma * max_a' Q(s', a'; theta^-)
            with torch.no_grad():
                next_state_actions = target_net(next_states_v)
                max_next_q = next_state_actions.max(1)[0]
                target_q_values = rewards_v + gamma * max_next_q * (1.0 - dones_v)

            # Compute Mean Squared Error or Smooth L1 Loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar("Loss/SmoothL1", loss.item(), global_step)

            # 6. SYNCHRONIZE TARGET NETWORK
            if global_step % target_sync_frequency < num_envs:
                target_net.load_state_dict(policy_net.state_dict())

        # Periodic progress logging
        if global_step % 5000 < num_envs and len(recent_rewards) > 0:
            avg_reward = np.mean(recent_rewards[-50:])
            print(f"Step: {global_step:6d}/{total_steps} | Episodes: {completed_episodes:4d} | Epsilon: {epsilon:.2f} | Avg Reward (Last 50): {avg_reward:6.1f}")

    # Save trained policy weights
    weights_path = "./cat_mouse_dqn.pt"
    torch.save(policy_net.state_dict(), weights_path)
    writer.close()
    envs.close()
    print("DQN Training completed successfully! Weights saved to:", weights_path)


def test():
    """
    Evaluates the trained DQN agent with human rendering enabled.
    """

    weights_path = "./cat_mouse_dqn.pt"

    print("\nStarting visual evaluation...")
    policy_net = GridWorldDQN(input_size=6, n_actions=8).to(device).eval()
    policy_net.load_state_dict(torch.load(weights_path, map_location=device))

    # Create a single human-rendered environment
    env = gym.make("gymnasium_envCatandMouse/GridWorld-v0", size=grid_size, render_mode="human")
    env = gym.wrappers.FlattenObservation(env)

    for episode in range(1, test_episodes + 1):
        state, _ = env.reset(seed=None)
        done = False
        truncation = False
        episode_reward = 0

        while not done and not truncation:
            state_tensor = preprocess_state(state, grid_size=grid_size).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            
            # Pure greedy evaluation: Select action with highest Q-value
            action = torch.argmax(q_values, dim=-1).item()
            state, reward, done, truncation, _ = env.step(action)
            episode_reward += reward

        print(f"Test Episode {episode} | Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    train_mode = True

    test_episodes = 20
    # Hyperparameters
    grid_size = 6
    num_envs = 8                  # Number of parallel environments to collect data faster
    replay_capacity = 100000       # Maximum transitions stored in memory
    min_replay_size = 1000        # Minimum transitions before gradient updates begin
    batch_size = 256              # Mini-batch size for training
    total_steps = 150000          # Total environment steps to run
    gamma = 0.99                  # Discount factor
    learning_rate = 1e-4          # Adam optimizer learning rate
    target_sync_frequency = 2000  # Steps between copying policy net to target net
    
    # Epsilon-greedy exploration decay parameters
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 50000
    
    warnings.filterwarnings("ignore", category=UserWarning)

    if train_mode:
        train()
    else:
        test()