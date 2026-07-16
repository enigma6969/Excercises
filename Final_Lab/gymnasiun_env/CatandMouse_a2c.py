import os
import gc
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
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
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GridWorldA2C(nn.Module):
    """
    Actor-Critic MLP network tailored for flattened 6D coordinate states.
    Replaces Lapan's Atari Conv2D architecture.
    """
    def __init__(self, input_size=10, n_actions=8):
        super().__init__()

        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        # Actor head: Outputs raw action logits (unnormalized log-probabilities)
        self.actor_head = nn.Linear(64, n_actions)

        # Critic head: Outputs scalar state-value estimate V(s)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


def preprocess_state(state, grid_size=5):
    """
    Normalizes grid coordinates from [0, size-1] to [0.0, 1.0] for stable gradients.
    """
    # normalized_state = torch.as_tensor(state, dtype=torch.float32, device=device)
    # return normalized_state / (grid_size - 1.0)
    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device
    )

    # Normalize coordinates to [0,1]
    coords = state / (grid_size - 1)

    mouse = coords[..., 0:2]
    cat = coords[..., 2:4]
    cheese = coords[..., 4:6]

    # Relative vectors
    mouse_to_cheese = cheese - mouse
    cat_to_mouse = mouse - cat

    # Final state
    augmented_state = torch.cat(
        (
            mouse,
            cat,
            cheese,
            mouse_to_cheese,
            cat_to_mouse,
        ),
        dim=-1,
    )

    return augmented_state


def make_env(size=6):
    """
    Factory function to create a wrapped GridWorldEnv for vectorized execution.
    FlattenObservation converts Dict{'cat': (2,), 'cheese': (2,), 'mouse': (2,)} -> Box(6,).
    """
    def _init():
        env = gym.make("gymnasium_envCatandMouse/GridWorld-v0", size=size)
        env = gym.wrappers.FlattenObservation(env)
        return env
    return _init


def train():

    # Initialize network and optimizer
    net = GridWorldA2C(input_size=10, n_actions=8).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-5)
    writer = SummaryWriter(comment="-cat-mouse-a2c")

    # Initialize N parallel synchronous environments
    envs = gym.vector.SyncVectorEnv([make_env(size=grid_size) for _ in range(num_envs)])
    states, _ = envs.reset(seed=seed)

    # Tracking variables for reporting
    global_step = 0
    completed_episodes = 0
    current_episode_rewards = np.zeros(num_envs)
    recent_rewards = []

    print(f"Starting A2C training across N={num_envs} parallel environments...")

    while global_step < total_steps:
        # Rollout buffers for the n-step trajectory
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        entropies_list = []

        # 1. COLLECT N-STEP ROLLOUT ACROSS PARALLEL ENVIRONMENTS
        for _ in range(n_steps):
            states_tensor = preprocess_state(states, grid_size=grid_size)
            
            # Forward pass through Actor-Critic
            logits, values = net(states_tensor)
            
            # Sample actions using categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            
            # Execute actions in parallel environments
            next_states, rewards, terminations, truncations, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminations, truncations)

            # Record rollout step data
            values_list.append(values.squeeze(-1))
            log_probs_list.append(dist.log_prob(actions))
            entropies_list.append(dist.entropy())
            rewards_list.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
            dones_list.append(torch.as_tensor(dones, dtype=torch.float32, device=device))

            # Update episodic reward trackers
            current_episode_rewards += rewards
            for i in range(num_envs):
                if dones[i]:
                    completed_episodes += 1
                    recent_rewards.append(current_episode_rewards[i])
                    writer.add_scalar("Episode Reward", current_episode_rewards[i], completed_episodes)
                    current_episode_rewards[i] = 0.0

            states = next_states
            global_step += num_envs

        # 2. BOOTSTRAP NEXT STATE VALUE FOR ONGOING EPISODES
        next_states_tensor = preprocess_state(states, grid_size=grid_size)
        with torch.no_grad():
            _, next_values = net(next_states_tensor)
            next_values = next_values.squeeze(-1)

        # 3. COMPUTE DISCOUNTED N-STEP RETURNS (BACKWARDS RECURSION)
        returns = []
        R = next_values
        for step in reversed(range(n_steps)):
            # If step was terminal (done=1.0), mask out future return R and just use immediate reward
            R = rewards_list[step] + gamma * R * (1.0 - dones_list[step])
            returns.insert(0, R)

        # Flatten batch tensors across (n_steps * num_envs)
        returns_t = torch.cat(returns)
        values_t = torch.cat(values_list)
        log_probs_t = torch.cat(log_probs_list)
        entropies_t = torch.cat(entropies_list)

        # 4. Computing ADVANTAGE (R_t - V(s_t)) 
        advantages_t = returns_t - values_t.detach()

        # 5. COMPUTE LOSS OBJECTIVES
        # Policy Loss: Increase log_prob of actions with positive advantage
        loss_policy = -(log_probs_t * advantages_t).mean()
        
        # Value Loss: Mean Squared Error between predicted value and discounted return
        loss_value = F.smooth_l1_loss(values_t, returns_t)
        
        # Entropy Loss: Maximize entropy to encourage exploration
        loss_entropy = -entropies_t.mean()

        total_loss = loss_policy + value_loss_coef * loss_value + entropy_beta * loss_entropy

        # 6. BACKPROPAGATION & GRADIENT CLIPPING
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()

        # Log training progress periodically
        if global_step % 2000 == 0 and len(recent_rewards) > 0:
            avg_reward = np.mean(recent_rewards[-50:])
            print(f"Step: {global_step:6d}/{total_steps} | Episodes: {completed_episodes:4d} | Avg Reward (Last 50): {avg_reward:6.1f} | Value Loss: {loss_value.item():.3f}")
            
        # Save final model weights - Save Periodically

    weights_path = "./cat_mouse_a2c.pt"
    torch.save(net.state_dict(), weights_path)
    writer.close()
    envs.close()
    print("A2C Training completed successfully! Weights saved to:", weights_path)


def test():
    """
    Evaluates the trained A2C policy with human rendering enabled.
    """
    weights_path = "./cat_mouse_a2c.pt"

    print("\nStarting visual evaluation...")
    net = GridWorldA2C(input_size=10, n_actions=8).to(device).eval()
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # Create a single human-rendered environment
    env = gym.make("gymnasium_envCatandMouse/GridWorld-v0", size=grid_size, render_mode="human")
    env = gym.wrappers.FlattenObservation(env)

    for episode in range(1, test_episodes + 1):
        state, _ = env.reset(seed=None)
        done = False
        truncation = False
        episode_reward = 0.0

        while not done and not truncation:
            state_tensor = preprocess_state(state, grid_size=grid_size).unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(state_tensor)
            
            # Select the greedy action (argmax over logits) during evaluation
            action = torch.argmax(logits, dim=-1).item()
            state, reward, done, truncation, _ = env.step(action)
            episode_reward += reward

        print(f"Test Episode {episode} | Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    train_mode = True
    test_episodes = 20
    
    # Hyperparameters
    grid_size = 6
    num_envs = 16          # Number of parallel environments running synchronously
    n_steps = 6            # A2C rollout trajectory length before bootstrapping
    total_steps = 1000000  # Total environmental interactions to train
    gamma = 0.99           # Discount factor
    learning_rate = 5e-4   # Adam learning rate
    entropy_beta = 0.015    # Entropy regularization coefficient for exploration
    value_loss_coef = 0.5  # Weight for value loss in total objective
    clip_grad = 0.5        # Gradient clipping threshold to prevent exploding gradients    
    
    warnings.filterwarnings("ignore", category=UserWarning)

    if train_mode:
        train()
    else:
        test()