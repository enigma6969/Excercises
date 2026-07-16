import os
import gc
import torch
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# Import your custom environment from grid_world.py
import gymnasium_envCatandMouse

# PyTorch device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Garbage collection and emptying CUDA cache
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


class policy_network(torch.nn.Module):
    """
    Neural network for policy approximation.
    Modified for N parallel environments: Softmax uses dim=-1 to handle 2D batch matrices (N, output_size).
    """
    def __init__(self, input_size=6, output_size=8):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)  # dim=-1 ensures probabilities sum to 1 across actions for EACH environment
        )

    def forward(self, x):
        return self.FC(x)


def preprocess_state(state, grid_size=5):
    """
    Preprocesses the state by transforming it to a tensor and normalizing grid coordinates.
    Mapping coordinates from [0, size-1] to [0.0, 1.0] stabilizes gradient descent.
    """
    normalized_state = torch.as_tensor(state, dtype=torch.float32, device=device)
    return normalized_state / (grid_size - 1.0)


def compute_returns(rewards):
    """
    Computes discounted returns for a single finished trajectory.
    """
    t_steps = np.arange(len(rewards))
    r = rewards * discount_factor ** t_steps
    r = np.cumsum(r[::-1])[::-1] / discount_factor ** t_steps
    return r


def compute_loss(log_probs, returns):
    """
    Computes the policy gradient loss: -sum(log_prob * G_t).
    """
    loss = []
    for log_prob, ret in zip(log_probs, returns):
        loss.append(-log_prob * ret)
    return torch.stack(loss).sum()


def make_env(size=6):
    """
    Factory function to create a wrapped GridWorldEnv for vectorized execution.
    FlattenObservation converts Dict{'cat': (2,), 'cheese': (2,), 'mouse': (2,)} -> Box(6,).
    """
    def _init():
        env = gym.make("gymnasium_envCatandMouse/GridWorld-v0",
                    size = size)
        env = gym.wrappers.FlattenObservation(env)
        return env
    return _init


def train():
    # Input size = 6 (cat x,y + cheese x,y + mouse x,y)
    # Output size = 8 (8 discrete movement directions)
    policy = policy_network(input_size=6, output_size=8).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # Create N independent synchronous parallel environments
    envs = gym.vector.SyncVectorEnv([make_env(size=grid_size) for _ in range(num_envs)])
    
    # Reset environments; initial states shape: (N, 6)
    states, _ = envs.reset(seed=seed)

    # Maintain independent trajectory buffers for each of the N environments
    log_probs = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]
    
    # Track which environments have finished their episode in the current synchronous rollout
    finished = [False] * num_envs
    batch_losses = []
    batch_rewards = []    
    
    completed_episodes = 0
    print(f"Starting training across N={num_envs} parallel environments...")

    while completed_episodes < num_episodes:
        # 1. Preprocess batch of states -> Shape: (N, 6)
        states_tensor = preprocess_state(states, grid_size=grid_size)
        
        actions_list = []

        # 2. INDEPENDENT FORWARD PASSES: Isolate autograd graphs per environment
        for i in range(num_envs):
            # Pass individual state (1, 6) so env 'i' gets its own dedicated computation graph
            action_prob = policy(states_tensor[i].unsqueeze(0)).squeeze(0)
            
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            
            actions_list.append(action.item())
            
            # ONLY record transitions if environment 'i' has not terminated yet
            if not finished[i]:
                log_probs[i].append(dist.log_prob(action))

        # 3. Step all N environments simultaneously in Gymnasium
        next_states, rewards, terminations, truncations, _ = envs.step(np.array(actions_list))

        # 4. Process transitions individually for each environment
        for i in range(num_envs):
            if not finished[i]:
                episode_rewards[i].append(rewards[i])

                # Check if environment 'i' finished its episode
                if terminations[i] or truncations[i]:
                    finished[i] = True
                    completed_episodes += 1
                    
                    # Compute discounted returns and loss for this completed trajectory
                    returns = compute_returns(episode_rewards[i])
                    loss = compute_loss(log_probs[i], returns)
                    
                    batch_losses.append(loss)
                    batch_rewards.append(sum(episode_rewards[i]))

                    # Clear individual buffers
                    log_probs[i] = []
                    episode_rewards[i] = []

        # 5. SYNCHRONOUS UPDATE: Only step optimizer when all N environments finish an episode
        if all(finished) or completed_episodes >= num_episodes:
            if len(batch_losses) > 0:
                optimizer.zero_grad()
                # Average the policy gradient loss across all N parallel episodes
                total_loss = torch.stack(batch_losses).mean()
                total_loss.backward()
                optimizer.step()

                # Log metrics
                avg_reward = np.mean(batch_rewards)
                writer.add_scalar("Average Batch Reward", avg_reward, completed_episodes)
                
                if completed_episodes%5000 == 0:
                    print(f"Episodes {completed_episodes:4d}/{num_episodes} | Avg Batch Reward: {avg_reward:6.1f}")

                # Reset batch tracking and hard-reset environments for a clean on-policy start
                batch_losses = []
                batch_rewards = []
                finished = [False] * num_envs
                states, _ = envs.reset()
        else:
            states = next_states

    torch.save(policy.state_dict(), weights_path)
    writer.close()
    envs.close()
    print("Training finished. Weights saved to:", weights_path)


def test():
    """
    Evaluates the trained policy on a single environment with human rendering enabled.
    """
    print("\nStarting evaluation...")
    policy = policy_network(input_size=6, output_size=8).to(device).eval()
    policy.load_state_dict(torch.load(weights_path, map_location=device))

    # Create a single wrapped environment for visual testing
    env = gym.make("gymnasium_envCatandMouse/GridWorld-v0",
                    render_mode="human",
                    size = grid_size)
    env = gym.wrappers.FlattenObservation(env)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        truncation = False
        episode_reward = 0

        while not done and not truncation:
            state_tensor = preprocess_state(state, grid_size=grid_size)
            with torch.no_grad():
                action_probs = policy(state_tensor)
            
            # Select the action with the highest probability (Greedy evaluation)
            action = torch.argmax(action_probs, dim=-1).item()
            state, reward, done, truncation, _ = env.step(action)
            episode_reward += reward
        
        print(f"Test Episode {episode} | Reward: {episode_reward}")

    env.close()


if __name__ == '__main__':
    # Hyperparameters
    train_mode = True
    render = not train_mode

    weights_path = './cat_mouse_reinforce.pt'
    
    grid_size = 6
    num_envs = 16           # N independent parallel environments
    num_episodes = 250000 if train_mode else 20  # Sparse reward games require more episodes to converge
    
    discount_factor = 0.98
    learning_rate = 2e-4   # Slightly lower learning rate for stable pursuit-evasion convergence

    warnings.filterwarnings("ignore", category=UserWarning)

    if train_mode:
        train()
    else:
        test()