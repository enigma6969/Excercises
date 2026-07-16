import os
import gc
import torch
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# PyTorch device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Garbage collection and emptying CUDA cache
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

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

    Args:
        input_size (int): Input size of the network (state space dimension).
        output_size (int): Output size of the network (number of actions.

    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, output_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor that shows the probability of each actions.

        """

        x = self.FC(x)

        return x


def preprocess_state(state):
    """
    Preprocesses the state by transforming it to tensor.

    Args:
        state (numpy.ndarray): Input state from the environment.

    Returns:
        torch.Tensor: State tensor.

    """

    normalized_state = torch.as_tensor(state, dtype=torch.float32, device=device)

    return normalized_state


def compute_returns(rewards):
    """
    Computes discounted returns.

    Args:
        rewards (list): List of rewards for each time step.

    Returns:
        numpy.ndarray: Array of discounted returns.

    """

    t_steps = np.arange(len(rewards))
    r = rewards * discount_factor ** t_steps  # Compute discounted rewards for each time step

    # Compute the discounted cumulative sum in reverse order and then reverse it again to restore the original order.
    r = np.cumsum(r[::-1])[::-1] / discount_factor ** t_steps

    return r


def compute_loss(log_probs, returns):
    """
    Computes the policy gradient loss based on the formula.

    Args:
        log_probs (list): List of log probabilities of selected actions.
        returns (numpy.ndarray): Array of discounted returns.

    Returns:
        torch.Tensor: Computed loss.

    """

    loss = []
    for log_prob, returns in zip(log_probs, returns):
        loss.append(-log_prob * returns)

    return torch.stack(loss).sum()


def train():
    # Define the policy network and its optimizer.
    policy = policy_network(input_size=4, output_size=2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    writer = SummaryWriter()  # Create a SummaryWriter object for writing TensorBoard logs

    # Main training loop that loops over each episode
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed)  # Reset the environment and get the initial state

        # Initialize empty lists to store log probabilities and rewards for the episode
        log_probs = []
        episode_reward = []

        # Loop until the episode is done
        while True:
            state = preprocess_state(state)  # Preprocesses the state to convert it to tensor
            action_probs = policy(state)  # Get action probabilities from the policy network

            # Sample an action from the action probabilities
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            # Compute the log probability of the sampled action
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # Take a step in the environment
            next_state, reward, done, truncation, _ = env.step(action.item())
            episode_reward.append(reward)

            state = next_state  # Update the current state

            if done or truncation:  # if the episode is done or truncated
                returns = compute_returns(
                    episode_reward)  # Compute the returns (discounted sum of rewards) for the episode
                loss = compute_loss(log_probs,
                                    returns)  # Compute the loss for the episode using the log probabilities and returns
                optimizer.zero_grad()  # Zero the gradients of the optimizer
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the parameters of the policy network

                episode_reward = sum(episode_reward)  # Compute the total reward for the episode
                writer.add_scalar("Episode Reward", episode_reward, episode)  # Write the total reward to TensorBoard

                print(f"Episode {episode}, Ep Reward: {episode_reward}")

                break  # Exit the episode loop

    torch.save(policy.state_dict(), weights_path)  # Save the weigts of the policy network to a .pt file
    writer.close()  # Close the SummaryWriter


def test():
    # Define the policy network and set the policy network to evaluation mode to disable gradient computation
    policy = policy_network(input_size=4, output_size=2).to(device).eval()

    # Load the trained weights of the policy network from the specified file
    policy.load_state_dict(torch.load(weights_path))

    # Loop over each episode for testing
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed)  # Reset the environment and get the initial state
        done = False
        truncation = False
        episode_reward = 0

        # Loop until the episode is done or truncated
        while not done and not truncation:
            state = preprocess_state(state)  # Preprocesses the state to convert it to tensor
            action_probs = policy(state)  # Get action probabilities from the policy network
            action = torch.argmax(action_probs, dim=0).item()  # Select the action with the highest probability (argmax)
            state, reward, done, truncation, _ = env.step(action)  # Take a step in the environment
            episode_reward += reward  # Accumulate the reward for the episode

        print(f"Episode {episode}, Ep Reward: {episode_reward}")

    env.close()  # Close the environment (for closing the Pygame rendering window)


if __name__ == '__main__':
    # Parameters
    train_mode = True
    render = not train_mode

    weights_path = './final_weights_N.pt'

    num_episodes = 1000 if train_mode else 3
    discount_factor = 0.99
    learning_rate = 9.8e-4

    # Construct the environment, seed the action space, and wrap the environmet for normalized state returns
    env = gym.make("CartPole-v1", max_episode_steps=500,
                    render_mode="human" if render else None)
    env = gym.wrappers.NormalizeObservation(env)
    env.action_space.seed(seed)
    env.metadata['render_fps'] = 120  # For max frame rate make it 0

    warnings.filterwarnings("ignore", category=UserWarning)

    if train_mode:
        train()
    else:
        test()