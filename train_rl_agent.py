import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import our poker environment and opponent agent classes.
from gym_env import PokerEnv
from agents.agent import Agent
from agents.test_agents import ProbabilityAgent

# --- Helper Functions for Preprocessing and Equity Calculation ---

def compute_equity(obs, num_simulations=100):
    """
    Compute an approximate equity (win probability) given the observation.
    Uses a Monte Carlo simulation by sampling the missing cards.
    """
    my_cards = obs["my_cards"]
    community_cards = [c for c in obs["community_cards"] if c != -1]
    opp_cards = []
    if obs["opp_discarded_card"] != -1:
        opp_cards.append(obs["opp_discarded_card"])
    if obs["opp_drawn_card"] != -1:
        opp_cards.append(obs["opp_drawn_card"])

    shown_cards = set(my_cards + community_cards + opp_cards)
    deck = list(range(27))  # our deck consists of 27 cards
    non_shown_cards = [card for card in deck if card not in shown_cards]

    wins = 0
    for _ in range(num_simulations):
        opp_needed = 2 - len(opp_cards)
        board_needed = 5 - len(community_cards)
        sample_size = opp_needed + board_needed

        if sample_size > len(non_shown_cards):
            continue

        sample = random.sample(non_shown_cards, sample_size)
        opp_sample = sample[:opp_needed]
        board_sample = sample[opp_needed:]
        opp_full = opp_cards + opp_sample
        community_full = community_cards + board_sample

        my_hand = [PokerEnv.int_to_card(card) for card in my_cards]
        opp_hand = [PokerEnv.int_to_card(card) for card in opp_full]
        board = [PokerEnv.int_to_card(card) for card in community_full]

        evaluator = PokerEnv().evaluator  # temporary evaluator instance
        my_rank = evaluator.evaluate(my_hand, board)
        opp_rank = evaluator.evaluate(opp_hand, board)
        if my_rank < opp_rank:  # lower rank is better in Treys
            wins += 1

    return wins / num_simulations if num_simulations > 0 else 0.0

def preprocess_observation(obs):
    """
    Converts the observation dictionary for one player into a feature tensor.
    Features include:
      - street (normalized by 3)
      - my_cards (2 values, shifted and normalized)
      - community_cards (5 values, shifted and normalized)
      - my_bet and opp_bet (normalized by 100)
      - min_raise and max_raise (normalized by 100)
      - computed equity (a scalar between 0 and 1)
    """
    street = np.array([obs["street"] / 3.0])
    my_cards = np.array([(card + 1) / 28.0 for card in obs["my_cards"]])
    community_cards = np.array([(card + 1) / 28.0 for card in obs["community_cards"]])
    my_bet = np.array([obs["my_bet"] / 100.0])
    opp_bet = np.array([obs["opp_bet"] / 100.0])
    min_raise = np.array([obs["min_raise"] / 100.0])
    max_raise = np.array([obs["max_raise"] / 100.0])
    equity = np.array([compute_equity(obs)])
    features = np.concatenate([street, my_cards, community_cards, my_bet, opp_bet, min_raise, max_raise, equity])
    return torch.tensor(features, dtype=torch.float32)

# --- Define the Policy Network ---

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=3):
        """
        The network uses a shared base and three heads:
          - Action type head: outputs logits over 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise head: outputs logits over 100 values (to be shifted to [1, 100])
          - Discard head: outputs logits over 3 values (mapped to [-1, 0, 1])
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        return action_type_logits, raise_logits, discard_logits

# --- Define the RL Agent using REINFORCE ---

# --- Define the Policy Network with enhancements ---

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=3):
        """
        The network uses a shared base and three heads:
          - Action type head: outputs logits over 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise head: outputs logits over 100 values (to be shifted to [1, 100])
          - Discard head: outputs logits over 3 values (mapped to [-1, 0, 1])
        
        Enhanced with:
          - Deeper architecture
          - Batch normalization
          - Dropout regularization
        """
        super(PolicyNetwork, self).__init__()
        
        # Deeper network with multiple hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second hidden layer (same size)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third hidden layer (double size)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Fourth hidden layer (back to original size)
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(0.2)
        
        # Action heads remain the same
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        # Handle batch dimension for BatchNorm
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if processing single state
            
        # First hidden layer with BatchNorm and Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Fourth hidden layer
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        # Action heads
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        
        return action_type_logits, raise_logits, discard_logits

# --- Define the RL Agent using REINFORCE ---

class RLAgent:
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        # Add L2 regularization via weight_decay parameter
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma = 0.99

    def select_action(self, state, valid_actions, min_raise, max_raise):
        """
        Given the state tensor and valid actions, sample an action tuple.
        Returns:
        (action_type, raise_amount, card_to_discard), log_prob
        """
        # Set to evaluation mode to disable dropout during inference
        self.policy_net.eval()
        
        # Ensure state has batch dimension for BatchNorm
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Forward pass through the network
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        
        # Squeezing output tensors to remove batch dimension for sampling
        action_type_logits = action_type_logits.squeeze(0)
        raise_logits = raise_logits.squeeze(0)
        discard_logits = discard_logits.squeeze(0)
        
        # Create mask for invalid actions (simpler approach)
        action_mask = torch.ones_like(action_type_logits)
        for i in range(len(valid_actions)):
            if not valid_actions[i]:
                action_mask[i] = -1e9
        
        # Apply mask to logits
        masked_logits = action_type_logits + action_mask
        
        # Sample from distributions
        action_type_dist = torch.distributions.Categorical(logits=masked_logits)
        raise_dist = torch.distributions.Categorical(logits=raise_logits)
        discard_dist = torch.distributions.Categorical(logits=discard_logits)
        
        action_type = action_type_dist.sample()
        raise_amount = raise_dist.sample()
        discard_action = discard_dist.sample()
        
        # Get log probabilities
        log_prob = (action_type_dist.log_prob(action_type) +
                    raise_dist.log_prob(raise_amount) +
                    discard_dist.log_prob(discard_action))
        
        # Convert to Python values
        action_type = action_type.item()
        raise_amount = raise_amount.item() + 1
        
        # Ensure raise amount is within valid range
        if action_type == PokerEnv.ActionType.RAISE.value:
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
        else:
            raise_amount = 0
        
        # Handle discard action mapping
        discard_action = discard_action.item() - 1
        if action_type == PokerEnv.ActionType.DISCARD.value:
            if discard_action < 0:
                discard_action = 0  # force a valid index (0 or 1)
        else:
            discard_action = -1
        
        # Set back to training mode
        self.policy_net.train()
        
        return (action_type, raise_amount, discard_action), log_prob

    def update_policy(self, trajectory):
        R = 0
        returns = []
        for _, r in reversed(trajectory):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            returns = returns - returns.mean()
        loss = 0
        for (log_prob, _), R in zip(trajectory, returns):
            loss += -log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

# --- Training Loop with Opponent Agent, CUDA support, and Weight Saving ---

def train_agent(num_episodes=500, save_every=50, weight_path="rl_agent_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = PokerEnv()
    # Feature vector is 13-dimensional.
    agent = RLAgent(input_dim=13)
    agent.policy_net.to(device)
    
    # Use ProbabilityAgent as the opponent.
    opponent_agent = ProbabilityAgent()
    print(f"Using opponent: {opponent_agent.__name__()}")

    for episode in range(num_episodes):
        obs, info = env.reset()
        trajectory = []  # stores (log_prob, reward) for RL agent's actions
        total_reward = 0
        done = False

        # Ensure model is in training mode at the start of each episode
        agent.policy_net.train()

        while not done:
            acting_agent = obs[0]["acting_agent"]
            if acting_agent == 0:
                # RL agent's turn (player 0)
                state = preprocess_observation(obs[0]).to(device)
                valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device=device)
                min_raise_val = obs[0]["min_raise"]
                max_raise_val = obs[0]["max_raise"]
                action, log_prob = agent.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
                our_turn = True
            else:
                # Opponent's turn (player 1) using the provided opponent agent.
                action = opponent_agent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                our_turn = False
                log_prob = None

            obs, reward, done, truncated, info = env.step(action)
            r = reward[0]  # RL agent's reward is at index 0
            total_reward += r
            if our_turn:
                trajectory.append((log_prob, r))

        agent.update_policy(trajectory)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

        # Save weights every 'save_every' episodes.
        if (episode + 1) % save_every == 0:
            torch.save(agent.policy_net.state_dict(), weight_path)
            print(f"Saved weights to {weight_path}")

    # Final weight save.
    torch.save(agent.policy_net.state_dict(), weight_path)
    print(f"Final weights saved to {weight_path}")

if __name__ == "__main__":
    train_agent(num_episodes=500)