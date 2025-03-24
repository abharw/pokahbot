# train_model.py
# This script is used for training the poker agent but doesn't go in the submission folder
import os
import numpy as np
import tensorflow as tf
from gym_env import PokerEnv
import random
import time
from collections import deque
import pickle
import argparse

# Import our agent components
from submission.player import PlayerAgent
from submission.model import PolicyNetwork
from submission.poker_utils import encode_game_state, get_hand_strength, get_action_mask

def parse_args():
    parser = argparse.ArgumentParser(description='Train poker agent')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Save model every N episodes')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load existing checkpoint')
    
    return parser.parse_args()

class Experience:
    """Experience replay buffer for reinforcement learning"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

class TrainerAgent(PlayerAgent):
    """Extension of PlayerAgent for training purposes"""
    def __init__(self, model, epsilon=0.1, stream=False):
        super().__init__(stream=stream)
        self.model = model
        self.epsilon = epsilon
        self.current_state = None
        self.last_action = None
        self.training_mode = True
    
    def act(self, observation, reward, terminated, truncated, info):
        """Act with epsilon-greedy exploration"""
        # Encode the state
        my_cards = observation["my_cards"]
        community_cards = observation["community_cards"]
        hand_strength = get_hand_strength(my_cards, community_cards)
        
        state = encode_game_state(
            observation, 
            hand_strength, 
            self.opponent_aggression_factor, 
            self.discard_used
        )
        
        # Store the current state for training
        self.current_state = state
        
        # Get valid actions mask
        valid_actions = observation["valid_actions"]
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        action_mask = get_action_mask(observation)
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            # Random action
            action_type = random.choice(valid_action_indices)
            raise_amount = 0
            card_to_discard = -1
            
            if action_type == PokerEnv.ActionType.RAISE.value:
                raise_amount = random.randint(
                    observation["min_raise"],
                    observation["max_raise"]
                )
            elif action_type == PokerEnv.ActionType.DISCARD.value:
                card_to_discard = random.randint(0, 1)
        else:
            # Model-based action
            try:
                state_tensor = tf.convert_to_tensor(np.array([state]), dtype=tf.float32)
                action_probs = self.model(state_tensor)[0].numpy()
                
                # Apply mask for valid actions
                masked_probs = action_probs[:4] * action_mask[:4]
                
                # If no valid actions after masking, choose randomly
                if np.sum(masked_probs) == 0:
                    action_type = random.choice(valid_action_indices)
                else:
                    # Get action with highest probability among valid actions
                    action_type = np.argmax(masked_probs)
                
                # Determine raise amount if raising
                raise_amount = 0
                if action_type == PokerEnv.ActionType.RAISE.value:
                    # Use the raise sizing output from the network
                    sizing_factor = action_probs[4]
                    
                    # Convert to actual raise amount
                    min_raise = observation["min_raise"]
                    max_raise = observation["max_raise"]
                    raise_range = max_raise - min_raise
                    
                    if raise_range > 0:
                        raise_amount = int(min_raise + sizing_factor * raise_range)
                    else:
                        raise_amount = min_raise
                
                # Determine whether to discard
                card_to_discard = -1
                if PokerEnv.ActionType.DISCARD.value in valid_action_indices and not self.discard_used:
                    discard_prob = action_probs[5]
                    
                    if discard_prob > 0.5:
                        hand_value = [get_hand_strength([my_cards[0]], []), 
                                    get_hand_strength([my_cards[1]], [])]
                        
                        # Discard the weaker card
                        card_to_discard = 0 if hand_value[0] < hand_value[1] else 1
                        action_type = PokerEnv.ActionType.DISCARD.value
                        self.discard_used = True
            except Exception as e:
                print(f"Error in model-based action: {e}")
                # Fallback to random
                action_type = random.choice(valid_action_indices)
                raise_amount = random.randint(
                    observation["min_raise"] if action_type == PokerEnv.ActionType.RAISE.value else 0,
                    observation["max_raise"] if action_type == PokerEnv.ActionType.RAISE.value else 0
                )
                card_to_discard = -1
        
        # Store the action for training
        self.last_action = (action_type, raise_amount, card_to_discard)
        
        return action_type, raise_amount, card_to_discard

def train():
    """Train the poker agent using self-play"""
    args = parse_args()
    
    # Create the model or load from checkpoint
    input_dim = 42  # Encoded game state size
    hidden_layers = [128, 64, 32]
    output_dim = 6  # Number of action types + raise sizing + discard
    
    policy_network = PolicyNetwork(input_dim, hidden_layers, output_dim)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Load checkpoint if specified
    if args.load_checkpoint and os.path.exists(args.load_checkpoint + ".index"):
        policy_network.load_weights(args.load_checkpoint)
        print(f"Loaded model from {args.load_checkpoint}")
    else:
        print("Starting with a new model")
    
    # Create the environment
    env = PokerEnv(num_hands=10)
    
    # Create agents
    agent0 = TrainerAgent(policy_network, epsilon=args.epsilon, stream=False)
    agent1 = TrainerAgent(policy_network, epsilon=args.epsilon, stream=False)
    
    # Experience replay buffer
    replay_buffer = Experience(capacity=100000)
    
    # Training metrics
    rewards_history = []
    win_rates = []
    losses = []
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    for episode in range(args.episodes):
        episode_start = time.time()
        
        # Reset environment
        (obs0, obs1), info = env.reset()
        
        # Reset agent state
        agent0.discard_used = False
        agent1.discard_used = False
        agent0.last_action = None
        agent1.last_action = None
        agent0.current_state = None
        agent1.current_state = None
        
        episode_rewards = [0, 0]
        episode_states = [[], []]
        episode_actions = [[], []]
        episode_done = False
        
        # Hand loop
        while not episode_done:
            # Player 0's turn
            if obs0['acting_agent'] == 0:
                action0 = agent0.act(obs0, episode_rewards[0], False, False, info)
                obs_next, reward, terminated, truncated, info = env.step(action0)
                
                # Store transition
                if agent0.current_state is not None and agent0.last_action is not None:
                    episode_states[0].append(agent0.current_state)
                    episode_actions[0].append(agent0.last_action)
                    episode_rewards[0] += reward
                
                obs0, obs1 = obs_next
                episode_done = terminated or truncated
            
            # Player 1's turn
            elif not episode_done and obs1['acting_agent'] == 1:
                action1 = agent1.act(obs1, episode_rewards[1], False, False, info)
                obs_next, reward, terminated, truncated, info = env.step(action1)
                
                # Store transition
                if agent1.current_state is not None and agent1.last_action is not None:
                    episode_states[1].append(agent1.current_state)
                    episode_actions[1].append(agent1.last_action)
                    episode_rewards[1] += reward
                
                obs0, obs1 = obs_next
                episode_done = terminated or truncated
        
        # Process episode data for both agents
        for i in range(2):
            for t in range(len(episode_states[i])):
                state = episode_states[i][t]
                action = episode_actions[i][t]
                
                # Calculate discounted returns
                G = 0
                discount = 1
                for r in range(t, len(episode_states[i])):
                    G += discount * episode_rewards[i]
                    discount *= args.gamma
                
                # Add to replay buffer
                replay_buffer.add(state, action, G, None, True)
        
        # Update model if enough experiences
        if len(replay_buffer) >= args.batch_size:
            # Sample batch
            batch = replay_buffer.sample(args.batch_size)
            
            # Unpack batch
            states = np.array([exp[0] for exp in batch])
            actions = [exp[1] for exp in batch]
            returns = np.array([exp[2] for exp in batch])
            
            # Convert to tensors
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            
            # Training step
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = policy_network(states_tensor)
                
                # Extract action probabilities and values
                action_probs = predictions[:, :4]
                raise_sizing = predictions[:, 4]
                discard_probs = predictions[:, 5]
                
                # Calculate losses for each output head
                action_mask = np.zeros((args.batch_size, 4), dtype=np.float32)
                for i, action in enumerate(actions):
                    action_type = action[0]
                    if action_type < 4:  # Valid action type
                        action_mask[i, action_type] = 1.0
                
                # Policy loss (cross-entropy for actions)
                log_probs = tf.math.log(action_probs + 1e-10)
                action_loss = -tf.reduce_sum(log_probs * action_mask, axis=1) * returns_tensor
                
                # Raise sizing loss (mean squared error)
                raise_indices = [i for i, action in enumerate(actions) if action[0] == PokerEnv.ActionType.RAISE.value]
                if raise_indices:
                    target_sizes = np.array([actions[i][1] for i in raise_indices], dtype=np.float32)
                    predicted_sizes = tf.gather(raise_sizing, raise_indices)
                    raise_loss = tf.reduce_mean(tf.square(target_sizes - predicted_sizes))
                else:
                    raise_loss = 0.0
                
                # Discard loss (binary cross-entropy)
                discard_indices = [i for i, action in enumerate(actions) if action[0] == PokerEnv.ActionType.DISCARD.value]
                if discard_indices:
                    discard_targets = np.ones(len(discard_indices), dtype=np.float32)
                    predicted_discards = tf.gather(discard_probs, discard_indices)
                    discard_loss = tf.keras.losses.binary_crossentropy(discard_targets, predicted_discards)
                    discard_loss = tf.reduce_mean(discard_loss)
                else:
                    discard_loss = 0.0
                
                # Combine losses
                total_loss = tf.reduce_mean(action_loss) + raise_loss + discard_loss
            
            # Compute gradients and apply
            gradients = tape.gradient(total_loss, policy_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))
            
            # Store loss
            losses.append(float(total_loss))
        
        # Store episode metrics
        rewards_history.append(episode_rewards)
        win_rates.append(1 if episode_rewards[0] > 0 else 0)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean([r[0] for r in rewards_history[-100:]])
            avg_win_rate = np.mean(win_rates[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            
            print(f"Episode {episode+1}/{args.episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Win Rate: {avg_win_rate:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Replay Buffer: {len(replay_buffer)}")
            print(f"  Episode Time: {time.time() - episode_start:.2f}s")
        
        # Checkpoint model
        if (episode + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/poker_model_ep{episode+1}"
            policy_network.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save training metrics
            with open(f"checkpoints/metrics_ep{episode+1}.pkl", "wb") as f:
                pickle.dump({
                    "rewards": rewards_history,
                    "win_rates": win_rates,
                    "losses": losses
                }, f)
    
    # Save final model
    final_path = "submission/model/poker_model"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    policy_network.save_weights(final_path)
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    train()