from agents.agent import Agent
from gym_env import PokerEnv
import numpy as np
import tensorflow as tf
import os
import time
import random
from collections import deque

# Import your custom modules
from poker_utils import (
    encode_cards, encode_game_state, get_hand_strength, 
    calculate_pot_odds, get_action_mask
)
from model import PolicyNetwork

# Define action types for cleaner code
action_types = PokerEnv.ActionType


class PlayerAgent(Agent):
    def __name__(self):
        return "PokerRL_Agent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        
        # Game tracking variables
        self.hand_number = 0
        self.opponent_actions_history = deque(maxlen=1000)
        self.my_actions_history = deque(maxlen=1000)
        self.bankroll_history = []
        self.start_time = None
        self.time_per_decision = []
        self.discard_used = False
        self.my_cards_history = []
        
        # Opponent modeling
        self.opponent_fold_frequency = 0.0
        self.opponent_raise_frequency = 0.0
        self.opponent_aggression_factor = 1.0  # Default neutral
        
        # Load the model
        self.initialize_model()
        
        # Time management
        self.max_think_time = 0.1  # Default time to spend thinking (seconds)
        self.time_buffer = 0.5  # Reserved time buffer
        
        # Strategy parameters
        self.exploit_factor = 0.8  # How much to exploit opponent tendencies
        self.bluff_threshold = 0.15  # Probability of bluffing with weak hands
        self.value_bet_threshold = 0.7  # Hand strength for value betting
        
        # Logging setup for debugging
        self.debug_mode = False  # Set to True for verbose logging

    def initialize_model(self):
        """Initialize or load the neural network model"""
        try:
            # Model parameters
            input_dim = 42  # Encoded game state size
            hidden_layers = [128, 64, 32]
            output_dim = 6  # Number of action types + raise sizing
            
            # Create the policy network
            self.model = PolicyNetwork(input_dim, hidden_layers, output_dim)
            
            # Check if we have a pre-trained model to load
            model_path = os.path.join(os.path.dirname(__file__), "model", "poker_model")
            if os.path.exists(model_path + ".index"):
                self.model.load_weights(model_path)
                self.logger.info("Loaded pre-trained model")
            else:
                self.logger.info("Using untrained model")
                
            # Convert to a SavedModel for faster inference
            self.predictor = tf.function(
                self.model.call,
                input_signature=[tf.TensorSpec(shape=(1, input_dim), dtype=tf.float32)]
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            # Fallback to a rule-based system if model loading fails
            self.model = None
            self.logger.info("Using rule-based fallback strategy")

    def act(self, observation, reward, terminated, truncated, info):
        """Determine the best poker action to take based on the current observation"""
        decision_start = time.time()
        
        # Track game progress
        if self.start_time is None:
            self.start_time = time.time()
            
        # Log hand info for important hands
        if observation["street"] == 0 and info.get("hand_number", 0) % 100 == 0:
            elapsed = time.time() - self.start_time
            self.logger.info(f"Hand {info.get('hand_number', 0)}/1000, Time: {elapsed:.1f}s")
            
            # Adjust thinking time based on remaining time
            if info.get("hand_number", 0) > 0:
                remaining_hands = 1000 - info.get("hand_number", 0)
                total_time_left = 1500 - elapsed  # Assuming final phase with 1500s
                if total_time_left > 0 and remaining_hands > 0:
                    self.max_think_time = min(0.5, total_time_left / remaining_hands - self.time_buffer)
        
        # Reset discard flag at the start of each hand
        if observation["street"] == 0 and info.get("hand_number", 0) != self.hand_number:
            self.hand_number = info.get("hand_number", 0)
            self.discard_used = False
            
        # Get the valid actions mask
        valid_actions = observation["valid_actions"]
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Initialize action variables
        action_type = None
        raise_amount = 0
        card_to_discard = -1
        
        # Prepare the game state for the model
        try:
            # Use neural network if available
            if self.model is not None:
                # If we're near the time limit, use the faster rule-based approach
                if time.time() - decision_start < self.max_think_time:
                    # Encode cards and game state
                    my_cards = observation["my_cards"]
                    community_cards = observation["community_cards"]
                    
                    # Store my cards for analysis
                    if observation["street"] == 0:
                        self.my_cards_history.append(my_cards)
                    
                    # Get hand strength
                    hand_strength = get_hand_strength(my_cards, community_cards)
                    
                    # Encode the full game state for the model
                    game_state = encode_game_state(
                        observation, 
                        hand_strength, 
                        self.opponent_aggression_factor, 
                        self.discard_used
                    )
                    
                    # Get action probabilities from the model
                    state_tensor = tf.convert_to_tensor(np.array([game_state]), dtype=tf.float32)
                    action_probs = self.predictor(state_tensor)[0].numpy()
                    
                    # Apply mask for valid actions
                    action_mask = get_action_mask(observation)
                    masked_probs = action_probs * action_mask
                    
                    # If no valid actions after masking, use fallback method
                    if np.sum(masked_probs[:4]) == 0:
                        action_type = self.rule_based_decision(observation, hand_strength)
                    else:
                        # Get action with highest probability among valid actions
                        action_idx = np.argmax(masked_probs[:4])
                        action_type = valid_action_indices[action_idx] if action_idx < len(valid_action_indices) else valid_action_indices[0]
                        
                        # Determine raise amount if raising
                        if action_type == action_types.RAISE.value:
                            # Use the raise sizing output from the network
                            sizing_factor = action_probs[4]  # Position 4 is raise sizing
                            
                            # Convert to actual raise amount
                            min_raise = observation["min_raise"]
                            max_raise = observation["max_raise"]
                            raise_range = max_raise - min_raise
                            
                            if raise_range > 0:
                                # Scale the raise size based on hand strength and model output
                                raise_amount = int(min_raise + sizing_factor * raise_range)
                            else:
                                raise_amount = min_raise
                        
                        # Determine whether to discard
                        if action_types.DISCARD.value in valid_action_indices and not self.discard_used:
                            discard_prob = action_probs[5]  # Position 5 is discard decision
                            
                            # If the model suggests discarding
                            if discard_prob > 0.5 and observation["street"] < 2:
                                hand_value = [get_hand_strength([my_cards[0]], []), 
                                             get_hand_strength([my_cards[1]], [])]
                                
                                # Discard the weaker card
                                card_to_discard = 0 if hand_value[0] < hand_value[1] else 1
                                action_type = action_types.DISCARD.value
                                self.discard_used = True
                else:
                    # Use faster rule-based approach if we're running short on time
                    action_type, raise_amount, card_to_discard = self.quick_decision(observation)
            else:
                # Fallback to rule-based approach if model isn't available
                action_type, raise_amount, card_to_discard = self.rule_based_decision(observation)
        
        except Exception as e:
            self.logger.error(f"Error in act: {e}")
            # Safe fallback - choose a random valid action
            action_type = random.choice(valid_action_indices)
            if action_type == action_types.RAISE.value:
                raise_amount = observation["min_raise"]
            
        # Track decision time
        decision_time = time.time() - decision_start
        self.time_per_decision.append(decision_time)
        
        # Log the action if in debug mode
        if self.debug_mode and observation["street"] == 0:
            self.logger.info(f"Action: {action_type}, Raise: {raise_amount}, Time: {decision_time:.3f}s")
        
        # Store my action
        self.my_actions_history.append((action_type, raise_amount))
        
        return action_type, raise_amount, card_to_discard
    
    def rule_based_decision(self, observation, hand_strength=None):
        """Fallback rule-based strategy"""
        my_cards = observation["my_cards"]
        community_cards = observation["community_cards"]
        
        # Calculate hand strength if not provided
        if hand_strength is None:
            hand_strength = get_hand_strength(my_cards, community_cards)
        
        # Calculate pot odds
        pot_odds = calculate_pot_odds(observation["my_bet"], observation["opp_bet"])
        
        # Get valid actions
        valid_actions = observation["valid_actions"]
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Initialize action variables
        action_type = None
        raise_amount = 0
        card_to_discard = -1
        
        # Consider discard action in early streets
        if action_types.DISCARD.value in valid_action_indices and not self.discard_used and observation["street"] < 2:
            # Evaluate individual card strength
            card1_value = get_hand_strength([my_cards[0]], [])
            card2_value = get_hand_strength([my_cards[1]], [])
            
            # Discard if one card is significantly weaker
            if min(card1_value, card2_value) < 0.3 and abs(card1_value - card2_value) > 0.2:
                card_to_discard = 0 if card1_value < card2_value else 1
                action_type = action_types.DISCARD.value
                self.discard_used = True
                return action_type, raise_amount, card_to_discard
        
        # Decision logic based on hand strength
        if hand_strength > self.value_bet_threshold:  # Strong hand
            if action_types.RAISE.value in valid_action_indices:
                action_type = action_types.RAISE.value
                # Size bet based on hand strength
                min_raise = observation["min_raise"]
                max_raise = observation["max_raise"]
                raise_factor = min(1.0, hand_strength + 0.1)  # Higher raise for stronger hands
                raise_amount = min(max_raise, max(min_raise, int(min_raise + (max_raise - min_raise) * raise_factor)))
            elif action_types.CALL.value in valid_action_indices:
                action_type = action_types.CALL.value
            else:
                action_type = action_types.CHECK.value
        elif hand_strength > 0.4:  # Medium hand
            if random.random() < self.bluff_threshold and action_types.RAISE.value in valid_action_indices:
                # Occasionally raise with medium hands
                action_type = action_types.RAISE.value
                min_raise = observation["min_raise"]
                raise_amount = min_raise
            elif action_types.CALL.value in valid_action_indices and pot_odds <= hand_strength * 1.3:
                # Call if pot odds are favorable
                action_type = action_types.CALL.value
            elif action_types.CHECK.value in valid_action_indices:
                action_type = action_types.CHECK.value
            else:
                action_type = action_types.FOLD.value
        else:  # Weak hand
            if random.random() < self.bluff_threshold * 0.5 and action_types.RAISE.value in valid_action_indices:
                # Occasionally bluff with weak hands
                action_type = action_types.RAISE.value
                min_raise = observation["min_raise"]
                raise_amount = min_raise
            elif action_types.CHECK.value in valid_action_indices:
                action_type = action_types.CHECK.value
            elif action_types.CALL.value in valid_action_indices and pot_odds <= hand_strength * 2:
                # Call only if pot odds are very favorable
                action_type = action_types.CALL.value
            else:
                action_type = action_types.FOLD.value
        
        # Default to a safe action if none was selected
        if action_type is None or action_type not in valid_action_indices:
            if action_types.CHECK.value in valid_action_indices:
                action_type = action_types.CHECK.value
            elif action_types.CALL.value in valid_action_indices:
                action_type = action_types.CALL.value
            else:
                action_type = valid_action_indices[0]  # Select the first valid action
        
        return action_type, raise_amount, card_to_discard
    
    def quick_decision(self, observation):
        """Fast decision-making for time-critical situations"""
        valid_actions = observation["valid_actions"]
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Default to checking/calling when possible
        if action_types.CHECK.value in valid_action_indices:
            return action_types.CHECK.value, 0, -1
        elif action_types.CALL.value in valid_action_indices:
            return action_types.CALL.value, 0, -1
        else:
            # Choose the first valid action
            action_type = valid_action_indices[0]
            raise_amount = observation["min_raise"] if action_type == action_types.RAISE.value else 0
            return action_type, raise_amount, -1

    def observe(self, observation, reward, terminated, truncated, info):
        """Track opponent actions and update opponent model"""
        # Update opponent model based on their actions
        if not terminated and observation["acting_agent"] == 1:  # If it's our turn, opponent just acted
            if info.get("last_action") is not None:
                action_type, raise_amount, _ = info["last_action"]
                self.opponent_actions_history.append((action_type, raise_amount))
                
                # Update opponent tendencies
                if len(self.opponent_actions_history) > 10:
                    actions = [a[0] for a in self.opponent_actions_history]
                    total_actions = len(actions)
                    self.opponent_fold_frequency = actions.count(action_types.FOLD.value) / total_actions
                    self.opponent_raise_frequency = actions.count(action_types.RAISE.value) / total_actions
                    
                    # Calculate aggression factor (raises+bets / calls)
                    raises = actions.count(action_types.RAISE.value)
                    calls = actions.count(action_types.CALL.value)
                    self.opponent_aggression_factor = raises / max(1, calls)  # Avoid division by zero
        
        # Track performance at the end of hands
        if terminated:
            self.bankroll_history.append(reward)
            
            # Log significant wins/losses
            if abs(reward) > 50 and self.debug_mode:
                self.logger.info(f"Hand {self.hand_number} ended with reward: {reward}")
            
            # Performance analysis every 100 hands
            if self.hand_number % 100 == 0 and self.hand_number > 0:
                recent_performance = sum(self.bankroll_history[-100:])
                self.logger.info(f"Last 100 hands performance: {recent_performance}")
                
                # Average decision time
                if len(self.time_per_decision) > 0:
                    avg_time = sum(self.time_per_decision) / len(self.time_per_decision)
                    self.logger.info(f"Average decision time: {avg_time:.3f}s")