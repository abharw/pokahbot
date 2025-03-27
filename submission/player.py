import random
from agents.agent import Agent
from gym_env import PokerEnv
from submission.hand_evaluator import HandEvaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"
    
    def __init__(self, stream=None):
        super().__init__(stream)
        # Track if we've already discarded
        self.has_discarded = False
        
        # Initialize hand evaluator
        self.evaluator = HandEvaluator()
        
        # Game state tracking
        self.hand_history = []
        self.opponent_tendencies = {
            'aggression': 0.5,  # Start with neutral assumption
            'call_frequency': 0.5,
            'fold_frequency': 0.3,
        }
    
    def is_premium_hand(self, cards):
        """
        Determine if a hand is premium enough to call an all-in bet
        Only the absolute strongest hands qualify
        """
        # Extract ranks
        ranks = sorted([self.evaluator.get_rank(card) for card in cards])
        
        # Check for pocket pairs
        if ranks[0] == ranks[1]:
            # Pairs of 7s or better
            return ranks[0] >= 5
        
        # Check for A9+
        if 8 in ranks and min(ranks) >= 7:  # At least A9
            # Check if suited
            suits = [self.evaluator.get_suit(card) for card in cards]
            if suits[0] == suits[1]:  # Suited
                return True
                
        # Check for A7+ suited
        if 8 in ranks and min(ranks) >= 5:  # At least A7
            suits = [self.evaluator.get_suit(card) for card in cards]
            if suits[0] == suits[1]:  # Suited
                return True
        
        return False
    
    def should_redraw(self, cards, street):
        """
        Determine if we should redraw a card
        Strategy: Discard lower card unless we have a premium hand
        """
        # Don't redraw if we already have a premium hand
        if self.is_premium_hand(cards):
            return False, -1  # Don't redraw
        
        # Extract ranks
        ranks = [self.evaluator.get_rank(card) for card in cards]
        
        # Check for pairs (don't break up pairs)
        if ranks[0] == ranks[1]:
            return False, -1  # Don't redraw
        
        # If we have an ace, keep it and redraw the other card
        if 8 in ranks:
            return True, 0 if ranks[0] != 8 else 1  # Redraw non-ace
            
        # Redraw lower card
        return True, 0 if ranks[0] < ranks[1] else 1
    
    def log_game_state(self, observation):
        """Log important game state information for debugging and analysis"""
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")
    
    def calculate_pot_odds(self, observation):
        """Calculate pot odds based on current bet sizes"""
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        return pot_odds
    
    def calculate_optimal_bet(self, equity, pot_size, max_raise):
        """Calculate optimal bet size based on equity and pot size"""
        if equity > 0.8:  # Very strong hand
            bet = min(int(pot_size * 0.75), max_raise)
        elif equity > 0.7:  # Strong hand
            bet = min(int(pot_size * 0.5), max_raise)
        elif equity > 0.6:  # Good hand
            bet = min(int(pot_size * 0.33), max_raise)
        else:
            bet = 0  # Don't raise with weaker hands
        
        return bet
    
    def act(self, observation, reward, terminated, truncated, info):
        """
        Determine the action to take based on the current observation
        Returns a tuple of (action_type, raise_amount, card_to_discard)
        """
        # Log game state
        self.log_game_state(observation)
        
        # Extract relevant information from observation
        valid_actions = observation["valid_actions"]
        my_cards = observation["my_cards"]
        community_cards = observation["community_cards"]
        community_cards = [card for card in community_cards if card != -1]
        street = observation["street"]
        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        max_raise = observation["max_raise"]
        min_raise = observation["min_raise"] if "min_raise" in observation else 1
        opp_discarded_card = observation["opp_discarded_card"] if "opp_discarded_card" in observation else -1
        opp_drawn_card = observation["opp_drawn_card"] if "opp_drawn_card" in observation else -1
        
        # Prepare card lists
        opp_discarded_cards = [opp_discarded_card] if opp_discarded_card != -1 else []
        opp_drawn_cards = [opp_drawn_card] if opp_drawn_card != -1 else []
        
        # Calculate equity
        equity = self.evaluator.calculate_equity(
            my_cards, 
            community_cards, 
            opp_drawn_cards, 
            opp_discarded_cards
        )
        
        # Calculate pot odds
        pot_odds = self.calculate_pot_odds(observation)
        
        # Log equity and pot odds for significant decisions
        self.logger.debug(f"Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")
        
        # Check if we can discard and should discard
        can_discard = valid_actions[action_types.DISCARD.value]
        
        if can_discard and not self.has_discarded:
            should_discard, card_idx = self.should_redraw(my_cards, street)
            if should_discard:
                self.has_discarded = True
                self.logger.debug(f"Discarding card {card_idx}: {int_to_card(my_cards[card_idx])}")
                return action_types.DISCARD.value, 0, card_idx
        
        # Decision making based on equity, pot odds, and game state
        pot_size = my_bet + opp_bet
        
        # If we have very strong equity, consider raising
        if valid_actions[action_types.RAISE.value] and equity > 0.7:
            raise_amount = self.calculate_optimal_bet(equity, pot_size, max_raise)
            if raise_amount >= min_raise:
                if raise_amount > 20:  # Only log large raises
                    self.logger.info(f"Large raise to {raise_amount} with equity {equity:.2f}")
                return action_types.RAISE.value, raise_amount, -1
        
        # If our equity exceeds pot odds, call
        if valid_actions[action_types.CALL.value] and equity >= pot_odds:
            return action_types.CALL.value, 0, -1
        
        # Check when possible
        if valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        
        # Fold as last resort
        if valid_actions[action_types.FOLD.value]:
            if opp_bet > 20:  # Only log significant folds
                self.logger.info(f"Folding to large bet of {opp_bet}")
            return action_types.FOLD.value, 0, -1
        
        # If we reach here, something unexpected happened
        # Just take the first valid action
        for i, is_valid in enumerate(valid_actions):
            if is_valid:
                return i, 0, -1
    
    def observe(self, observation, reward, terminated, truncated, info):
        """Track game outcomes to adjust strategy"""
        if terminated and abs(reward) > 20:  # Only log significant hand results
            self.logger.info(f"Significant hand completed with reward: {reward}")
            
            # Update hand history for long-term strategy adaptation
            self.hand_history.append({
                'reward': reward,
                'terminal_state': observation
            })
            
            # Keep history manageable
            if len(self.hand_history) > 50:
                self.hand_history.pop(0)
    
    def reset(self):
        """Reset the bot state for a new hand"""
        self.has_discarded = False

def get_agent():
    """Required function to create agent instance"""
    return PlayerAgent()