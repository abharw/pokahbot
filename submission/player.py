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
    
    def act(self, observation, reward, terminated, truncated, info):
        """
        Determine the action to take based on the current observation
        Returns a tuple of (action_type, raise_amount, card_to_discard)
        """
        # Extract relevant information from observation
        valid_actions = observation["valid_actions"]
        my_cards = observation["my_cards"]
        street = observation["street"]
        opp_bet = observation["opp_bet"]
        max_raise = observation["max_raise"]
        
        # Check if we can discard and should discard
        can_discard = valid_actions[action_types.DISCARD.value]
        
        if can_discard and not self.has_discarded:
            should_discard, card_idx = self.should_redraw(my_cards, street)
            if should_discard:
                self.has_discarded = True
                return action_types.DISCARD.value, 0, card_idx
        
        # Check if hand is premium
        is_premium = self.is_premium_hand(my_cards)
        
        # If we can raise and have a premium hand, go all-in
        if valid_actions[action_types.RAISE.value] and is_premium and opp_bet <= 2:
            return action_types.RAISE.value, max_raise, -1
        
        # If we can call and opponent has bet (and we have a premium hand)
        if valid_actions[action_types.CALL.value] and opp_bet > 2:
            if is_premium:
                return action_types.CALL.value, 0, -1
            else:
                # Only fold if FOLD is valid
                if valid_actions[action_types.FOLD.value]:
                    return action_types.FOLD.value, 0, -1
                # If we can't fold, check
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
        
        # If we can check, do so
        if valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        
        # If we can fold, do so as a last resort
        if valid_actions[action_types.FOLD.value]:
            return action_types.FOLD.value, 0, -1
        
        # If we reach here, something unexpected happened
        # Just take the first valid action
        for i, is_valid in enumerate(valid_actions):
            if is_valid:
                return i, 0, -1
    
    def reset(self):
        """Reset the bot state for a new hand"""
        self.has_discarded = False

def get_agent():
    """Required function to create agent instance"""
    return PlayerAgent()