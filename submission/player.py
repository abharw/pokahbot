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
        self.opp_action_count = 0
        self.our_action_count = 0
        self.allan_threshold = 0.7
        self.min_actions_for_pattern = 3
    
    def is_premium_hand(self, cards):
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
    
    def should_redraw_preflop(self, cards):

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
    
    def should_redraw_postflop(self, hand, community_cards, street):
        
        if(self.evaluator.has_pair(hand)):
            return False, -1
        
        #first check for superior hands

        all_cards = hand + community_cards

        if(self.evaluator.get_strength_postflop(hand, community_cards)[1] > 3):
            return False, -1
        
        for i in range(len(hand)):
            for j in range(len(community_cards)):
                if(self.evaluator.has_pair([hand[i], community_cards[j]])):
                    if(i == 0):
                        return True, 1
                    return True, 0
                
        # check if close to a flush
        suits = [self.evaluator.get_suit(card) for card in all_cards]
        suit_counts = {}
            
        for suit in suits:
            if suit in suit_counts:                
                suit_counts[suit] += 1
            else:
                suit_counts[suit] = 1
            
        for (suit, count) in suit_counts.items():

            if count == 4 and street < 3:
                if(self.evaluator.get_suit(hand[0]) == suit):
                    if(self.evaluator.get_suit(hand[1]) != suit):
                        return True, 1
                else:
                    if(self.evaluator.get_suit(hand[1]) == 1):
                        return True, 0
                return False, -1
        
        if(self.evaluator.get_rank(hand[0]) > self.evaluator.get_rank(hand[1])):
            return True, 1
        return True, 0

    def calculate_optimal_bet(self, strength, pot_size, max_raise, min_raise):
        # Scale bet size based on hand strength
        if strength > 0.85:  # Very strong hand
            bet = min(int(pot_size * 0.75), max_raise)
        elif strength > 0.70:  # Strong hand
            bet = min(int(pot_size * 0.5), max_raise)
        elif strength > 0.60:  # Good hand
            bet = min(int(pot_size * 0.33), max_raise)
        else:  # Weaker hand
            bet = min_raise  # Minimum raise
        
        return max(bet, min_raise)  # Ensure we meet minimum raise requirements

    def bet_ratio_preflop(self, strength):
        if(strength > .85): return .6
        if(strength > .7): return .45
        if(strength > .5): return .25
        if(strength > .4): return .15 
        return 0

    def bet_ratio_postflop(self, strength, community_cards):
        result = self.evaluator.bet_size_helper(community_cards)
        ratio = strength / result
        return ratio

    def opp_estimate_preflop(self, opp_bet):
        if opp_bet > 50: return .8
        if opp_bet > 30: return .55
        if opp_bet > 10: return .45
        return 0
    
    def opp_estimate_postflop(self, opp_bet):
        if opp_bet > 50: return .7
        if opp_bet > 40: return .6
        if opp_bet > 30: return .5
        if opp_bet > 20: return .3
        if opp_bet > 10: return .2
        return 0

    def act(self, observation, reward, terminated, truncated, info):        
        # Extract relevant information from observation
        valid_actions = observation["valid_actions"]
        my_cards = observation["my_cards"]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        street = observation["street"]
        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        max_raise = observation["max_raise"]
        min_raise = observation["min_raise"] if "min_raise" in observation else 1
        

        # Check if we can discard and should discard
        can_discard = valid_actions[action_types.DISCARD.value]
        
        if can_discard and not self.has_discarded:
            if street == 0:
                should_discard, card_idx = self.should_redraw_preflop(my_cards)
            else:
                should_discard, card_idx = self.should_redraw_postflop(my_cards, community_cards, street)
            if should_discard:
                self.has_discarded = True
                return action_types.DISCARD.value, 0, card_idx
      
        # Calculate current hand strength based on street
        if street == 0:  # Preflop
            hand_strength = self.evaluator.get_strength_preflop(my_cards)
            # hand_ranking = 0  # Not relevant preflop
        else:  # Postflop
            hand_strength, hand_ranking = self.evaluator.get_strength_postflop(my_cards, community_cards)
        
        # PREFLOP ACTIONS #
        if street == 0:
            is_premium = self.is_premium_hand(my_cards)
            opp_strength = self.bet_ratio_preflop(self.opp_estimate_preflop(opp_bet))
            our_strength = self.bet_ratio_preflop(hand_strength)
            # calling ALL IN 
            if opp_bet == max_raise:
                if is_premium:
                    if valid_actions[action_types.CALL.value]:
                        return action_types.CALL.value, 0, -1
                
                # For non-premium hands, use the dynamic threshold based on opponent patterns
                allin_frequency = self.opp_allin_count / max(self.opp_action_count, 1)
                
                # If we have enough data and opponent goes all-in frequently
                if self.opp_action_count >= self.min_actions_for_pattern and allin_frequency > 0.4:
                    # Adjust calling threshold based on all-in frequency
                    adjusted_threshold = max(0.4, 0.8 - allin_frequency * 0.5)
                    
                    # Call with strong non-premium hands against frequent all-in player
                    if hand_strength > adjusted_threshold:
                        if valid_actions[action_types.CALL.value]:
                            return action_types.CALL.value, 0, -1
            # everything else
            if our_strength > opp_strength:
                tentative_raise = int(our_strength * max_raise * .8)
                if min_raise < tentative_raise and tentative_raise < max_raise:
                    if valid_actions[action_types.RAISE.value]:
                        return action_types.RAISE.value, tentative_raise, -1
                    elif valid_actions[action_types.CALL.value]:
                        return action_types.CALL.value, 0, -1
                    elif valid_actions[action_types.CHECK.value]:
                        return action_types.CHECK.value, 0, -1
                    else:
                        return action_types.FOLD.value, 0, -1
                else:
                    if valid_actions[action_types.CALL.value]:
                        return action_types.CALL.value, 0, -1
                    elif valid_actions[action_types.CHECK.value]:
                        return action_types.CHECK.value, 0, -1
                    else:
                        return action_types.FOLD.value, 0, -1 
            elif abs(opp_strength - our_strength) <= .15:
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1  
            elif opp_bet < 10:
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1 
            else:
                if valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1  

        else: #POST FLOP
            opp_strength = self.bet_ratio_postflop(self.opp_estimate_postflop(opp_bet), community_cards)
            our_strength = self.bet_ratio_postflop(hand_strength, community_cards)

            if opp_bet == max_raise and our_strength > opp_strength:
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1 
            # everything else
            if our_strength > opp_strength:
                tentative_raise = int(our_strength * max_raise * .8)
                if min_raise < tentative_raise and tentative_raise < max_raise:
                    if valid_actions[action_types.RAISE.value]:
                        return action_types.RAISE.value, tentative_raise, -1
                    elif valid_actions[action_types.CALL.value]:
                        return action_types.CALL.value, 0, -1
                    elif valid_actions[action_types.CHECK.value]:
                        return action_types.CHECK.value, 0, -1
                    else:
                        return action_types.FOLD.value, 0, -1
                else:
                    if valid_actions[action_types.CALL.value]:
                        return action_types.CALL.value, 0, -1
                    elif valid_actions[action_types.CHECK.value]:
                        return action_types.CHECK.value, 0, -1
                    else:
                        return action_types.FOLD.value, 0, -1 
                    
            elif abs(opp_strength - our_strength) <= .15:
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1  
            elif opp_bet < 10:
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1 
            else:
                if valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1  

        if valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        else:
            return action_types.FOLD.value, 0, -1  
                    
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

        if "opp_action" in observation and observation["opp_action"] is not None:
            self.opp_action_count += 1
        
        # Check if opponent went all-in
        if (observation["opp_action"] == action_types.RAISE.value and 
            observation["opp_bet"] == observation["max_raise"]):
            self.opp_allin_count += 1
            
        # Update the all-in frequency threshold dynamically
        if self.opp_action_count >= 10:  # Need some minimum sample
            allin_frequency = self.opp_allin_count / self.opp_action_count
            
            # If opponent goes all-in often, loosen our calling standards
            if allin_frequency > 0.6:  # They go all-in more than 60% of the time
                self.allin_threshold = 0.4  # Lower threshold means we call more often
            elif allin_frequency > 0.3:  # They go all-in sometimes
                self.allin_threshold = 0.55
            else:  # They rarely go all-in
                self.allin_threshold = 0.7


    def reset(self):
        """Reset the bot state for a new hand"""
        self.has_discarded = False

def get_agent():
    """Required function to create agent instance"""
    return PlayerAgent()