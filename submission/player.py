import random
import math
from agents.agent import Agent
from gym_env import PokerEnv
from submission.hand_evaluator import HandEvaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __str__(self):
        return "PlayerAgent"
    
    def __init__(self, stream=None):
        super().__init__(stream)
        # Track if we've already discarded
        self.has_discarded = False
        
        # Initialize hand evaluator
        self.evaluator = HandEvaluator()
        
        # Game state tracking
        self.hand_history = []
        self.hand_count = 0
        
        # Terminal opponent bet tracking with max size 25
        self.our_terminal_bets = []
        self.opp_terminal_bets = []
        self.opp_terminal_bets_max_size = 50
        self.our_terminal_bets_max_size = 50
        self.last_opp_action = None
    
    def compute_average_bet(self, bet_history):
        """Compute the average of opponent bets at terminal states"""
        sum = 0
        count = 0
        for i in range(0, len(bet_history)):
            if(bet_history[i] == -1 and i != 0 and bet_history[i-1] != 2):
                sum += bet_history[i - 1]
                count += 1
            if(i == len(bet_history) - 1):
                sum += bet_history[i]
                count += 1
    
        return sum/count
        
    def should_redraw_preflop(self, cards):
        """Decide whether to redraw a card preflop"""
        # Don't redraw premium hands or pairs        
        ranks = [self.evaluator.get_rank_value(self.evaluator.get_rank(card)) for card in cards]
        
        # Don't break up any pairs
        if ranks[0] == ranks[1]:
            return False, -1
    
            
        # Redraw the lower card
        if ranks[0] < ranks[1]:
            if(ranks[0] < 6):
                return True, 0
            return False, -1
        else:
            if(ranks[1] < 6):
                return True, 1
        
        return False, -1
    
    def should_redraw_postflop(self, hand, community_cards, street):
        """Decide whether to redraw a card postflop"""
        # Don't redraw if we already have a pair
        if self.evaluator.has_pair(hand):
            return False, -1
        
        # Don't redraw strong hands
        if self.evaluator.get_strength_postflop(hand, community_cards)[1] > 3:
            return False, -1
        
        all_cards = hand + community_cards
        
        # Check for pair with community cards
        for i in range(len(hand)):
            for j in range(len(community_cards)):
                if self.evaluator.has_pair([hand[i], community_cards[j]]):
                    # Discard the card that doesn't make the pair
                    return True, 1 - i
        
        # Check if close to a flush
        suits = [self.evaluator.get_suit(card) for card in all_cards]
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
        for suit, count in suit_counts.items():
            if count == 4 and street < 3:
                # Try to complete flush by keeping the card with matching suit
                if self.evaluator.get_suit(hand[0]) == suit:
                    return True, 1 if self.evaluator.get_suit(hand[1]) != suit else -1
                else:
                    return True, 0 if self.evaluator.get_suit(hand[1]) == suit else -1
        
        # Default: discard the lower ranked card
        if self.evaluator.get_rank(hand[0]) > self.evaluator.get_rank(hand[1]):
            return True, 1
        return True, 0
    
    def should_call_all_in(self, hand_strength, is_premium, street, community_cards=[]):
        """Simplified all-in calling decision"""
        # Always call with premium hands
        if is_premium:
            self.logger.info(f"Calling all-in with premium hand")
            return True
        
        # Get average terminal bet for decision making
        avg_terminal_bet = self.compute_average_bet(self.opp_terminal_bets)
        
        # Calculate calling threshold
        if street == 0:  # Preflop
            # Base threshold, adjusted for average terminal bet
            threshold = 0.5
            
            # If average terminal bet is high, opponent may be aggressive
            if avg_terminal_bet > 30:
                # Lower our threshold since opponent might be bluffing more often
                threshold = max(0.3, threshold - (avg_terminal_bet / 200))
                
        else:  # Postflop
            # Postflop thresholds are lower
            threshold = 0.45
            
            # Very strong absolute hands always call
            if hand_strength > 0.6:
                self.logger.info(f"Calling all-in with strong postflop hand: {hand_strength:.2f}")
                return True
                
            # If average terminal bet is high, opponent may be aggressive
            if avg_terminal_bet > 30:
                threshold = max(0.25, threshold - (avg_terminal_bet / 200))
        
        # Log the threshold and decision
        self.logger.info(f"All-in decision: strength={hand_strength:.2f}, threshold={threshold:.2f}, avg_bet={avg_terminal_bet:.2f}")
        
        return hand_strength > threshold
    
    def bet_size_for_strength(self, hand_strength, max_raise):
        """Determine appropriate bet size based on hand strength"""
        # Calculate pot-based sizing instead of max_raise-based
        if hand_strength > 0.6:  # Very strong hand
            return min(int(max_raise * 0.4), max_raise)
        elif hand_strength > 0.5:  # Strong hand
            return min(int(max_raise * 0.3), max_raise)
        elif hand_strength > 0.4:  # Good hand
            return min(int(max_raise * 0.1), max_raise)
        else:  # Weak hand
            return -1
    
    def gtr(self):
        """Get tightness ratio"""
        our_avg = self.compute_average_bet(self.our_terminal_bets)

        opp_avg = self.compute_average_bet(self.opp_terminal_bets)

        if len(self.our_terminal_bets) <= 10 or our_avg < 0.1:  # Very little history
            return 1.0
        return(opp_avg/our_avg) ** (1/3)

    def estimate_opponent_strength(self, opp_bet, street, community_cards=[]):
        """Estimate opponent's hand strength based on their bet"""
        # Calculate average terminal bet for decision making
        avg_opp_terminal_bet = self.compute_average_bet(self.opp_terminal_bets)
        
        # Preflop estimates
        if street == 0:
            if opp_bet > 50:
                # Large preflop bets - assume strong
                return 0.82
            elif opp_bet > 30:
                return 0.58
            elif opp_bet > 10:
                return 0.4
            return 0.25
        
        # Postflop estimates
        else:
            if opp_bet > 50:
                # Large postflop bets - assume stron
                return 0.55
            elif opp_bet > 40:
                return 0.5
            elif opp_bet > 30:
                return 0.45
            elif opp_bet > 20:
                return 0.4
            elif opp_bet > 10:
                return 0.2
            return 0.1
    
    def is_valid_raise(self, min, max, bet):
        return min <= bet and bet <= max

    def act(self, observation, reward, terminated, truncated, info):
        street = observation["street"]
        if self.our_terminal_bets:
            bet_ratio = self.gtr()
        else:
            bet_ratio = 1

        if(street == 0):
            self.opp_terminal_bets.append(-1)
            self.our_terminal_bets.append(-1)

        # Record opponent's terminal bet
        if "opp_bet" in observation:
            opp_terminal_bet = observation["opp_bet"]
            our_terminal_bet = observation["my_bet"]
            
            # If the opponent's last action was a fold, record a 0 bet

            if "opp_last_action" in observation and observation["opp_last_action"] == "FOLD":
                opp_terminal_bet = 0
            
            # Add to our terminal bets array
            self.opp_terminal_bets.append(opp_terminal_bet)
            self.our_terminal_bets.append(our_terminal_bet)

            # Keep array size at most self.opp_terminal_bets_max_size
            if len(self.opp_terminal_bets) > self.opp_terminal_bets_max_size:
                self.opp_terminal_bets.pop(0)
            if len(self.our_terminal_bets) > self.our_terminal_bets_max_size:
                self.our_terminal_bets.pop(0)
            
            # Log the terminal bet and current average
            opp_avg_bet = self.compute_average_bet(self.opp_terminal_bets)
            our_avg_bet = self.compute_average_bet(self.our_terminal_bets)

        self.hand_count += 1
     
        # Extract observation data
        valid_actions = observation["valid_actions"]
        my_cards = observation["my_cards"]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        
        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        max_raise = observation["max_raise"]
        min_raise = observation["min_raise"] if "min_raise" in observation else 1


        # Track opponent's last action
        if "opp_last_action" in observation:
            self.last_opp_action = observation["opp_last_action"]
        
        # Reset discard flag on new hand
        if street == 0:
            self.has_discarded = False

        # Check for all-in
        is_opp_allin = (opp_bet == max_raise and opp_bet > 20)
        
        # Handle discards
        can_discard = valid_actions[action_types.DISCARD.value]
        if can_discard and not self.has_discarded:
            if street == 0:
                should_discard, card_idx = self.should_redraw_preflop(my_cards)
            else:
                should_discard, card_idx = self.should_redraw_postflop(my_cards, community_cards, street)
            
            if should_discard:
                self.has_discarded = True
                return action_types.DISCARD.value, 0, card_idx
      
        # Calculate hand strength
        if street == 0:  # Preflop
            hand_strength = self.evaluator.get_strength_preflop(my_cards)
        else:  # Postflop
            hand_strength, _ = self.evaluator.get_strength_postflop(my_cards, community_cards)
        
        # Estimate opponent's hand strength
        opp_strength = self.estimate_opponent_strength(opp_bet, street, community_cards)
        
        # Decision making logic
        if street == 0:

            if valid_actions[action_types.CALL.value]:
                # print("hand strength ", hand_strength)
                # print("opp strength ", opp_strength)
                # print("bet strength ", bet_ratio)
                # print("cards ", my_cards)
                # print(hand_strength * bet_ratio > (opp_strength - 0.07))
                # quit()
                if hand_strength * bet_ratio > (opp_strength - 0.07):
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1
        else: # post flop

            if(valid_actions[action_types.RAISE.value]):
                if hand_strength * bet_ratio > opp_strength:
                    bet = self.bet_size_for_strength(hand_strength, max_raise)
                    if(bet < min_raise):
                        bet = min_raise

                    return action_types.RAISE.value, bet, -1
                
            #maybe raise anyway if u have a good hand
            if valid_actions[action_types.CALL.value]:
                if hand_strength * bet_ratio > (opp_strength - 0.07):
                    return action_types.CALL.value, 0, -1
                elif valid_actions[action_types.CHECK.value]:
                    return action_types.CHECK.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1
                
        if valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        else:
            return action_types.FOLD.value, 0, -1     
      
    def observe(self, observation, reward, terminated, truncated, info):
        """Track game outcomes and opponent patterns"""
        
        if terminated:
            self.hand_count += 1
            
            # Log opponent stats periodically
            if self.hand_count % 5 == 0:
                all_in_frequency = self.all_in_count / max(1, self.hand_count)
                self.logger.info(f"Opponent stats after {self.hand_count} hands: all_in_freq={all_in_frequency:.2f}")
            
            # Reset for next hand
            self.previous_opp_bet = 0
            self.has_discarded = False
            
            # Update hand history
            self.logger.info(f"Hand completed with reward: {reward}")
            self.hand_history.append({
                'reward': reward,
                'terminal_state': observation,
                'hand_number': self.hand_count
            })
            
            # Keep history manageable
            if len(self.hand_history) > 50:
                self.hand_history.pop(0)
    
    def reset(self):
        """Reset the bot state for a new hand"""
        self.has_discarded = False
        self.previous_opp_bet = 0
        # Note: We don't reset pattern recognition counters here
        # as we want to remember opponent patterns across hands

def get_agent():
    """Required function to create agent instance"""
    return PlayerAgent()