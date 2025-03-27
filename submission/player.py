import random
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
        
        # Opponent modeling
        self.hand_count = 0
        self.all_in_count = 0
        self.previous_opp_bet = 0
        self.min_hands_for_pattern = 2
        
        # Bet history tracking
        self.our_bets = []
        self.opp_bets = []
        self.max_history_size = 20
    
    def is_premium_hand(self, cards):
        """Identify premium hands that should always call an all-in"""
        ranks = sorted([self.evaluator.get_rank(card) for card in cards])
        suits = [self.evaluator.get_suit(card) for card in cards]
        is_suited = suits[0] == suits[1]
        
        # Pairs 4s or better (44+)
        if ranks[0] == ranks[1] and ranks[0] >= 2:
            return True
        
        # Any suited Ace
        if 8 in ranks and is_suited:
            return True
        
        # Any A9 or better (A9+)
        if 8 in ranks and min(ranks) >= 7:
            return True
        
        # Connected high cards that are suited (89s+)
        if min(ranks) >= 6 and abs(ranks[0] - ranks[1]) == 1 and is_suited:
            return True
            
        return False
    
    def should_redraw_preflop(self, cards):
        """Decide whether to redraw a card preflop"""
        # Don't redraw premium hands or pairs
        if self.is_premium_hand(cards):
            return False, -1
        
        ranks = [self.evaluator.get_rank(card) for card in cards]
        
        # Don't break up any pairs
        if ranks[0] == ranks[1]:
            return False, -1
        
        # If we have an ace, keep it and redraw the other card
        if 8 in ranks:
            return True, 0 if ranks[0] != 8 else 1
            
        # Redraw the lower card
        return True, 0 if ranks[0] < ranks[1] else 1
    
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
        
        # Get all-in frequency if we have data
        all_in_frequency = 0
        if self.hand_count >= self.min_hands_for_pattern:
            all_in_frequency = self.all_in_count / self.hand_count
        
        # Calculate calling threshold
        if street == 0:  # Preflop
            # Base threshold, adjusted for all-in frequency
            threshold = 0.5 - (all_in_frequency * 0.3)
            
            # More we've seen the opponent go all-in, the more we call
            threshold = max(0.3, min(0.5, threshold))
        else:  # Postflop
            # Postflop thresholds are lower
            threshold = 0.45 - (all_in_frequency * 0.3)
            
            # Very strong absolute hands always call
            if hand_strength > 0.6:
                self.logger.info(f"Calling all-in with strong postflop hand: {hand_strength:.2f}")
                return True
                
            # If we have lots of all-in data, be more aggressive
            if all_in_frequency > 0.4:
                threshold = max(0.25, threshold - 0.1)
        
        # Log the threshold and decision
        self.logger.info(f"All-in decision: strength={hand_strength:.2f}, threshold={threshold:.2f}, freq={all_in_frequency:.2f}")
        
        return hand_strength > threshold
    
    def bet_size_for_strength(self, hand_strength, max_raise, min_raise):
        """Determine appropriate bet size based on hand strength"""
        # Calculate pot-based sizing instead of max_raise-based
        if hand_strength > 0.8:  # Very strong hand
            return min(int(max_raise * 0.4), max_raise)
        elif hand_strength > 0.7:  # Strong hand
            return min(int(max_raise * 0.3), max_raise)
        elif hand_strength > 0.6:  # Good hand
            return min(int(max_raise * 0.2), max_raise)
        elif hand_strength > 0.5:  # Decent hand
            return min(int(max_raise * 0.15), max_raise)
        elif hand_strength > 0.4:  # Marginal hand
            return min(int(max_raise * 0.10), max_raise)
        else:  # Weak hand
            return min_raise
    
    def estimate_opponent_strength(self, opp_bet, street, community_cards=[]):
        """Estimate opponent's hand strength based on their bet"""
        # Calculate all-in frequency if we have data
        all_in_frequency = 0
        if self.hand_count >= self.min_hands_for_pattern:
            all_in_frequency = self.all_in_count / self.hand_count
        
        # Preflop estimates
        if street == 0:
            if opp_bet > 50:
                # Large preflop bets - assume strong unless we've seen all-ins
                base_estimate = 0.65
                if all_in_frequency > 0.3:
                    # Reduce estimated strength for frequent all-in players
                    return max(0.35, base_estimate - all_in_frequency * 0.6)
                return base_estimate
            elif opp_bet > 30:
                return 0.5
            elif opp_bet > 10:
                return 0.4
            return 0.3
        
        # Postflop estimates
        else:
            if opp_bet > 50:
                # Large postflop bets - assume strong unless we've seen all-ins
                base_estimate = 0.55
                if all_in_frequency > 0.3:
                    # Reduce estimated strength for frequent all-in players
                    return max(0.25, base_estimate - all_in_frequency * 0.6)
                return base_estimate
            elif opp_bet > 40:
                return 0.5
            elif opp_bet > 30:
                return 0.4
            elif opp_bet > 20:
                return 0.3
            elif opp_bet > 10:
                return 0.2
            return 0.1
    
    def act(self, observation, reward, terminated, truncated, info):        
        # Extract observation data
        valid_actions = observation["valid_actions"]
        my_cards = observation["my_cards"]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        street = observation["street"]
        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        max_raise = observation["max_raise"]
        min_raise = observation["min_raise"] if "min_raise" in observation else 1
        
        # Reset discard flag on new hand
        if street == 0 and my_bet == 0 and opp_bet <= 2:
            self.has_discarded = False
        
        # Track opponent's bet
        if opp_bet > 0:
            self.opp_bets.append(opp_bet)
            if len(self.opp_bets) > self.max_history_size:
                self.opp_bets.pop(0)
                
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
        is_premium = self.is_premium_hand(my_cards)
        if street == 0:  # Preflop
            hand_strength = self.evaluator.get_strength_preflop(my_cards)
        else:  # Postflop
            hand_strength, hand_ranking = self.evaluator.get_strength_postflop(my_cards, community_cards)
            # Log hand details for debugging
            hand_str = [int_to_card(c) for c in my_cards]
            board_str = [int_to_card(c) for c in community_cards] if community_cards else "[]"
            self.logger.info(f"Street: {street}, Hand: {hand_str}, Board: {board_str}, Strength: {hand_strength:.2f}")
        
        # Handle all-in situations first
        if is_opp_allin:
            if self.should_call_all_in(hand_strength, is_premium, street, community_cards):
                if valid_actions[action_types.CALL.value]:
                    return action_types.CALL.value, 0, -1
            else:
                if valid_actions[action_types.FOLD.value]:
                    return action_types.FOLD.value, 0, -1
        
        # Estimate opponent's hand strength
        opp_strength = self.estimate_opponent_strength(opp_bet, street, community_cards)
        
        # Decision making logic
        if hand_strength > opp_strength + 0.1:  # We're significantly stronger
            # Calculate appropriate bet size
            # Consider pot size when calculating bet amount
            pot_size = my_bet + opp_bet
            bet_amount = self.bet_size_for_strength(hand_strength, max_raise, min_raise)
            
            # Limit raise to pot size for more reasonable bets
            if pot_size > 0 and bet_amount > pot_size * 1.5:
                bet_amount = min(int(pot_size * 1.5), bet_amount)
            
            # Raise if we have a good hand and can make a reasonable bet
            # Adding additional check to avoid raising too frequently
            if hand_strength > 0.45 and bet_amount > min_raise:
                # Only raise with a certain probability based on hand strength
                # This avoids being too predictable
                should_raise = random.random() < (hand_strength - 0.3)  # More likely with stronger hands
                if should_raise and valid_actions[action_types.RAISE.value]:
                    # Ensure our bet isn't too large for the situation
                    # Add a sanity check - don't bet more than 20% of max_raise unless very strong hand
                    if hand_strength < 0.7 and bet_amount > max_raise * 0.2:
                        bet_amount = int(max_raise * 0.2)
                        
                    self.our_bets.append(bet_amount)
                    if len(self.our_bets) > self.max_history_size:
                        self.our_bets.pop(0)
                    return action_types.RAISE.value, bet_amount, -1
            
            # Call if we can't or shouldn't raise
            if valid_actions[action_types.CALL.value]:
                self.our_bets.append(opp_bet)
                if len(self.our_bets) > self.max_history_size:
                    self.our_bets.pop(0)
                return action_types.CALL.value, 0, -1
                
        elif abs(hand_strength - opp_strength) <= 0.1:  # We're close in strength
            # Call if bet is reasonable or we have a decent hand
            if opp_bet < 20 or hand_strength > 0.4:
                if valid_actions[action_types.CALL.value]:
                    self.our_bets.append(opp_bet)
                    if len(self.our_bets) > self.max_history_size:
                        self.our_bets.pop(0)
                    return action_types.CALL.value, 0, -1
                    
        else:  # We're weaker
            # Call small bets with almost any hand
            if opp_bet < 10 and hand_strength > 0.3:
                if valid_actions[action_types.CALL.value]:
                    self.our_bets.append(opp_bet)
                    if len(self.our_bets) > self.max_history_size:
                        self.our_bets.pop(0)
                    return action_types.CALL.value, 0, -1
                    
            # Only call larger bets with good hands
            elif hand_strength > 0.5:
                if valid_actions[action_types.CALL.value]:
                    self.our_bets.append(opp_bet)
                    if len(self.our_bets) > self.max_history_size:
                        self.our_bets.pop(0)
                    return action_types.CALL.value, 0, -1
        
        # Check if possible, otherwise fold
        if valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        else:
            return action_types.FOLD.value, 0, -1
                    
    def observe(self, observation, reward, terminated, truncated, info):
        """Track game outcomes and opponent patterns"""
        
        # Track opponent's betting pattern
        if "opp_bet" in observation and "max_raise" in observation:
            opp_bet = observation["opp_bet"]
            max_raise = observation["max_raise"]
            
            # Track potential all-ins (large bets or actual all-ins)
            if opp_bet > 15 and opp_bet > self.previous_opp_bet:
                if opp_bet == max_raise or opp_bet > max_raise * 0.8:
                    self.all_in_count += 1
                    self.logger.info(f"All-in behavior detected: {opp_bet} chips")
            
            # Store current bet for comparison next time
            self.previous_opp_bet = opp_bet
        
        # Handle end of hand
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