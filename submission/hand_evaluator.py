import random

class HandEvaluator:
    """
    Enhanced hand evaluator for the modified 27-card poker deck
    Includes Monte Carlo simulation for equity calculation
    """
    
    def __init__(self):
        """Initialize the evaluator with card definitions"""
        self.ranks = "23456789A"
        self.suits = "dhs"  # diamonds, hearts, spades
    
    def get_rank(self, card: int) -> int:
        """Extract rank from card value (0-8, where 8 is Ace)"""
        if card == -1:
            return -1
        return card % 9
    
    def get_suit(self, card: int) -> int:
        """Extract suit from card value (0-2, representing diamonds, hearts, spades)"""
        if card == -1:
            return -1
        return card // 9
    
    def get_rank_value(self, rank_index: int) -> int:
        """Convert rank index to numeric value (2-10, where 10 is Ace)"""
        if rank_index == 8:  # Ace
            return 10
        else:
            return rank_index + 2
    
    def is_suited(self, cards):
        """Check if cards are of the same suit"""
        if len(cards) < 2:
            return False
        suit = self.get_suit(cards[0])
        return all(self.get_suit(card) == suit for card in cards)
    
    def has_pair(self, cards):
        """Check if cards contain a pair"""
        if len(cards) < 2:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) < len(ranks)
    
    def get_hand_strength(self, cards):
        """
        Get simplified hand strength value
        Returns a value between 0 and 1, where 1 is the strongest
        """
        if len(cards) < 2:
            return 0.0
        
        # Extract ranks and sort them (high to low)
        ranks = sorted([self.get_rank(card) for card in cards], reverse=True)
        
        # Check for pairs
        if ranks[0] == ranks[1]:
            # Pair strength increases with rank
            pair_rank = ranks[0]
            # Ace pair
            if pair_rank == 8:
                return 0.95  # AA
            # High pairs (99, 88, 77)
            elif pair_rank >= 5:
                return 0.85 + (pair_rank - 5) * 0.03  # 99=0.91, 88=0.88, 77=0.85
            # Medium pairs
            elif pair_rank >= 3:
                return 0.65 + (pair_rank - 3) * 0.05  # 66=0.75, 55=0.70, 44=0.65
            # Low pairs
            else:
                return 0.50 + pair_rank * 0.05  # 33=0.60, 22=0.55
        
        # Check for suited cards
        is_suited = self.is_suited(cards)
        
        # High card hands
        high_card = max(ranks)
        second_card = min(ranks)
        
        # Ace-high hands
        if high_card == 8:  # Ace
            # A9+
            if second_card >= 7:
                return 0.65 + 0.10 * is_suited  # A9=0.65, A9s=0.75
            # A7-A8
            elif second_card >= 5:
                return 0.55 + 0.15 * is_suited  # A7=0.55, A7s=0.70
            # A2-A6
            else:
                return 0.40 + 0.15 * is_suited  # A2=0.40, A2s=0.55
        
        # High connected cards (89, 78, etc.)
        if abs(high_card - second_card) == 1 and high_card >= 6:
            return 0.35 + 0.15 * is_suited  # 89=0.35, 89s=0.50
        
        # Other high cards
        if high_card >= 6:
            return 0.30 + 0.10 * is_suited
        
        # Low cards
        return 0.20 + 0.10 * is_suited
    
    def evaluate_hand(self, hand, community_cards):
        """Evaluate a hand with community cards to get its rank"""
        # Basic implementation - this would be replaced with actual poker hand evaluation
        # Here we just sum up the rank values, but in practice you'd use a proper poker hand evaluator
        total_value = 0
        for card in hand + community_cards:
            if card != -1:
                total_value += self.get_rank_value(self.get_rank(card))
        
        # Add bonus for pairs, three of a kind, etc.
        all_cards = [card for card in hand + community_cards if card != -1]
        ranks = [self.get_rank(card) for card in all_cards]
        rank_counts = {}
        
        for rank in ranks:
            if rank in rank_counts:
                rank_counts[rank] += 1
            else:
                rank_counts[rank] = 1
        
        # Bonuses for pairs, three of a kind, etc.
        for rank, count in rank_counts.items():
            if count == 2:  # Pair
                total_value += 15
            elif count == 3:  # Three of a kind
                total_value += 40
            elif count > 3:  # Four of a kind or better
                total_value += 100
        
        # Bonus for flush
        if len(all_cards) >= 5:
            suits = [self.get_suit(card) for card in all_cards]
            suit_counts = {}
            
            for suit in suits:
                if suit in suit_counts:
                    suit_counts[suit] += 1
                else:
                    suit_counts[suit] = 1
            
            # Check for flush
            for suit, count in suit_counts.items():
                if count >= 5:
                    total_value += 50
        
        return total_value
    
    def calculate_equity(self, my_cards, community_cards, opp_drawn_card=None, opp_discarded_card=None, num_simulations=300):
        """
        Calculate equity through Monte Carlo simulation
        Returns a value between 0 and 1, where 1 means 100% chance of winning
        """
        if opp_drawn_card is None:
            opp_drawn_card = []
        if opp_discarded_card is None:
            opp_discarded_card = []
        
        # Cards that are already shown
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        shown_cards = [card for card in shown_cards if card != -1]
        
        # Cards that could be drawn
        non_shown_cards = [i for i in range(27) if i not in shown_cards]
        
        if len(non_shown_cards) < 2:  # Not enough cards left to simulate
            return 0.5
        
        wins = 0
        
        for _ in range(num_simulations):
            # Sample remaining cards for opponent and community
            drawn_cards = random.sample(non_shown_cards, min(7 - len(community_cards) - len(opp_drawn_card), len(non_shown_cards)))
            
            # Assign cards to opponent and community
            opp_cards = opp_drawn_card.copy()
            remaining_drawn = drawn_cards.copy()
            
            # Fill opponent's hand first
            while len(opp_cards) < 2 and remaining_drawn:
                opp_cards.append(remaining_drawn.pop(0))
            
            # Then fill community cards
            sim_community = community_cards.copy()
            while len(sim_community) < 5 and remaining_drawn:
                sim_community.append(remaining_drawn.pop(0))
            
            # Evaluate both hands
            my_hand_value = self.evaluate_hand(my_cards, sim_community)
            opp_hand_value = self.evaluate_hand(opp_cards, sim_community)
            
            if my_hand_value > opp_hand_value:
                wins += 1
            elif my_hand_value == opp_hand_value:
                wins += 0.5  # Split pot
        
        return wins / num_simulations