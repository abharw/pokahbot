class HandEvaluator:
    """
    Simplified hand evaluator for the modified 27-card poker deck
    Focused on quick classification for use against all-in opponents
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