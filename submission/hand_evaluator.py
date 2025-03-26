class HandEvaluator:
    
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
    
    def evaluate_hand_strength(self, hand):
    
        ranks = sorted([self.get_rank(card) for card in hand], reverse=True)
        suits = [self.get_suit(card) for card in hand]

        # Count occurrences of each rank
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}

        is_flush = len(set(suits)) == 1  # All suits are the same
        is_straight = len(rank_counts) == 5 and max(ranks) - min(ranks) == 4  # Consecutive ranks

        # Assign strength values based on hand type
        if is_straight and is_flush:
            return 0.95 + max(ranks) / 100  # Strongest hand
        elif 4 in rank_counts.values():  # Four of a kind
            return 0.90 + max(ranks) / 100
        elif sorted(rank_counts.values()) == [2, 3]:  # Full house
            return 0.85 + max(ranks) / 100
        elif is_flush:
            return 0.80 + max(ranks) / 100
        elif is_straight:
            return 0.75 + max(ranks) / 100
        elif 3 in rank_counts.values():  # Three of a kind
            return 0.65 + max(ranks) / 100
        elif list(rank_counts.values()).count(2) == 2:  # Two pair
            return 0.52 + max(ranks) / 100
        elif 2 in rank_counts.values():  # One pair
            return 0.35 + max(ranks) / 100
        else:  # High card only
            return 0.1 + max(ranks) / 100

    def hand_strength_preflop(self, cards):
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

    def hand_strength_postflop(self, player_cards, community_cards):
        """
        Calculate hand strength for the modified 27-card deck (2-9, A in diamonds, hearts, spades)
        Uses pronounced differences between hand types
        
        Args:
            player_cards (list): List of player card integers
            community_cards (list): List of community card integers
            
        Returns:
            float: Hand strength between 0 and 1, with wider gaps between hand types


        Straight Flush: 0.99 (fixed value)
        Full House: 0.90 - 0.98
            Base: 0.90
            Formula: 0.90 + (trip_rank / 8) * 0.08
            Example: Full house with 7s over 2s = 0.90 + (5/8) * 0.08 = 0.95
        Flush: 0.80 - 0.89
            Base: 0.80
            Formula: 0.80 + (highest_flush_card / 8) * 0.09
            Example: Flush with Ace high = 0.80 + (8/8) * 0.09 = 0.89
        Straight: 0.70 - 0.79
            Base: 0.70
            Formula: 0.70 + (straight_high_card / 8) * 0.09
            Example: 9-high straight = 0.70 + (7/8) * 0.09 = 0.78
        Three of a Kind: 0.60 - 0.69
            Base: 0.60
            Formula: 0.60 + (trip_rank / 8) * 0.09
            Example: Three 7s = 0.60 + (5/8) * 0.09 = 0.66
        Two Pair: 0.45 - 0.59
            Base: 0.45
            Formula: 0.45 + (top_pair_rank / 8) * 0.10 + (second_pair_rank / 8) * 0.04
            Example: 9s and 7s = 0.45 + (7/8) * 0.10 + (5/8) * 0.04 = 0.54
        One Pair (with player card): 0.30 - 0.42
            Base: 0.30
            Formula: 0.30 + (pair_rank / 8) * 0.12 + kicker_value
            Example: Pair of Aces = 0.30 + (8/8) * 0.12 + kicker = 0.42 + kicker
        One Pair (board only): 0.25 - 0.35
            Base: 0.25
            Formula: 0.25 + (pair_rank / 8) * 0.10 + kicker_value
            Example: Board pair of 2s = 0.25 + (0/8) * 0.10 + kicker = 0.25 + kicker
        High Card: 0.10 - 0.24
            Base: 0.10
            Formula: 0.10 + (highest_card / 8) * 0.14
            Example: Ace high = 0.10 + (8/8) * 0.14 = 0.24
        """
        # If no community cards yet, use basic hand strength
        if not community_cards:
            return self.get_hand_strength(player_cards)
        
        # Extract ranks and suits
        player_ranks = [self.get_rank(card) for card in player_cards]
        player_suits = [self.get_suit(card) for card in player_cards]
        community_ranks = [self.get_rank(card) for card in community_cards]
        community_suits = [self.get_suit(card) for card in community_cards]
        all_ranks = player_ranks + community_ranks
        all_suits = player_suits + community_suits
        
        # Count rank frequencies
        rank_count = {}
        for rank in all_ranks:
            if rank in rank_count:
                rank_count[rank] += 1
            else:
                rank_count[rank] = 1
                
        # Count suit frequencies
        suit_count = {}
        for suit in all_suits:
            if suit in suit_count:
                suit_count[suit] += 1
            else:
                suit_count[suit] = 1

        # For the 27-card deck: check for straights including Ace-low
        def has_straight_in_ranks(ranks):
            # Sort unique ranks
            unique_ranks = sorted(set(ranks))
            
            # Check for regular straights (5 consecutive cards)
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    return True, unique_ranks[i+4]  # Return high card of straight
            
            # Check for Ace-low straight (A,2,3,4,5)
            if 0 in unique_ranks and 1 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 8 in unique_ranks:
                return True, 3  # 5 is the high card (rank 3)
                
            # Check for Ace-high straight (6,7,8,9,A) - only in this modified deck
            if 4 in unique_ranks and 5 in unique_ranks and 6 in unique_ranks and 7 in unique_ranks and 8 in unique_ranks:
                return True, 8  # Ace is the high card (rank 8)
                
            return False, -1
        
        # Check for straight
        has_straight, straight_high = has_straight_in_ranks(all_ranks)
        
        # Check for flush
        flush_suit = -1
        has_flush = False
        for suit, count in suit_count.items():
            if count >= 5:
                has_flush = True
                flush_suit = suit
                break
        
        # Get flush cards if there's a flush
        flush_ranks = []
        if has_flush:
            flush_ranks = [rank for i, rank in enumerate(all_ranks) if all_suits[i] == flush_suit]
        
        # Check for straight flush
        straight_flush_high = -1
        if has_flush and len(flush_ranks) >= 5:
            has_sf, straight_flush_high = has_straight_in_ranks(flush_ranks)
            if has_sf:
                return 0.99  # Straight flush is the best hand
        
        # Check for four of a kind - Not possible in 27-card deck with only 3 suits
        
        # Check for full house (three of a kind + pair)
        trips = [rank for rank, count in rank_count.items() if count >= 3]
        pairs = [rank for rank, count in rank_count.items() if count == 2]
        
        if trips and (len(trips) > 1 or pairs):
            best_trip = max(trips)
            # Scale from 0.90 to 0.98 based on the rank of the trips
            return 0.90 + (best_trip / 8) * 0.08
        
        # Check for flush
        if has_flush:
            # Find the highest card in the flush
            best_flush_card = max(flush_ranks)
            return 0.80 + (best_flush_card / 8) * 0.09  # 0.80-0.89
        
        # Check for straight
        if has_straight:
            # Scale based on high card of straight
            return 0.70 + (straight_high / 8) * 0.09  # 0.70-0.79
        
        # Check for three of a kind
        if trips:
            best_trip = max(trips)
            # Scale from 0.60 to 0.69 based on rank
            return 0.60 + (best_trip / 8) * 0.09
        
        # Check for two pair
        if len(pairs) >= 2:
            sorted_pairs = sorted(pairs, reverse=True)
            top_pair = sorted_pairs[0]
            second_pair = sorted_pairs[1]
            
            # Scale based on ranks of both pairs, with top pair weighted more
            return 0.45 + (top_pair / 8) * 0.10 + (second_pair / 8) * 0.04  # 0.45-0.59
        
        # Check for one pair
        if pairs:
            pair_rank = max(pairs)
            
            # Check if the pair includes player cards
            player_pair = pair_rank in player_ranks
            
            # Find kickers (highest 3 non-paired cards)
            kickers = sorted([r for r in all_ranks if r != pair_rank], reverse=True)[:3]
            kicker_value = 0
            if kickers:
                kicker_value = 0.03 * (kickers[0] / 8)
            
            # Scale pair value based on rank
            # This gives much higher value to high pairs (like 9s, Aces)
            if player_pair:
                base_value = 0.30 + (pair_rank / 8) * 0.12  # 0.30-0.42
            else:
                # Pairs only on board are worth less
                base_value = 0.25 + (pair_rank / 8) * 0.10  # 0.25-0.35
                
            return min(0.44, base_value + kicker_value)
        
        # High card hand
        # Sort all ranks in descending order
        sorted_ranks = sorted(all_ranks, reverse=True)
        
        # Value based on top card, with smaller contributions from other cards
        high_card_value = 0.10 + (sorted_ranks[0] / 8) * 0.14  # 0.10-0.24
        
        return high_card_value
    
    def card_notation_to_int(self, card_str):
        # Convert rank
        rank_char = card_str[0].upper()
        if rank_char == 'A':
            rank = 8  # Ace is 8
        elif rank_char.isdigit():
            rank = int(rank_char) - 2  # 2 becomes 0, 3 becomes 1, etc.
        else:
            raise ValueError(f"Invalid rank: {rank_char}")
        
        # Convert suit
        suit_char = card_str[1].lower()
        if suit_char == 'd':
            suit = 0  # diamonds
        elif suit_char == 'h':
            suit = 1  # hearts
        elif suit_char == 's':
            suit = 2  # spades
        else:
            raise ValueError(f"Invalid suit: {suit_char}")
        
        # Calculate card value: suit * 9 + rank
        return suit * 9 + rank

    def get_strength_from_notation(self, hand_notation, community_notation=""):
        """
        Calculate hand strength using card notation
        
        Args:
            hand_notation (str): Player's hand in notation format, e.g., "As 7d"
            community_notation (str): Community cards in notation format, e.g., "2s 3h 5d"
        
        Returns:
            float: Hand strength
        """
        # Parse hand cards
        hand_cards = [self.card_notation_to_int(card.strip())
                        for card in hand_notation.split()]
        
        # If no community cards, get simple hand strength
        if not community_notation:
            return self.evaluator.get_hand_strength(hand_cards)
        
        # Parse community cards
        community_cards = [self.card_notation_to_int(card.strip()) 
                            for card in community_notation.split()]
        
        # Get after strength (considering community cards)
        return self.get_after_strength(hand_cards, community_cards)

def test_hand_strength():
    evaluator = HandEvaluator()

    
    # Test cases - using string notation for readability and the conversion methods
    test_cases = [
        # (player_cards, community_cards, description)
        ("As Ad", "3h 5d 7s", "Pair of Aces"),
        ("2s 2d", "3h 5d 7s", "Pair of 2s"),
        ("7s 7d", "3h 5d 9s", "Pair of 7s"),
        ("2s 7d", "7h 3h 5d", "Pair of 7s (made on board)"),
        ("2s 7d", "2h 3h 5d", "Pair of 2s (made on board)"),
        ("As 7d", "7h 7s 5d", "Three of a Kind (7s)"),
        ("As 7d", "2h 2s 5d", "Pair of 2s (on board only)"),
        ("As 2d", "3h 4s 5d", "Straight (A-5)"),
        ("6s 7d", "8h 9s Ad", "Straight (6-10/A)"),
        ("As 3s", "5s 7s 9s", "Flush (Spades)"),
        ("2s 2d", "2h 7s 7d", "Full House (2s over 7s)"),
        ("7s 7d", "7h 2s 2d", "Full House (7s over 2s)")
    ]
    
    print("Hand Strength Evaluation Test:\n")
    print(f"{'Hand':<20} {'Strength':<10} {'Description'}")
    print("-" * 50)
    
    for player_cards, community_cards, description in test_cases:
        # Convert string notation to card integers
        p_cards = [evaluator.card_notation_to_int(card) for card in player_cards.split()]
        c_cards = [evaluator.card_notation_to_int(card) for card in community_cards.split()]
        
        # Get hand strength
        strength = evaluator.get_after_strength(p_cards, c_cards)
        
        # Print result
        hand = f"{player_cards} + {community_cards}"
        print(f"{hand:<20} {strength:.4f}     {description}")
    
    # Test special cases
    print("\nSpecial Test Cases:")
    
    # Compare pair of 7s vs pair of 2s
    p_cards1 = [evaluator.card_notation_to_int(card) for card in "2s 7d".split()]
    c_cards1 = [evaluator.card_notation_to_int(card) for card in "7h 3h 5d".split()]
    strength1 = evaluator.get_after_strength(p_cards1, c_cards1)
    
    p_cards2 = [evaluator.card_notation_to_int(card) for card in "2s 7d".split()]
    c_cards2 = [evaluator.card_notation_to_int(card) for card in "2h 3h 5d".split()]
    strength2 = evaluator.get_after_strength(p_cards2, c_cards2)
    
    print(f"Pair of 7s strength: {strength1:.4f}")
    print(f"Pair of 2s strength: {strength2:.4f}")
    print(f"Is 7s > 2s? {strength1 > strength2}")

if __name__ == "__main__":
    test_hand_strength()



    