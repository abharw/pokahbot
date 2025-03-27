import random

class HandEvaluator:
    
    def __init__(self):
        """Initialize the evaluator with card definitions"""
        self.ranks = "23456789A"
        self.suits = "dhs"  # diamonds, hearts, spades
        # Define hand rankings constants
        self.HAND_RANKINGS = {
            "HIGH_CARD": 0,
            "PAIR": 1,
            "TWO_PAIR": 2,
            "THREE_OF_KIND": 3,
            "STRAIGHT": 4,
            "FLUSH": 5,
            "FULL_HOUSE": 6,
            "STRAIGHT_FLUSH": 7
        }
    
    def get_rank(self, card: int) -> int:
        """Extract rank from card value (0-8, where 8 is Ace)"""
        if card == -1:
            return -1
        return card % 9
    
    def get_suit(self, card: int) -> int:
        if card == -1:
            return -1
        return card // 9
    
    def get_rank_value(self, rank_index: int) -> int:
        if rank_index == 8:  # Ace
            return 10
        else:
            return rank_index + 2
    
    def is_suited(self, cards):
     
        if len(cards) < 2:
            return False
        suit = self.get_suit(cards[0])
        return all(self.get_suit(card) == suit for card in cards)
    
    def has_pair(self, cards):
        if len(cards) < 2:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) < len(ranks)
    
    def get_strength_preflop(self, cards):
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
    
    def get_strength_postflop(self, player_cards, community_cards):
    
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
                return (0.99, self.HAND_RANKINGS["STRAIGHT_FLUSH"])  # Straight flush is the best hand
        
        
        # Check for full house (three of a kind + pair)
        trips = [rank for rank, count in rank_count.items() if count >= 3]
        pairs = [rank for rank, count in rank_count.items() if count == 2]
        
        if trips and (len(trips) > 1 or pairs):
            best_trip = max(trips)
            
            # For the pair part, consider all pairs and other trips
            all_pairs = pairs.copy()
            for trip in trips:
                if trip != best_trip:  # Don't use the main trips as a pair
                    all_pairs.append(trip)
            
            if all_pairs:
                best_pair = max(all_pairs)
                # Scale from 0.90 to 0.98 based on trips and pair ranks
                return (0.90 + (best_trip / 8) * 0.07 + (best_pair / 8) * 0.01, self.HAND_RANKINGS["FULL_HOUSE"])
            else:
                # Just in case there are no other pairs
                return (0.90 + (best_trip / 8) * 0.08, self.HAND_RANKINGS["FULL_HOUSE"])
        
        # Check for flush
        if has_flush:
            # Find the highest card in the flush
            best_flush_card = max(flush_ranks)
            return (0.80 + (best_flush_card / 8) * 0.09, self.HAND_RANKINGS["FLUSH"])  # 0.80-0.89
        
        # Check for straight
        if has_straight:
            # Scale based on high card of straight
            return (0.70 + (straight_high / 8) * 0.09, self.HAND_RANKINGS["STRAIGHT"])  # 0.70-0.79
        
        # Check for three of a kind
        if trips:
            best_trip = max(trips)
            # Scale from 0.60 to 0.69 based on rank
            return (0.60 + (best_trip / 8) * 0.09, self.HAND_RANKINGS["THREE_OF_KIND"])
        
        # Check for two pair
        if len(pairs) >= 2:
            sorted_pairs = sorted(pairs, reverse=True)
            top_pair = sorted_pairs[0]
            second_pair = sorted_pairs[1]
            
            # Scale based on ranks of both pairs, with top pair weighted more
            return (0.45 + (top_pair / 8) * 0.10 + (second_pair / 8) * 0.04, self.HAND_RANKINGS["TWO_PAIR"])  # 0.45-0.59
        
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
                
            return (min(0.44, base_value + kicker_value), self.HAND_RANKINGS["PAIR"])
        
        # High card hand
        # Sort all ranks in descending order
        sorted_ranks = sorted(all_ranks, reverse=True)
        
        # Value based on top card, with smaller contributions from other cards
        high_card_value = 0.10 + (sorted_ranks[0] / 8) * 0.14  # 0.10-0.24
        
        return (high_card_value, self.HAND_RANKINGS["HIGH_CARD"])
    
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
    
    def int_to_card_notation(self, card_int):

        if card_int == -1:
            return "??"
            
        # Extract rank and suit
        rank = card_int % 9
        suit = card_int // 9
        
        # Convert rank
        if rank == 8:
            rank_str = 'A'  # Ace
        else:
            rank_str = str(rank + 2)  # 0 becomes 2, 1 becomes 3, etc.
        
        # Convert suit
        if suit == 0:
            suit_str = 'd'  # diamonds
        elif suit == 1:
            suit_str = 'h'  # hearts
        elif suit == 2:
            suit_str = 's'  # spades
        else:
            suit_str = '?'
            
        return rank_str + suit_str

    def best_and_worst_hands(self, community_cards):
   
        # Convert community cards to integers
        community_ints = [self.card_notation_to_int(card) for card in community_cards]
        
        # Generate all possible remaining cards
        all_cards = []
        for rank in "23456789A":
            for suit in "dhs":
                card = rank + suit
                try:
                    card_int = self.card_notation_to_int(card)
                    if card_int not in community_ints:
                        all_cards.append(card_int)
                except ValueError:
                    # Skip invalid cards
                    continue
        
        best_hand = {"strength": -1, "hole_cards": [], "description": "", "ranking": -1}
        worst_hand = {"strength": 2, "hole_cards": [], "description": "", "ranking": 8}
        
        # Test all possible 2-card combinations as hole cards
        for i in range(len(all_cards)):
            for j in range(i + 1, len(all_cards)):
                hole_cards = [all_cards[i], all_cards[j]]
                
                # Calculate hand strength and ranking
                strength, ranking = self.get_strength_postflop(hole_cards, community_ints)
                
                # Convert hole cards to notation for output
                hole_notation = [self.int_to_card_notation(card) for card in hole_cards]
                
                # Check if this is the best hand so far
                if strength > best_hand["strength"]:
                    best_hand = {
                        "strength": strength,
                        "hole_cards": hole_notation,
                        "description": self.get_hand_description(hole_cards, community_ints),
                        "ranking": ranking
                    }
                
                # Check if this is the worst hand so far
                if strength < worst_hand["strength"]:
                    worst_hand = {
                        "strength": strength,
                        "hole_cards": hole_notation,
                        "description": self.get_hand_description(hole_cards, community_ints),
                        "ranking": ranking
                    }

        return {
            "best_hand": best_hand,
            "worst_hand": worst_hand,
            "community_cards": community_cards
        }

    def get_hand_description(self, player_cards, community_cards):

        # Extract ranks and suits
        all_cards = player_cards + community_cards
        ranks = [self.get_rank(card) for card in all_cards]
        suits = [self.get_suit(card) for card in all_cards]
        
        # Count rank frequencies
        rank_count = {}
        for rank in ranks:
            if rank in rank_count:
                rank_count[rank] += 1
            else:
                rank_count[rank] = 1
                
        # Count suit frequencies
        suit_count = {}
        for suit in suits:
            if suit in suit_count:
                suit_count[suit] += 1
            else:
                suit_count[suit] = 1
        
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
            flush_ranks = [rank for i, rank in enumerate(ranks) if suits[i] == flush_suit]
        
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
        has_straight, straight_high = has_straight_in_ranks(ranks)
        
        # Check for straight flush
        if has_flush and len(flush_ranks) >= 5:
            has_sf, straight_flush_high = has_straight_in_ranks(flush_ranks)
            if has_sf:
                high_card = "A" if straight_flush_high == 8 else str(straight_flush_high + 2)
                suit_name = {"d": "Diamonds", "h": "Hearts", "s": "Spades"}[self.suits[flush_suit]]
                return f"Straight Flush, {high_card}-high ({suit_name})"
        
        # Check for three of a kind + pair = full house
        trips = [rank for rank, count in rank_count.items() if count >= 3]
        pairs = [rank for rank, count in rank_count.items() if count == 2]
        
        if trips and (len(trips) > 1 or pairs):
            best_trip = max(trips)
            trip_rank = "A" if best_trip == 8 else str(best_trip + 2)
            
            # Collect all possible pairs (including those from trips)
            all_pairs = pairs.copy()
            for trip in trips:
                if trip != best_trip:  # Don't count the main trips as a pair
                    all_pairs.append(trip)
            
            if all_pairs:
                best_pair = max(all_pairs)
                pair_rank = "A" if best_pair == 8 else str(best_pair + 2)
            else:
                # This should rarely happen, but just in case
                smallest_pair = min(pairs) if pairs else 0
                pair_rank = "A" if smallest_pair == 8 else str(smallest_pair + 2)
                
            return f"Full House, {trip_rank}s over {pair_rank}s"
        
        # Check for flush
        if has_flush:
            high_card = max(flush_ranks)
            high_card_str = "A" if high_card == 8 else str(high_card + 2)
            suit_name = {"d": "Diamonds", "h": "Hearts", "s": "Spades"}[self.suits[flush_suit]]
            return f"Flush, {high_card_str}-high ({suit_name})"
        
        # Check for straight
        if has_straight:
            if straight_high == 3 and 8 in ranks:  # A-5 straight
                return "Straight, 5-high (A-5)"
            elif straight_high == 8:  # A-high straight
                return "Straight, A-high (9-A)"
            else:
                high_card = straight_high + 2
                return f"Straight, {high_card}-high"
        
        # Check for three of a kind
        if trips:
            trip_rank = max(trips)
            rank_str = "A" if trip_rank == 8 else str(trip_rank + 2)
            return f"Three of a Kind, {rank_str}s"
        
        # Check for two pair
        if len(pairs) >= 2:
            sorted_pairs = sorted(pairs, reverse=True)
            top_pair = sorted_pairs[0]
            second_pair = sorted_pairs[1]
            
            top_str = "A" if top_pair == 8 else str(top_pair + 2)
            second_str = "A" if second_pair == 8 else str(second_pair + 2)
            
            return f"Two Pair, {top_str}s and {second_str}s"
        
        # Check for one pair
        if pairs:
            pair_rank = max(pairs)
            rank_str = "A" if pair_rank == 8 else str(pair_rank + 2)
            return f"Pair of {rank_str}s"
        
        # High card hand
        high_card = max(ranks)
        rank_str = "A" if high_card == 8 else str(high_card + 2)
        return f"High Card, {rank_str}"
    
    def bet_size_helper(self, community_cards):
        """
        Calculate the strength of the best possible hand with given community cards.
        
        Args:
            community_cards: List of integer card values
            
        Returns:
            float: Strength value of best possible hand
        """
        # Fixed: Check if community cards is empty to avoid errors
        if not community_cards:
            return 0.5  # Return a medium strength if no community cards
            
        # Generate all possible remaining cards
        all_cards = []
        for rank_idx in range(9):  # 0-8 for ranks 2-A
            for suit_idx in range(3):  # 0-2 for suits d, h, s
                card_int = suit_idx * 9 + rank_idx
                if card_int not in community_cards:
                    all_cards.append(card_int)
        
        best_hand_strength = -1
        
        # Test all possible 2-card combinations as hole cards
        for i in range(len(all_cards)):
            for j in range(i + 1, len(all_cards)):
                hole_cards = [all_cards[i], all_cards[j]]
                
                # Calculate hand strength and ranking
                strength, _ = self.get_strength_postflop(hole_cards, community_cards)
                
                # Check if this is the best hand so far
                if strength > best_hand_strength:
                    best_hand_strength = strength
        
        return best_hand_strength
    
# Example usage:
# def test_best_worst_hands():
#     evaluator = HandEvaluator()
#     """Test the best_and_worst_hands function with various community card combinations"""
#     test_cases = [
#         ["As", "7d", "2h"],
#         ["9s", "9h", "9d"],
#         ["2s", "3h", "4d", "5s"],
#         ["As", "2d", "3h", "4s", "5d"],
#         ["6d", "7h", "8s", "9d"]
#     ]
    
#     for community_cards in test_cases:
#         result = evaluator.best_and_worst_hands(community_cards)
        
#         print(f"\nCommunity Cards: {', '.join(community_cards)}")
#         print(f"Best Hand: {result['best_hand']['description']} with hole cards {result['best_hand']['hole_cards']}")
#         print(f"Worst Hand: {result['worst_hand']['description']} with hole cards {result['worst_hand']['hole_cards']}")
#         print(f"Best Hand Strength: {result['best_hand']['strength']:.4f}")
#         print(f"Worst Hand Strength: {result['worst_hand']['strength']:.4f}")
#         print(f"Best Hand Ranking: {result['best_hand']['ranking']}")  # Added ranking output
#         print(f"Worst Hand Ranking: {result['worst_hand']['ranking']}")  # Added ranking output

#     print("-------------------------------------------------------------------")
#     # Test Case 1: Flop with high cards
#     print("\nTest Case 1: Flop with high cards")
#     community_cards = ["As", "7d", "2h"]
#     # Player has a decent Ace
#     player_cards_str = ["Ad", "9h"] 
#     player_cards = [evaluator.card_notation_to_int(card) for card in player_cards_str]
    
#     gap = evaluator.bet_size_helper(player_cards, community_cards)
#     print(f"Community Cards: {', '.join(community_cards)}")
#     print(f"Player Cards: {', '.join(player_cards_str)}")
#     print(f"Strength Gap: {gap:.4f}")
    
#     # Test Case 2: Flopped set
#     print("\nTest Case 2: Flopped set")
#     community_cards = ["7s", "7d", "2h"]
#     # Player has pocket 7s (flopped set)
#     player_cards_str = ["7h", "As"] 
#     player_cards = [evaluator.card_notation_to_int(card) for card in player_cards_str]
    
#     gap = evaluator.bet_size_helper(player_cards, community_cards)
#     print(f"Community Cards: {', '.join(community_cards)}")
#     print(f"Player Cards: {', '.join(player_cards_str)}")
#     print(f"Strength Gap: {gap:.4f}")
    
#     # Test Case 3: Weak hand
#     print("\nTest Case 3: Weak hand")
#     community_cards = ["As", "9s", "8s"]
#     # Player has a low unconnected hand
#     player_cards_str = ["2d", "4h"] 
#     player_cards = [evaluator.card_notation_to_int(card) for card in player_cards_str]
    
#     gap = evaluator.bet_size_helper(player_cards, community_cards)
#     print(f"Community Cards: {', '.join(community_cards)}")
#     print(f"Player Cards: {', '.join(player_cards_str)}")
#     print(f"Strength Gap: {gap:.4f}")
    
#     # Test Case 4: Straight draw
#     print("\nTest Case 4: Straight draw")
#     community_cards = ["6s", "7d", "8h", "3d"]
#     # Player has a straight draw
#     player_cards_str = ["9s", "5h"] 
#     player_cards = [evaluator.card_notation_to_int(card) for card in player_cards_str]
    
#     gap = evaluator.bet_size_helper(player_cards, community_cards)
#     print(f"Community Cards: {', '.join(community_cards)}")
#     print(f"Player Cards: {', '.join(player_cards_str)}")
#     print(f"Strength Gap: {gap:.4f}")
    
#     # Test Case 5: Full house
#     print("\nTest Case 5: Full house")
#     community_cards = ["7s", "7d", "9h", "9d", "2s"]
#     # Player has a full house
#     player_cards_str = ["9s", "7h"] 
#     player_cards = [evaluator.card_notation_to_int(card) for card in player_cards_str]
    
#     gap = evaluator.bet_size_helper(player_cards, community_cards)
#     print(f"Community Cards: {', '.join(community_cards)}")
#     print(f"Player Cards: {', '.join(player_cards_str)}")
#     print(f"Strength Gap: {gap:.4f}")

# # Run the test if this script is executed directly
# if __name__ == "__main__":
#     test_best_worst_hands()