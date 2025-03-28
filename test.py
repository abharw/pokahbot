from submission.hand_evaluator import HandEvaluator
def get_strength_preflop(evaluator, cards):
    if len(cards) < 2:
        return 0.0
    
    # Extract ranks and sort them (high to low)
    ranks = sorted([evaluator.get_rank(card) for card in cards], reverse=True)
    
    # Check for pairs
    if ranks[0] == ranks[1]:
        # Pair strength increases with rank
        pair_rank = ranks[0]
        # Ace pair
        if pair_rank == 8:
            return 0.95  # AA
            # High pairs (99, 88, 77)
        else: return 0.95 + ((pair_rank - 10) * 0.042)
    
    # Check for suited cards
    is_suited = evaluator.is_suited(cards)
    
    # High card hands
    high_card = max(ranks)
    second_card = min(ranks)
    
    # Ace-high hands
    if high_card == 8:  # Ace
            # A9+
            if second_card >= 7:
                return 0.55 + 0.10 * is_suited  # A9=0.65, A9s=0.75
            # A7-A8
            elif second_card >= 5:
                return 0.5 + 0.10 * is_suited  # A7=0.55, A7s=0.70
            # A2-A6
            else:
                return 0.40 + 0.12 * is_suited  # A2=0.40, A2s=0.55
        
        # High connected cards (89, 78, etc.)
    if abs(high_card - second_card) == 1 and high_card >= 6:
            return 0.32 + 0.10 * is_suited  # 89=0.35, 89s=0.50
        
        # Other high cards
    if high_card >= 6:
            return 0.3 + 0.05 * is_suited
        
        # Low cards
    return 0.20 + 0.05 * is_suited

def get_strength_postflop(evaluator, player_cards, community_cards):

    # Extract ranks and suits
    player_ranks = [evaluator.get_rank(card) for card in player_cards]
    player_suits = [evaluator.get_suit(card) for card in player_cards]
    community_ranks = [evaluator.get_rank(card) for card in community_cards]
    community_suits = [evaluator.get_suit(card) for card in community_cards]
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
            return (0.99, evaluator.HAND_RANKINGS["STRAIGHT_FLUSH"])  # Straight flush is the best hand
    
    
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
            return (0.90 + (best_trip / 8) * 0.07 + (best_pair / 8) * 0.01, evaluator.HAND_RANKINGS["FULL_HOUSE"])
        else:
            # Just in case there are no other pairs
            return (0.90 + (best_trip / 8) * 0.08, evaluator.HAND_RANKINGS["FULL_HOUSE"])
    
    # Check for flush
    if has_flush:
        # Find the highest card in the flush
        best_flush_card = max(flush_ranks)
        return (0.80 + (best_flush_card / 8) * 0.09, evaluator.HAND_RANKINGS["FLUSH"])  # 0.80-0.89
    
    # Check for straight
    if has_straight:
        # Scale based on high card of straight
        return (0.70 + (straight_high / 8) * 0.09, evaluator.HAND_RANKINGS["STRAIGHT"])  # 0.70-0.79
    
    # Check for three of a kind
    if trips:
        best_trip = max(trips)
        # Scale from 0.60 to 0.69 based on rank
        return (0.60 + (best_trip / 8) * 0.09, evaluator.HAND_RANKINGS["THREE_OF_KIND"])
    
    # Check for two pair
    if len(pairs) >= 2:
        sorted_pairs = sorted(pairs, reverse=True)
        top_pair = sorted_pairs[0]
        second_pair = sorted_pairs[1]
        
        # Scale based on ranks of both pairs, with top pair weighted more
        return (0.45 + (top_pair / 8) * 0.10 + (second_pair / 8) * 0.04, evaluator.HAND_RANKINGS["TWO_PAIR"])  # 0.45-0.59
    
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
            
        return (min(0.44, base_value + kicker_value), evaluator.HAND_RANKINGS["PAIR"])
    
    # High card hand
    # Sort all ranks in descending order
    sorted_ranks = sorted(all_ranks, reverse=True)
    
    # Value based on top card, with smaller contributions from other cards
    high_card_value = 0.10 + (sorted_ranks[0] / 8) * 0.14  # 0.10-0.24
    
    return (high_card_value, evaluator.HAND_RANKINGS["HIGH_CARD"])

    # For the 27-card deck: check for straights including Ace-low
"""HELPER IGNORE"""
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

if __name__ == "__main__":
    eval = HandEvaluator()

    print(get_strength_postflop(eval, [4, 5], [6, 2, 3]))
    print(get_strength_postflop(eval, [8, 5], [8, 15, 16]))


    #one mid pair = 0.4
    #high one pair = 0.44
    #mid 2 pair = 0.515
    #good 2 pair = 0.57
    #set on the board  = 0.645
    #2 on board, one in hand set = 0.645
    #really high set = 0.68
    #straight (mid) = 0.77
    #straight (high) = 0.79
    #flush = 0.88
    # straight flush = 0.99e
    
