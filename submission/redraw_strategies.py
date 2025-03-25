import random
from typing import List, Dict, Any

def should_redraw(evaluator, obs: Dict[str, Any], strategy: int) -> bool:
    """
    Determine if we should redraw a card based on the chosen strategy
    
    Args:
        evaluator: Hand evaluator instance
        obs: Current observation
        strategy: Redraw strategy (1, 2, or 3)
        
    Returns:
        Boolean indicating whether to redraw
    """
    my_cards = obs["my_cards"]
    street = obs["street"]
    community_cards = obs["community_cards"]
    
    # Against an all-in bot, we want to optimize for strong made hands
    
    # Strategy 1: Redraw based on hand strength (bottom 50% - increased threshold)
    if strategy == 1:
        if street == 0:
            # Pre-flop strategy: redraw more aggressively with weaker hands
            hand_strength = evaluator.get_preflop_strength(my_cards)
            
            # Check for specific hands that benefit from redraw
            ranks = [evaluator.get_rank(card) for card in my_cards]
            
            # Always keep pocket pairs
            if ranks[0] == ranks[1]:
                return False
                
            # Always keep A-9 suited
            if (ranks[0] == 8 and ranks[1] == 7) or (ranks[0] == 7 and ranks[1] == 8):
                if evaluator.get_suit(my_cards[0]) == evaluator.get_suit(my_cards[1]):
                    return False
            
            # Keep any hand with Ace + good card
            if 8 in ranks:
                other_rank = ranks[0] if ranks[1] == 8 else ranks[1]
                # Keep Ace with 7+
                if other_rank >= 5:  # 7 or higher
                    return False
            
            # Redraw anything below threshold
            return hand_strength < 0.5
            
        elif street == 1:
            # Flop strategy: redraw if hand didn't connect well
            hand_strength = evaluator.calculate_hand_strength(my_cards, community_cards)
            
            # Get made hand classification
            visible_cards = my_cards + [c for c in community_cards if c != -1]
            hand_type, _ = evaluator.evaluate_hand(visible_cards)
            
            # Keep any made hand better than a pair
            if hand_type > evaluator.ONE_PAIR:
                return False
                
            # Redraw with weaker hands
            return hand_strength < 0.5
    
    # Strategy 2: Redraw as late as possible (before turn)
    elif strategy == 2:
        if street == 1:  # On flop
            # Check if hand is weak but not terrible
            hand_strength = evaluator.calculate_hand_strength(my_cards, community_cards)
            return hand_strength < 0.6  # Redraw more hands
        return False
    
    # Strategy 3: Always redraw before flop (except with premium hands)
    elif strategy == 3:
        if street == 0:
            # Only keep premium starting hands
            hand_strength = evaluator.get_preflop_strength(my_cards)
            return hand_strength < 0.7  # Only keep the top 30% of hands
        return False
    
    # Default: no redraw
    return False

def choose_card_to_discard(evaluator, my_cards: List[int], community_cards: List[int] = None) -> int:
    """
    Choose which card to discard (0 or 1)
    
    Args:
        evaluator: Hand evaluator instance
        my_cards: Player's hole cards
        community_cards: Known community cards
        
    Returns:
        Index of the card to discard (0 or 1)
    """
    # If we only have one card, discard it
    if len(my_cards) == 1:
        return 0
        
    # If pre-flop, discard the lower card unless they form a pair
    if not community_cards or all(c == -1 for c in community_cards):
        rank0 = evaluator.get_rank(my_cards[0])
        rank1 = evaluator.get_rank(my_cards[1])
        
        # Keep pairs
        if rank0 == rank1:
            # For pairs, keep both if possible, but if forced to discard, randomly choose
            return random.randint(0, 1)
        
        # Handle Ace which is highest
        if rank0 == 8:  # Ace in first position
            return 1   # Keep Ace, discard other card
        if rank1 == 8:  # Ace in second position
            return 0   # Keep Ace, discard other card
            
        # Otherwise discard lower rank
        return 0 if rank0 < rank1 else 1
    
    # On the flop, evaluate which card contributes less to the hand
    # Try keeping each card and evaluate the resulting hand strength
    card0_with_community = [my_cards[0]] + [c for c in community_cards if c != -1]
    card1_with_community = [my_cards[1]] + [c for c in community_cards if c != -1]
    
    # Evaluate the potential of each card with the community cards
    hand_type0, kickers0 = evaluator.evaluate_hand(card0_with_community)
    hand_type1, kickers1 = evaluator.evaluate_hand(card1_with_community)
    
    # Keep the card that makes a better hand
    if hand_type0 > hand_type1:
        return 1  # Discard card 1
    elif hand_type1 > hand_type0:
        return 0  # Discard card 0
    elif kickers0 and kickers1:
        # If hand types are equal, compare kickers
        if kickers0[0] > kickers1[0]:
            return 1  # Discard card 1
        elif kickers1[0] > kickers0[0]:
            return 0  # Discard card 0
    
    # Check for specific draws
    suit0 = evaluator.get_suit(my_cards[0])
    suit1 = evaluator.get_suit(my_cards[1])
    
    # Count how many community cards match each suit
    suit0_matches = sum(1 for c in community_cards if c != -1 and evaluator.get_suit(c) == suit0)
    suit1_matches = sum(1 for c in community_cards if c != -1 and evaluator.get_suit(c) == suit1)
    
    # Keep the card that has more flush potential
    if suit0_matches > suit1_matches:
        return 1  # Discard card 1
    elif suit1_matches > suit0_matches:
        return 0  # Discard card 0
    
    # If still tied, keep the higher card
    rank0 = evaluator.get_rank_value(evaluator.get_rank(my_cards[0]))
    rank1 = evaluator.get_rank_value(evaluator.get_rank(my_cards[1]))
    return 0 if rank0 > rank1 else 1