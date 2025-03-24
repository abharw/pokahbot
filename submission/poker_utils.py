import numpy as np
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

# Card constants
RANKS = "23456789A"
SUITS = "dhs"  # diamonds, hearts, spades

def encode_cards(cards, max_cards=7):
    """
    Encode cards into a binary representation
    
    Args:
        cards (list): List of card indices
        max_cards (int): Maximum number of cards to encode
        
    Returns:
        numpy.ndarray: One-hot encoded representation of cards
    """
    # Create a 27-length vector (27 cards in the deck)
    encoded = np.zeros(27, dtype=np.float32)
    
    # Set the corresponding indices to 1
    for card in cards:
        if card >= 0:  # Ignore placeholder -1 cards
            encoded[card] = 1.0
            
    return encoded

def encode_game_state(observation, hand_strength, opponent_aggression, discard_used):
    """
    Encode the entire game state into a feature vector
    
    Args:
        observation (dict): Game observation
        hand_strength (float): Estimated hand strength (0-1)
        opponent_aggression (float): Opponent aggression factor
        discard_used (bool): Whether discard action has been used
        
    Returns:
        numpy.ndarray: Encoded game state vector
    """
    # Extract observation components
    street = observation["street"]
    my_cards = observation["my_cards"]
    community_cards = observation["community_cards"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    opp_discarded_card = observation["opp_discarded_card"]
    opp_drawn_card = observation["opp_drawn_card"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]
    
    # Encode cards
    my_cards_encoded = encode_cards(my_cards, max_cards=2)
    community_cards_encoded = encode_cards(community_cards, max_cards=5)
    
    # Encode discard information
    discard_info = np.zeros(2, dtype=np.float32)
    if opp_discarded_card >= 0:
        discard_info[0] = 1.0
    if discard_used:
        discard_info[1] = 1.0
    
    # Encode betting information
    pot_size = my_bet + opp_bet
    pot_odds = calculate_pot_odds(my_bet, opp_bet)
    
    # Normalize values
    street_norm = street / 3.0  # 4 streets (0-3)
    my_bet_norm = my_bet / 100.0  # Max bet is 100
    opp_bet_norm = opp_bet / 100.0
    pot_size_norm = pot_size / 200.0  # Max pot is 200
    min_raise_norm = min_raise / 100.0
    max_raise_norm = max_raise / 100.0
    
    # Combine all features
    game_state = np.concatenate([
        [street_norm],                  # Game stage
        [my_bet_norm, opp_bet_norm],    # Current bets
        [pot_size_norm, pot_odds],      # Pot information
        [min_raise_norm, max_raise_norm], # Raise limits
        [hand_strength],                # Hand strength
        [opponent_aggression],          # Opponent model
        discard_info,                   # Discard information
        my_cards_encoded,               # My cards (27 values)
        community_cards_encoded,        # Community cards (27 values)
    ])
    
    return game_state

def get_hand_strength(hole_cards, community_cards):
    """
    Estimate the strength of a poker hand using a simplified evaluation
    For a production system, implement a more accurate hand evaluator
    
    Args:
        hole_cards (list): List of hole card indices
        community_cards (list): List of community card indices
        
    Returns:
        float: Hand strength estimate (0-1)
    """
    # Filter out -1 placeholder cards
    hole = [c for c in hole_cards if c >= 0]
    community = [c for c in community_cards if c >= 0]
    
    # If no valid cards, return minimum strength
    if not hole:
        return 0.0
    
    # Card conversion helpers
    def get_rank(card_idx):
        # Card index from 0-26, convert to rank value (0-8)
        # 0-8 = 2-A of diamonds, 9-17 = 2-A of hearts, 18-26 = 2-A of spades
        return card_idx % 9
    
    def get_suit(card_idx):
        # Return 0 for diamonds, 1 for hearts, 2 for spades
        return card_idx // 9
    
    # Convert cards to rank and suit
    all_cards = hole + community
    ranks = [get_rank(c) for c in all_cards]
    suits = [get_suit(c) for c in all_cards]
    
    # Count rank frequencies
    rank_counts = [ranks.count(r) for r in range(9)]
    max_rank_count = max(rank_counts)
    
    # Count suit frequencies
    suit_counts = [suits.count(s) for s in range(3)]
    max_suit_count = max(suit_counts)
    
    # Check if we have a straight
    has_straight = False
    # Special case for A2345
    if set([0, 1, 2, 3, 8]).issubset(set(ranks)):
        has_straight = True
    # Check for regular straights (5 consecutive cards)
    if not has_straight and len(all_cards) >= 5:
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i:i+5] == list(range(sorted_ranks[i], sorted_ranks[i] + 5)):
                has_straight = True
                break
    
    # Determine hand strength based on poker hand rankings
    if max_suit_count >= 5 and has_straight:
        # Straight flush
        return 1.0
    elif max_rank_count >= 3 and ranks.count(2) >= 2:
        # Full house
        return 0.9
    elif max_suit_count >= 5:
        # Flush
        return 0.8
    elif has_straight:
        # Straight
        return 0.7
    elif max_rank_count >= 3:
        # Three of a kind
        return 0.6
    elif rank_counts.count(2) >= 2:
        # Two pair
        return 0.5
    elif max_rank_count >= 2:
        # One pair
        return 0.4
    else:
        # High card - assign value based on highest card
        highest_rank = max(ranks) if ranks else 0
        return 0.1 + (highest_rank / 9.0) * 0.2  # Scale between 0.1 and 0.3

def calculate_pot_odds(my_bet, opp_bet):
    """
    Calculate pot odds
    
    Args:
        my_bet (int): My current bet
        opp_bet (int): Opponent's current bet
        
    Returns:
        float: Pot odds (0-1)
    """
    call_amount = max(0, opp_bet - my_bet)
    pot_size = my_bet + opp_bet
    
    if call_amount == 0 or pot_size == 0:
        return 0.0
    
    return call_amount / (pot_size + call_amount)

def get_action_mask(observation):
    """
    Create a binary mask for valid actions
    
    Args:
        observation (dict): Game observation
        
    Returns:
        numpy.ndarray: Binary mask for valid actions
    """
    valid_actions = observation["valid_actions"]
    action_mask = np.zeros(6, dtype=np.float32)
    
    # Set first 4 positions based on valid actions (FOLD, RAISE, CHECK, CALL)
    for i in range(4):
        if valid_actions[i]:
            action_mask[i] = 1.0
    
    # Position 4 is raise sizing (always valid)
    action_mask[4] = 1.0
    
    # Position 5 is discard decision
    if valid_actions[action_types.DISCARD.value]:
        action_mask[5] = 1.0
    
    return action_mask