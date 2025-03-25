import random
from typing import Tuple

def calculate_bet_size(hand_strength: float, pot_size: int, min_raise: int, max_raise: int) -> int:
    """
    Calculate GTO-inspired bet size based on hand strength and pot size
    """
    # Against an all-in player, we want to go big with our strong hands
    if hand_strength >= 0.9:  # Premium hands
        return max_raise  # Always go all-in with premium hands
    elif hand_strength >= 0.8:  # Very strong hands
        return int(min(max_raise, max(min_raise, pot_size * 1.5)))  # Pot + 50%
    elif hand_strength >= 0.7:  # Strong hands
        return int(min(max_raise, max(min_raise, pot_size * 1.0)))  # Pot-sized bet
    elif hand_strength >= 0.6:  # Above average hands
        return int(min(max_raise, max(min_raise, pot_size * 0.7)))  # 70% pot
    elif hand_strength >= 0.5:  # Average hands
        return int(min(max_raise, max(min_raise, pot_size * 0.5)))  # Half pot
    else:
        # With weak hands, either check or make small bets
        return 0

def should_bluff(hand_strength: float, street: int) -> bool:
    """
    Determine if a bluff is appropriate (against all-in player, rarely bluff)
    """
    # Against an all-in player, bluffing is typically a bad strategy
    # Only bluff rarely and only on early streets
    if street == 0 and 0.4 <= hand_strength <= 0.5:
        return random.random() < 0.05  # Only 5% bluff frequency preflop
    return False  # No bluffing on later streets

def should_call(hand_strength: float, pot_odds: float, bet_size: int) -> bool:
    """
    Determine if a call is profitable based on hand strength and pot odds
    """
    # Against all-in players, we need to be extremely tight with calls
    if bet_size > 50:  # Large bet, likely all-in
        # Only call with premium hands
        return hand_strength > 0.8
    elif bet_size > 20:  # Significant bet
        # Only call with very strong hands
        return hand_strength > 0.7
    else:
        # For smaller bets, still be conservative but consider pot odds
        return hand_strength > (pot_odds + 0.15)  # Add buffer for implied odds

def get_action(hand_strength: float, street: int, can_check: bool, can_raise: bool, 
               can_call: bool, pot_size: int, min_raise: int, max_raise: int, 
               opp_bet: int) -> Tuple[int, int]:
    """
    Get the best action against an all-in bot
    """
    from gym_env import PokerEnv
    action_types = PokerEnv.ActionType
    
    # Basic strategy against an all-in player:
    # 1. Play very tight - only play premium hands
    # 2. Push all-in with our own premium hands
    # 3. Fold most marginal hands
    
    # If opponent has bet significantly
    if opp_bet > 0:
        # Calculate pot odds
        pot_odds = opp_bet / (pot_size + opp_bet)
        
        # Against an all-in player, we need to be extremely conservative with calls
        if can_call:
            if should_call(hand_strength, pot_odds, opp_bet):
                return action_types.CALL.value, 0
            else:
                return action_types.FOLD.value, 0
                
        # If we can't call, we must fold
        return action_types.FOLD.value, 0
    
    # If we can check, we're first to act or no one has bet
    if can_check:
        # With premium hands, raise
        if hand_strength >= 0.75 and can_raise:
            bet_size = calculate_bet_size(hand_strength, pot_size, min_raise, max_raise)
            if bet_size >= min_raise:
                return action_types.RAISE.value, bet_size
        
        # Otherwise check
        return action_types.CHECK.value, 0
    
    # If we can raise and should raise with a strong hand
    if can_raise and hand_strength >= 0.6:
        bet_size = calculate_bet_size(hand_strength, pot_size, min_raise, max_raise)
        if bet_size >= min_raise:
            return action_types.RAISE.value, bet_size
    
    # For any other scenario, check if possible or fold
    if can_check:
        return action_types.CHECK.value, 0
    
    # Last resort is to fold
    return action_types.FOLD.value, 0