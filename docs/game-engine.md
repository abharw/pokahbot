# Game Engine Deep Dive

## Overview

The poker game engine implements the [tournament variant](/docs/rules) of Texas Hold'em using OpenAI's Gym framework (now Gymnasium). This document focuses on the technical implementation details.

## Environment Structure

### Core Components

```python
class PokerEnv(gym.Env):
    SMALL_BLIND_PLAYER = 0
    BIG_BLIND_PLAYER = 1
    MAX_PLAYER_BET = 100
```

The environment maintains game state including:

- Player cards
- Community cards
- Current bets
- Street number
- Acting player
- Bankrolls
- Discarded/drawn cards

## Action Space

### Available Actions

```python
class ActionType(Enum):
    FOLD = 0
    RAISE = 1
    CHECK = 2
    CALL = 3
    DISCARD = 4
    INVALID = 5
```

Each action is a tuple of three values:

```python
(action_type: int, raise_amount: int, card_to_discard: int)
```

- `action_type`: One of the ActionType enum values
- `raise_amount`: Amount to raise (1 to MAX_PLAYER_BET)
- `card_to_discard`: Card index to discard (-1, 0, or 1)

## Observation Space

Each player receives an observation dictionary containing:

```python
{
    "street": int,              # Current street (0-3)
    "acting_agent": int,        # Which player acts next (0 or 1)
    "my_cards": List[int],      # Player's hole cards
    "community_cards": List[int], # Visible community cards
    "my_bet": int,             # Player's current bet
    "opp_bet": int,            # Opponent's current bet
    "opp_discarded_card": int, # Card opponent discarded (-1 if none)
    "opp_drawn_card": int,     # Card opponent drew (-1 if none)
    "min_raise": int,          # Minimum raise amount
    "max_raise": int,          # Maximum raise amount
    "valid_actions": List[bool] # Which actions are currently valid
}
```

## Card Representation

Cards are represented as integers from 0-26:

```python
RANKS = "23456789A"
SUITS = "dhs"  # diamonds, hearts, spades
```

Example card mappings:

- 0 = 2♦
- 9 = A♦
- 10 = 2♥
- 19 = A♥
- 20 = 2♠
- 26 = A♠

## Game Flow

### 1. Initialization

```python
env = PokerEnv(num_hands=1000)
(obs0, obs1), info = env.reset()
```

### 2. Betting Streets

The game progresses through 4 streets:

- Street 0: Pre-flop
- Street 1: Flop
- Street 2: Turn
- Street 3: River

### 3. Taking Actions

```python
obs, reward, terminated, truncated, info = env.step((action_type, raise_amount, card_to_discard))
```

### 4. Hand Evaluation

Uses the `treys` library to evaluate final hands:

```python
evaluator = Evaluator()
hand_rank = evaluator.evaluate(player_cards, board_cards)
# Lower rank = better hand
```

## Example Game Sequences

### Example 1: Basic Call/Check Sequence

```python
rigged_deck = [25, 14, 1, 4, 8, 16, 9, 11, 23]
actions = [
    (ActionType.CALL.value, 0, -1),   # P0 calls big blind
    (ActionType.CHECK.value, 0, -1),   # P1 checks
    (ActionType.RAISE.value, 2, -1),   # P0 raises 2
    (ActionType.CALL.value, 0, -1),    # P1 calls
    # ... continues to showdown
]
```

### Example 2: Max Bet Scenario

```python
rigged_deck = [24, 10, 14, 5]
actions = [
    (ActionType.CALL.value, 0, -1),     # P0 calls big blind
    (ActionType.RAISE.value, 98, -1),    # P1 raises to max
    (ActionType.FOLD.value, 0, -1),      # P0 folds
]
```

### Example 3: Discard Action

```python
# Player discards first card (index 0)
action = (ActionType.DISCARD.value, 0, 0)

# Observation will show:
{
    "opp_discarded_card": 3,    # Card that was discarded
    "opp_drawn_card": 0,        # New card that was drawn
    # ... other observation fields
}
```

## Important Rules and Constraints

1. **Betting Limits**
   - Minimum raise: Previous raise amount
   - Maximum bet: 100 (MAX_PLAYER_BET)
   - Small blind: 1
   - Big blind: 2

2. **Discard Rules**
   - Only allowed once per player per hand
   - Not allowed during river
   - Must discard before seeing turn and river cards

3. **Valid Actions**
   - Can't check if opponent has bet
   - Can't call if bets are equal
   - Can't raise beyond MAX_PLAYER_BET
   - Can't discard after street 1

## Common Pitfalls

1. **Invalid Actions**
   - Always check `valid_actions` before choosing an action
   - Invalid actions are treated as folds

2. **Raise Amounts**
   - Must be between `min_raise` and `max_raise`
   - Raises are cumulative (total bet, not increment)

3. **Discarding**
   - Can only discard once
   - Must discard before street 2
   - Card indices must be 0 or 1

4. **Observation Processing**
   - Convert numpy values to native Python types
   - Handle -1 values for hidden/unused cards

> For terminology and strategic concepts, see the [Terminology Guide](/docs/terminology).
