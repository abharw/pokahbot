# Poker Terminology

> For complete game rules and tournament structure, see the [Rules Documentation](/docs/rules).
> For technical implementation details, see the [Gym Environment Documentation](/docs/gym-env).

## Core Game Structure

### Match

A match consists of 1000 hands between two bots. The bot with the most chips at the end of all hands wins the match. Each bot starts each match with 1000 chips.

### Hand

A single poker hand is one complete round of play, from dealing cards to awarding the pot. Each hand:

- Starts with dealing hole cards
- Progresses through up to 4 betting streets
- Ends when either:
  - One player folds
  - All betting is complete and remaining players show their cards (showdown)

### Streets

A street is one complete round of betting. Each hand has up to 4 streets:

- **Pre-flop** (Street 0):
  - Players receive their hole cards
  - Small blind (1 chip) and big blind (2 chips) are posted
  - First round of betting occurs

- **Flop** (Street 1):
  - Three community cards are dealt
  - Second round of betting occurs
  - Players may use the discard action

- **Turn** (Street 2):
  - Fourth community card is dealt
  - Third round of betting occurs

- **River** (Street 3):
  - Final community card is dealt
  - Final round of betting occurs
  - If no one folds, showdown occurs

## Basic Terms

### Positions

- **Small Blind (SB)**: Player 0, posts 1 chip before cards are dealt
- **Big Blind (BB)**: Player 1, posts 2 chips before cards are dealt

### Actions

- **Fold**: Give up the hand and any chips bet
- **Check**: Pass the action when no additional bet is required
- **Call**: Match the current bet amount
- **Raise**: Increase the current bet amount
- **Discard**: Exchange one hole card for a new one (tournament-specific rule)

> See the [Gym Environment Documentation](/docs/gym-env#action-space) for technical implementation details.

## Hand Rankings

> See [Rules Documentation](/docs/rules#hand-rankings) for detailed hand rankings and examples.

## Strategic Concepts

### Pot Odds

The ratio of the current pot size to the cost of a call. Used to make mathematically sound calling decisions.

Example:

- Pot contains 10 chips
- Opponent bets 5 chips
- Pot odds are 15:5 or 3:1

### Position

Your place in the betting order, which affects strategic decisions:

- **Out of Position (OOP)**: Acting first (Small Blind)
- **In Position (IP)**: Acting last (Big Blind)

### Ranges

The collection of possible hands an opponent might have based on their actions.

## Engine-Specific Terms

For technical details about how the game engine represents and handles:

- [Card Representation](/docs/gym-env#card-representation)
- [Action Space](/docs/gym-env#action-space)
- [Valid Actions](/docs/gym-env#observation-space)

The following terms are commonly used in the API:

### Observation Dictionary Keys

- `street`: Current betting round (0-3)
- `acting_agent`: Which player acts next (0 or 1)
- `my_cards`: Your hole cards
- `community_cards`: Visible shared cards

## Common Abbreviations

- **BB**: Big Blind
- **SB**: Small Blind
- **OOP**: Out of Position
- **IP**: In Position
- **EV**: Expected Value
- **SPR**: Stack-to-Pot Ratio
