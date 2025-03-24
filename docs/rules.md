# Tournament Rules

> For technical implementation details, see the [Gym Environment Documentation](/docs/gym-env).
> For common poker terms, see the [Terminology Guide](/docs/terminology).

## Bot Technical Requirements

### Computational Resources

Compute resources will increase in phases throughout the tournament:

1. **Phase 1** (until March 23, 11:59PM EST):
   - 1 vCPU
   - 2GB RAM per bot
   - 500 seconds time limit per match

2. **Phase 2** (March 24 - March 25, 11:59PM EST):
   - 2 vCPU
   - 4GB RAM per bot
   - 1000 seconds time limit per match

3. **Final Phase** (March 26 - March 29, 11:00AM EST):
   - 4 vCPU
   - 8GB RAM per bot
   - 1500 seconds time limit per match

All bots run in a Python 3.12 runtime environment on ARM64 architecture (AWS Graviton2).

### Time Constraints

- Each bot's time bank depends on the current phase (see above)
- Time bank depletes whenever the bot is computing an action
- If a bot's time bank is depleted during a match:
  - The match immediately ends
  - The bot automatically forfeits the match
  - The opponent is awarded the win

> **Important**: With 1000 hands per match, bots should aim to use their time efficiently to avoid depleting their time bank. For example, in the final phase with 1500 seconds, this means using no more than 1.5 seconds per hand on average.

## Game Variant

Our tournament uses a modified version of Texas Hold'em with unique features:

### Card Deck

- Modified 27-card deck
- Uses cards 2-9 and A from ♦♥♠ suits only
- No face cards (J,Q,K) or clubs (♣)

### Hand Rankings

From strongest to weakest:

1. **Straight Flush**: Five consecutive cards of the same suit
   - Example: 5♦ 6♦ 7♦ 8♦ 9♦
   - Example: A♦ 2♦ 3♦ 4♦ 5♦ (Ace can be low)
   - Example: 6♦ 7♦ 8♦ 9♦ A♦ (Ace can be high)

2. **Full House**: Three of a kind plus a pair
   - Example: 7♦ 7♥ 7♠ 4♦ 4♥

3. **Flush**: Five cards of the same suit
   - Example: 2♦ 4♦ 6♦ 8♦ 9♦

4. **Straight**: Five consecutive cards of any suit
   - Example: 5♦ 6♥ 7♦ 8♠ 9♥
   - Example: A♦ 2♥ 3♠ 4♦ 5♥ (Ace can be low)
   - Example: 6♦ 7♥ 8♠ 9♦ A♥ (Ace can be high)

5. **Three of a Kind**: Three cards of the same rank
   - Example: 8♦ 8♥ 8♠ 2♦ 3♥

6. **Two Pair**: Two different pairs
   - Example: 9♦ 9♥ 3♦ 3♥ 8♠

7. **One Pair**: Two cards of the same rank
   - Example: A♦ A♥ 7♦ 4♠ 2♥

8. **High Card**: Highest single card
   - Example: A♦ 8♠ 6♥ 4♦ 2♠

> **Special Ace Rule**: The Ace (A) can be used as either high (above 9) or low (below 2) when forming straights and straight flushes.
>
> Note: Four-of-a-kind is impossible with this deck configuration since there are only three suits.

### Redraw Rule

- Each player may discard and redraw one card per hand
- Must be done before the turn card
- Discarded card and drawn card are revealed to opponent
- Discarded card removed from play for that hand

## Tournament Structure

### Open Season (Feb 22 - Feb 29, 2025)

- **Format**: ELO-based ranking system
- **Match Structure**:
  - 1000 hands per match
  - Stack resets between hands
  - Total chips tracked across entire match
  - Winner determined by having more chips at the end
- **Matchmaking**:
  - Matches scheduled every ~12.5 minutes (116 matches per day)
  - Players matched to achieve uniform ELO distribution exposure
  - Each player faces opponents across full ELO range daily
  - ELO ratings updated after each match
  - Match logs provided to participants

### Match Requests

In addition to regular matchmaking, teams can request specific matches:

- Up to 5 match requests per day
- All match requests are automatically accepted
- Match results do not count towards official rankings
- Use for testing and practice purposes

### Finals (February 29, 2025)

- **Format**: Round-robin tournament between top 12 teams by ELO
- **Location**: Giant Eagle Auditorium (Baker Hall A51)
- **Scoring**: Winners determined by most matches won in the round-robin
- **Note**: We reserve the right to invite additional teams if the ELO cutoff is close or if resources permit extra matches

## Important Dates

- **Competition Opens**: February 22, 2025
- **Finals**: February 29, 2025
- **Submission Deadline**: March 29, 2025, 11:00AM EST

## Legal

See the full rules and terms of service [here](https://poker.cmudsc.com/terms/).

See the privacy policy [here](https://poker.cmudsc.com/privacy/).
