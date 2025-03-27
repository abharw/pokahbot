import random

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class FoldAgent(Agent):
    def __name__(self):
        return "FoldAgent"

    def act(self, observation, reward, terminated, truncated, info):
        action_type = action_types.FOLD.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class CallingStationAgent(Agent):
    def __name__(self):
        return "CallingStationAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        else:
            action_type = action_types.CHECK.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class AllInAgent(Agent):
    def __name__(self):
        return "AllInAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["street"] == 0:
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")

        if observation["valid_actions"][action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            raise_amount = observation["max_raise"]
            if raise_amount > 20:
                self.logger.info(f"Going all-in for {raise_amount}")
        elif observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
            raise_amount = 0
        else:
            action_type = action_types.CHECK.value
            raise_amount = 0

        card_to_discard = -1
        return action_type, raise_amount, card_to_discard

class RandomAgent(Agent):
    def __name__(self):
        return "RandomAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = [i for i, is_valid in enumerate(observation["valid_actions"]) if is_valid]
        action_type = random.choice(valid_actions)

        if action_type == action_types.RAISE.value:
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
            if raise_amount > 20:
                self.logger.info(f"Random large raise: {raise_amount}")
        else:
            raise_amount = 0

        card_to_discard = -1
        if observation["valid_actions"][action_types.DISCARD.value]:
            card_to_discard = random.randint(0, 1)
            self.logger.debug(f"Randomly discarding card {card_to_discard}")

        return action_type, raise_amount, card_to_discard

import random

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class ProbabilityAgent(Agent):
    def __name__(self):
        return "ProbabilityAgent"

    def __init__(self):
        super().__init__()
        # Card strength by rank (0-8 corresponding to ranks 2-A)
        # Note: We're using the integer representation directly
        self.card_strength = {
            0: 0.1,  # 2
            1: 0.2,  # 3
            2: 0.3,  # 4
            3: 0.4,  # 5
            4: 0.5,  # 6
            5: 0.6,  # 7
            6: 0.7,  # 8
            7: 0.8,  # 9
            8: 1.0,  # A
        }
        # Track if we've already discarded
        self.has_discarded = False

    def _get_card_strength(self, card_int):
        """
        Get the strength of a card based on its rank.
        Works directly with integer representation.
        """
        # Extract the rank (0-8) from the card integer (0-26)
        rank = card_int % 9
        return self.card_strength[rank]

    def _calculate_hand_strength(self, observation):
        """
        Simple hand strength calculation based on card strengths
        """
        my_cards = observation["my_cards"]
        
        # Convert cards to strengths directly from integers
        strength_values = [self._get_card_strength(card) for card in my_cards]
        
        # Average card strength (0-1 scale)
        avg_strength = sum(strength_values) / len(strength_values) if strength_values else 0
        
        # Check if we have a pair (same rank)
        ranks = [card % 9 for card in my_cards]
        has_pair = len(set(ranks)) < len(ranks)
        
        # Normalize to 0-1 range and boost for pairs
        normalized_strength = avg_strength
        if has_pair:
            normalized_strength += 0.3
            
        return min(1.0, normalized_strength)  # Cap at 1.0

    def act(self, observation, reward, terminated, truncated, info):
        # Reset discard tracking at the beginning of a hand
        if observation["street"] == 0 and observation["my_bet"] == 0:
            self.has_discarded = False
            
        # Log cards for debugging
        if observation["street"] == 0:
            card_strings = [int_to_card(c) for c in observation["my_cards"]]
            self.logger.debug(f"Hole cards: {card_strings}")

        # Get valid actions
        valid_actions = [i for i, is_valid in enumerate(observation["valid_actions"]) if is_valid]
        
        # Calculate hand strength (0-1 scale)
        hand_strength = self._calculate_hand_strength(observation)
        self.logger.debug(f"Hand strength: {hand_strength:.2f}")
        
        # Consider discarding if we have a weak card and haven't discarded yet
        if action_types.DISCARD.value in valid_actions and not self.has_discarded:
            # Get card strengths
            card_strengths = [self._get_card_strength(card) for card in observation["my_cards"]]
            
            # If our weakest card is below threshold, discard it
            weakest_card = card_strengths.index(min(card_strengths))
            if card_strengths[weakest_card] < 0.4 and random.random() < 0.7:  # 70% chance to discard weak cards
                self.has_discarded = True
                card_str = int_to_card(observation["my_cards"][weakest_card])
                self.logger.debug(f"Discarding weak card {weakest_card}: {card_str}")
                return action_types.DISCARD.value, 0, weakest_card
        
        # Decision making based on hand strength
        if hand_strength > 0.8:  # Very strong hand
            # Aggressive play with strong hands
            if action_types.RAISE.value in valid_actions:
                # Size bet according to hand strength
                min_raise = observation["min_raise"]
                max_raise = observation["max_raise"]
                raise_amount = int(min_raise + (max_raise - min_raise) * hand_strength * 0.8)
                raise_amount = max(min_raise, min(max_raise, raise_amount))
                self.logger.info(f"Strong hand raise: {raise_amount}")
                return action_types.RAISE.value, raise_amount, -1
        elif hand_strength > 0.5:  # Medium strength hand
            # Mix of calls and small raises
            if action_types.RAISE.value in valid_actions and random.random() < 0.4:
                # More conservative raises with medium hands
                min_raise = observation["min_raise"]
                max_raise = observation["max_raise"]
                raise_amount = int(min_raise + (max_raise - min_raise) * 0.3)
                raise_amount = max(min_raise, min(max_raise, raise_amount))
                return action_types.RAISE.value, raise_amount, -1
            elif action_types.CALL.value in valid_actions:
                return action_types.CALL.value, 0, -1
            elif action_types.CHECK.value in valid_actions:
                return action_types.CHECK.value, 0, -1
        else:  # Weak hand
            # Mostly check/fold with occasional bluff
            if action_types.CHECK.value in valid_actions:
                return action_types.CHECK.value, 0, -1
            elif action_types.CALL.value in valid_actions:
                # Call only if the bet is small relative to pot
                if observation["opp_bet"] <= 5:
                    return action_types.CALL.value, 0, -1
                elif random.random() < 0.2:  # 20% chance to bluff call
                    return action_types.CALL.value, 0, -1
                else:
                    return action_types.FOLD.value, 0, -1
            # Random bluff with very low frequency
            elif action_types.RAISE.value in valid_actions and random.random() < 0.1:
                min_raise = observation["min_raise"]
                max_raise = observation["max_raise"]
                raise_amount = min_raise
                self.logger.info(f"Bluff raise: {raise_amount}")
                return action_types.RAISE.value, raise_amount, -1
        
        # Default actions if no specific condition was met
        if action_types.CHECK.value in valid_actions:
            return action_types.CHECK.value, 0, -1
        elif action_types.CALL.value in valid_actions:
            return action_types.CALL.value, 0, -1
        elif action_types.RAISE.value in valid_actions:
            min_raise = observation["min_raise"]
            return action_types.RAISE.value, min_raise, -1
        else:
            return action_types.FOLD.value, 0, -1
        

all_agent_classes = (FoldAgent, CallingStationAgent, AllInAgent, RandomAgent, ProbabilityAgent)
