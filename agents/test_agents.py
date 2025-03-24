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


all_agent_classes = (FoldAgent, CallingStationAgent, AllInAgent, RandomAgent)
