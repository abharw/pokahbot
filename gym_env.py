"""
CMU Poker Bot Competition Game Engine 2025

People working on this code, please refer to:
https://gymnasium.farama.org
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Keep in mind gym doesn't inherently support multi-agent environments.
We will have to use the Tuple space to represent the observation space and
action space for each agent.
"""

import logging
import os
from enum import Enum

import gym
import numpy as np
from gym import spaces
from treys import Card, Evaluator


class PokerEnv(gym.Env):
    SMALL_BLIND_PLAYER = 0
    BIG_BLIND_PLAYER = 1
    MAX_PLAYER_BET = 100

    RANKS = "23456789A"
    SUITS = "dhs"  # diamonds hearts spade

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer desiged for fast lookup & comparison
        """
        return Card.new(PokerEnv.int_card_to_str(card_int))

    @staticmethod
    def int_card_to_str(card_int: int):
        RANKS, SUITS = PokerEnv.RANKS, PokerEnv.SUITS
        rank = RANKS[card_int % len(RANKS)]
        suit = SUITS[card_int // len(RANKS)]
        return rank + suit

    class ActionType(Enum):
        FOLD = 0
        RAISE = 1
        CHECK = 2
        CALL = 3
        DISCARD = 4
        INVALID = 5

    def __init__(self, logger=None, small_blind_amount=1, num_hands=1):
        """
        Represents a single hand of poker.
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = small_blind_amount * 2
        self.min_raise = self.big_blind_amount
        self.acting_agent = PokerEnv.SMALL_BLIND_PLAYER
        self.last_street_bet = None
        self.evaluator = Evaluator()

        # Action space is a Tuple (action_type, raise_amount, card_to_discard)
        # where action is a Discrete(4), raise_amount is a Discrete(100), and card_to_discard is a Discrete(3) (-1 means no card is discarded)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(len(self.ActionType) - 1),  # user can pass any besides INVALID
                spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                spaces.Discrete(3, start=-1),
            ]
        )

        # Card space is a Discrete(28), -1 means the card is not shown
        cards_space = spaces.Discrete((len(self.SUITS) * len(self.RANKS)) + 1, start=-1)

        # Single observation space is a Dict.
        # Since we have two players, acting_agent is a Discrete(2)
        # Make sure to check (acting_agent == agent_num) before taking an action
        # opp_shown_card is "0" if the opp's card is not shown
        # Two players, so the observation space is a Tuple of two single_observation_spaces
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),
                "acting_agent": spaces.Discrete(2),
                "my_cards": spaces.Tuple([cards_space for _ in range(2)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_discarded_card": cards_space,
                "opp_drawn_card": cards_space,
                "min_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "max_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "valid_actions": spaces.MultiBinary(len(self.ActionType) - 1),
            }
        )

        # Since we have two players, the observation space is a tuple of
        # (observation_space_one_player, observation_space_one_player)
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(2)])

        # New episode
        self.reset(seed=int.from_bytes(os.urandom(32)))

    def _get_valid_actions(self, player_num: int):
        valid_actions = [1, 1, 1, 1, 1]
        # You can't check if the other player has a larger bet
        if self.bets[player_num] < self.bets[1 - player_num]:
            valid_actions[self.ActionType.CHECK.value] = 0
        # You can't call if you have an equal bet
        if self.bets[player_num] == self.bets[1 - player_num]:
            valid_actions[self.ActionType.CALL.value] = 0
        # You can't discard if you have already discarded
        if self.discarded_cards[player_num] != -1:
            valid_actions[self.ActionType.DISCARD.value] = 0
        # You can discard only in the street (0,1)
        if self.street > 1:
            valid_actions[self.ActionType.DISCARD.value] = 0

        if (max(self.bets)) == self.MAX_PLAYER_BET:
            valid_actions[self.ActionType.RAISE.value] = 0

        return valid_actions

    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for the player_num player.
        """
        if self.street == 0:
            num_cards_to_reveal = 0
        else:
            num_cards_to_reveal = self.street + 2

        obs = {
            "street": self.street,
            "acting_agent": self.acting_agent,
            "my_cards": self.player_cards[player_num],
            "community_cards": self.community_cards[:num_cards_to_reveal] + [-1 for _ in range(5 - num_cards_to_reveal)],
            "my_bet": self.bets[player_num],
            "opp_bet": self.bets[1 - player_num],
            "opp_discarded_card": self.discarded_cards[1 - player_num],
            "opp_drawn_card": self.drawn_cards[1 - player_num],
            "my_discarded_card": self.discarded_cards[player_num],
            "my_drawn_card": self.drawn_cards[player_num],
            "min_raise": self.min_raise,
            "max_raise": self.MAX_PLAYER_BET - max(self.bets),
            "valid_actions": self._get_valid_actions(player_num),
        }
        # All in situation
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]

        info = {
            "player_cards": [self.int_card_to_str(card) for card in obs["my_cards"] if card != -1],
            "community_cards": [self.int_card_to_str(card) for card in obs["community_cards"] if card != -1],
        }
        return obs, info

    def _get_obs(self, winner, invalid_action=False):
        """
        Returns the observation for both players.
        """
        obs0, info0 = self._get_single_player_obs(0)
        obs1, info1 = self._get_single_player_obs(1)
        if winner == 0:
            reward = (min(self.bets), -min(self.bets))
        elif winner == 1:
            reward = (-min(self.bets), min(self.bets))
        else:
            reward = (0, 0)
        terminated = winner is not None
        truncated = False
        info = {"player_0_cards": info0["player_cards"], "player_1_cards": info1["player_cards"], "community_cards": info0["community_cards"], "invalid_action": invalid_action}
        return (obs0, obs1), reward, terminated, truncated, info

    def _draw_card(self):
        drawn_card = self.cards[0]
        self.cards = self.cards[1:]
        return drawn_card

    def reset(self, *, seed=None, options=None):
        """
        Resets the entire game.
        Default is random deal, but options can be provided to set the initial state.

        options is a dict with the following keys:
        - cards: a list of 27 cards to be used in the game
        """
        super().reset(seed=seed)
        self.street = 0
        self.bets = [0, 0]
        self.discarded_cards = [-1, -1]
        self.drawn_cards = [-1, -1]

        # Set default values first
        self.cards = np.arange(27)
        np.random.shuffle(self.cards)
        self.small_blind_player = 0  # Default to player 0

        # Override with any provided options, using defaults as fallbacks
        if options is not None:
            self.cards = options.get("cards", self.cards)
            self.small_blind_player = options.get("small_blind_player", self.small_blind_player)

        self.big_blind_player = 1 - self.small_blind_player
            
        # Deal to players and community
        self.player_cards = [[self._draw_card() for _ in range(2)] for _ in range(2)]
        self.community_cards = [self._draw_card() for _ in range(5)]

        # Assign blinds
        self.acting_agent = self.small_blind_player
        self.bets = [0, 0]
        self.bets[self.small_blind_player] = self.small_blind_amount
        self.bets[self.big_blind_player] = self.big_blind_amount
        self.min_raise = self.big_blind_amount
        self.last_street_bet = 0

        obs0, info0 = self._get_single_player_obs(0)
        obs1, info1 = self._get_single_player_obs(1)
        info = {
            "player_0_cards": info0["player_cards"],
            "player_1_cards": info1["player_cards"],
            "community_cards": info0["community_cards"],
        }

        return (obs0, obs1), info

    def _next_street(self):
        """
        Update to the next street of the game.
        """
        self.street += 1
        self.min_raise = self.big_blind_amount
        assert self.bets[0] == self.bets[1], self.logger.log(f"Bet amounts are not equal: {self.bets}")
        self.last_street_bet = self.bets[0]
        self.acting_agent = self.small_blind_player

    def _get_winner(self):
        """
        Returns the winner of the game.
        """
        board_cards = list(map(self.int_to_card, self.community_cards))
        player_0_cards = list(map(self.int_to_card, [c for c in self.player_cards[0] if c != -1]))
        player_1_cards = list(map(self.int_to_card, [c for c in self.player_cards[1] if c != -1]))
        assert len(player_0_cards) == 2 and len(player_1_cards) == 2 and len(board_cards) == 5
        player_0_hand_rank = self.evaluator.evaluate(
            player_0_cards,
            board_cards,
        )
        player_1_hand_rank = self.evaluator.evaluate(
            player_1_cards,
            board_cards,
        )

        self.logger.debug(f"(get winner) Player 0 cards: {list(map(Card.int_to_str, player_0_cards))}; Player 1 cards: {list(map(Card.int_to_str, player_1_cards))}")
        self.logger.debug(f"Determined winner based on hand scores; p0 score: {player_0_hand_rank}; p1 score: {player_1_hand_rank}")

        # showdown
        if player_0_hand_rank == player_1_hand_rank:
            winner = -1  # tie
        elif player_1_hand_rank < player_0_hand_rank:
            winner = 1
        else:
            winner = 0
        return winner

    def step(self, action: tuple[int, int, int]):
        """
        Takes a step in the game, given the action taken by the active player.

        `action`: (action_type, raise_amount, card_to_discard)
            - `action_type`: `int`, index of the action type
            - `raise_amount`: `int`, how much to raise, or 0 for a check or call
            - `card_to_discard`: `int`, index of the card which you would like to discard (0, or 1) or -1
        """
        action_type, raise_amount, card_to_discard = action
        valid_actions = self._get_valid_actions(self.acting_agent)
        self.logger.debug(f"Action type: {action_type}, Valid actions: {valid_actions}, Street: {self.street}, Bets: {self.bets}")

        # Handle invalid actions
        if not valid_actions[action_type]:
            action_name = self.ActionType(action_type).name
            valid_action_names = [self.ActionType(i).name for i, is_valid in enumerate(valid_actions) if is_valid]
            self.logger.error(f"Player {self.acting_agent} attempted invalid action: {action_name}. Valid actions are: {valid_action_names}")
            action_type = self.ActionType.INVALID.value

        if action_type == self.ActionType.RAISE.value and not (self.min_raise <= raise_amount <= (self.MAX_PLAYER_BET - max(self.bets))):
            self.logger.error(f"Player {self.acting_agent} attempted invalid raise amount: {raise_amount}. Must be between {self.min_raise} and {self.MAX_PLAYER_BET - max(self.bets)}")
            action_type = self.ActionType.INVALID.value

        winner = None

        new_street = False
        if action_type in (self.ActionType.FOLD.value, self.ActionType.INVALID.value):
            # We consider invalid actions as a fold
            self.logger.debug(f"Player {self.acting_agent} Folded")
            winner = 1 - self.acting_agent
        elif action_type == self.ActionType.CALL.value:
            self.bets[self.acting_agent] = self.bets[1 - self.acting_agent]
            if not (self.street == 0 and self.acting_agent == self.small_blind_player):
                # on the first street, the little blind can "call" the big blind's bet of 2
                assert self.bets[self.acting_agent] >= self.big_blind_amount
                new_street = True
        elif action_type == self.ActionType.CHECK.value:
            if self.acting_agent == self.big_blind_player:
                new_street = True  # big blind checks mean next street
        elif action_type == self.ActionType.RAISE.value:
            assert (
                self.bets[1 - self.acting_agent] >= self.bets[self.acting_agent]
            ), "Expected the opponent to have bet at least as much as current player given current player is raising"
            self.bets[self.acting_agent] = self.bets[1 - self.acting_agent] + raise_amount
            raise_so_far = self.bets[1 - self.acting_agent] - self.last_street_bet

            max_raise = self.MAX_PLAYER_BET - max(self.bets)
            min_raise_no_limit = raise_so_far + raise_amount
            self.min_raise = min(min_raise_no_limit, max_raise)
        else:
            # Must be DISCARD at this point
            assert action_type == self.ActionType.DISCARD.value, f"Unexpected action type: {action_type}"
            if card_to_discard != -1:
                self.discarded_cards[self.acting_agent] = self.player_cards[self.acting_agent][card_to_discard]
                drawn_card = self._draw_card()
                self.drawn_cards[self.acting_agent] = drawn_card
                self.player_cards[self.acting_agent][card_to_discard] = drawn_card

        if new_street:
            self._next_street()
            if self.street > 3:
                winner = self._get_winner()

        if not new_street and action_type != self.ActionType.DISCARD.value:
            self.acting_agent = 1 - self.acting_agent

        obs, reward, terminated, truncated, info = self._get_obs(winner, action_type == self.ActionType.INVALID.value)
        if terminated:
            self.logger.debug(
                f"Game is terminated. P0 cards: {list(map(self.int_card_to_str, self.player_cards[0]))}; P1 cards: {list(map(self.int_card_to_str, self.player_cards[1]))}; board cards: {list(map(self.int_card_to_str, self.community_cards))}"
            )

            if winner == 0:
                reward = (min(self.bets), -min(self.bets))
            elif winner == 1:
                reward = (-min(self.bets), min(self.bets))
            else:
                # tie
                reward = (0, 0)

            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
