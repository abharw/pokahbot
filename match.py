"""
Script to run matches between agents.
"""

import csv
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests


from gym_env import PokerEnv

TIME_LIMIT_SECONDS = 500
GET_ACTION_ENDPOINT = "/get_action"
SEND_OBS_ENDPOINT = "/post_observation"


class AgentFailure(Exception):
    """Custom exception for tracking agent failures"""

    pass


class AgentFailureTracker:
    def __init__(self):
        self.failed_attempts = {0: 0, 1: 0}
        self.MAX_FAILURES = 3

    def record_failure(self, player_id: int):
        self.failed_attempts[player_id] += 1

        # Check for both players failing
        if self.failed_attempts[0] >= self.MAX_FAILURES and self.failed_attempts[1] >= self.MAX_FAILURES:
            raise AgentFailure("Both players have failed multiple times")

        # Check for single player persistent failure
        if self.failed_attempts[player_id] >= self.MAX_FAILURES:
            raise AgentFailure(f"Player {player_id} has failed {self.MAX_FAILURES} times")

    def record_success(self, player_id: int):
        self.failed_attempts[player_id] = 0


# Create a global instance
failure_tracker = AgentFailureTracker()


def get_street_name(street_num: int) -> str:
    """Convert numeric street value to human-readable name"""
    street_names = {0: "Pre-Flop", 1: "Flop", 2: "Turn", 3: "River"}
    return street_names.get(street_num, f"Unknown-{street_num}")


def prepare_payload(
    obs: Dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare the payload for API calls by converting numpy arrays and values to Python native types.

    Args:
        obs (Dict[str, Any]): The observation dictionary.
        reward (float): The reward value.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode has been truncated.
        info (Dict[str, Any]): Additional information.

    Returns:
        Dict[str, Any]: The prepared payload.
    """

    def _convert_numpy(v):
        if isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.floating):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: _convert_numpy(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [_convert_numpy(item) for item in v]
        return v

    def _prepare_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
        return {k: _convert_numpy(v) for k, v in observation.items()}

    return {
        "observation": _prepare_observation(obs),
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": _convert_numpy(info),
    }


def call_agent_api(
    method: str,
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    logger: logging.Logger,
    player_id: int,
) -> Dict[str, Any]:
    """
    Make an API call to an agent with retry logic and failure tracking.

    Args:
        method (str): The HTTP method to use.
        base_url (str): The base URL of the agent's API.
        endpoint (str): The API endpoint to call.
        payload (Dict[str, Any]): The payload to send with the request.
        logger (logging.Logger): Logger instance.
        player_id (int): ID of the player (0 or 1).

    Returns:
        Dict[str, Any]: The JSON response from the API.

    Raises:
        TimeoutError: If the player exceeds their time limit.
        AgentFailure: If both players are failing or one player consistently fails.
    """
    max_retries = 5
    base_delay = 1

    try:
        for attempt in range(max_retries):
            try:
                response = requests.request(method, base_url + endpoint, json=payload, timeout=5.0)
                response.raise_for_status()
                failure_tracker.record_success(player_id)
                return response.json()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as e:
                if attempt == max_retries - 1:
                    raise

                delay = base_delay * (2**attempt)
                logger.info(f"Backing off for {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)

    except Exception as e:
        failure_tracker.record_failure(player_id)
        logger.error(f"Bot {player_id} failed API call: {str(e)}")
        raise


bankrolls = [0, 0]  # Track total bankrolls across all hands


def run_api_match(
    base_url_0: str,
    base_url_1: str,
    logger: logging.Logger,
    num_hands: int = 1000,
    csv_path: str = "./match.csv",
    team_0_name: str = "Team 0",
    team_1_name: str = "Team 1",
) -> Dict[str, Any]:
    """
    Run a match of multiple hands between two API-based agents.
    Each iteration creates a new PokerEnv instance representing one hand.
    """
    global bankrolls
    csv_headers = [
        "hand_number",
        "street",
        "active_team",
        "team_0_bankroll",
        "team_1_bankroll",
        "action_type",
        "action_amount",
        "team_0_cards",
        "team_1_cards",
        "board_cards",
        "team_0_discarded",
        "team_1_discarded",
        "team_0_bet",
        "team_1_bet",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        # Comment header
        csv_file.write(f"# Team 0: {team_0_name}, Team 1: {team_1_name}\n")

        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()

        for hand_number in range(num_hands):
            env = PokerEnv(logger=logger)  # env for a single hand
            (obs0, obs1), info = env.reset()
            try:
                res = play_hand(env, base_url_0, base_url_1, logger, writer, hand_number)
                bankrolls[0] += res["bot0_reward"]
                bankrolls[1] += res["bot1_reward"]
                if hand_number % 50 == 0:
                    logger.info(
                        f"Hand number: {hand_number}, {team_0_name} bankroll: {bankrolls[0]}, {team_1_name} bankroll: {bankrolls[1]}"
                    )
            except TimeoutError as te:
                # Determine winner based on which player exceeded time
                winner = 1 if "Player 0" in str(te) else 0
                return get_match_result("timeout", winner=winner)
            except AgentFailure as af:
                if "Player 0 has failed" in str(af):
                    return get_match_result("timeout", winner=1)
                elif "Player 1 has failed" in str(af):
                    return get_match_result("timeout", winner=0)
                return get_match_result("error", error=str(af))
            except Exception as e:
                logger.error(f"Unexpected error during hand {hand_number}: {e}")
                return get_match_result("error", error=str(e))

        logger.info("All hands completed")
        logger.info(f"Final results - {team_0_name} bankroll: {bankrolls[0]}, {team_1_name} bankroll: {bankrolls[1]}")
        logger.info(f"Time used - {team_0_name}: {time_used_0:.2f} seconds, {team_1_name}: {time_used_1:.2f} seconds")
        logger.info(f"Time limit: {TIME_LIMIT_SECONDS} seconds")

        return get_match_result("completed", rewards=(bankrolls[0], bankrolls[1]))


time_used_0 = 0.0
time_used_1 = 0.0


def play_hand(
    env: PokerEnv, base_url_0: str, base_url_1: str, logger: logging.Logger, writer: csv.DictWriter, hand_number: int
):
    """
    Play a single hand in the given environment instance.
    This function loops until the single hand terminates.
    """
    global time_used_0, time_used_1, bankrolls

    small_blind_player = hand_number % 2

    # Initialize per-hand variables
    (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind_player})
    info["hand_number"] = hand_number
    reward0 = reward1 = 0
    terminated = truncated = False

    # Loop until hand terminates
    while not terminated:
        bot0_payload = prepare_payload(obs0, reward0, terminated, truncated, info)
        bot1_payload = prepare_payload(obs1, reward1, terminated, truncated, info)

        current_player = obs0["acting_agent"]
        current_payload = bot0_payload if current_player == 0 else bot1_payload
        observer_payload = bot1_payload if current_player == 0 else bot0_payload
        current_url = base_url_0 if current_player == 0 else base_url_1
        observer_url = base_url_1 if current_player == 0 else base_url_0

        action_start = time.time()
        action = call_agent_api("GET", current_url, GET_ACTION_ENDPOINT, current_payload, logger, current_player)
        action_duration = time.time() - action_start

        # Update time tracking
        if current_player == 0:
            time_used_0 += action_duration
            if time_used_0 > TIME_LIMIT_SECONDS:
                raise TimeoutError("Player 0 exceeded time limit")
        else:
            time_used_1 += action_duration
            if time_used_1 > TIME_LIMIT_SECONDS:
                raise TimeoutError("Player 1 exceeded time limit")

        # Notify other player
        call_agent_api("POST", observer_url, SEND_OBS_ENDPOINT, observer_payload, logger, 1 - current_player)

        # Log action
        current_state = {
            "hand_number": hand_number,
            "street": get_street_name(obs0["street"]),
            "active_team": obs0["acting_agent"],
            "team_0_bankroll": bankrolls[0],
            "team_1_bankroll": bankrolls[1],
            "team_0_cards": [env.int_card_to_str(c) for c in env.player_cards[0] if c != -1],
            "team_1_cards": [env.int_card_to_str(c) for c in env.player_cards[1] if c != -1],
            "board_cards": [env.int_card_to_str(c) for c in env.community_cards[: obs0["street"] + 2] if c != -1],
            "team_0_discarded": env.int_card_to_str(env.discarded_cards[0]) if env.discarded_cards[0] != -1 else "",
            "team_1_discarded": env.int_card_to_str(env.discarded_cards[1]) if env.discarded_cards[1] != -1 else "",
            "team_0_bet": obs0["my_bet"] if obs0["acting_agent"] == 0 else obs0["opp_bet"],
            "team_1_bet": obs1["my_bet"] if obs1["acting_agent"] == 1 else obs1["opp_bet"],
            "action_type": PokerEnv.ActionType(action["action"][0]).name,
            "action_amount": action["action"][1],
        }
        writer.writerow(current_state)

        # Step environment
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action["action"])
        info["hand_number"] = hand_number  # Maintain hand number after each step

    return {"bot0_reward": reward0, "bot1_reward": reward1}


def get_match_result(
    status: str,
    winner: Optional[int] = None,
    rewards: Optional[Tuple[float, float]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standardized match result dictionary.

    Args:
        status (str): Match status ('completed', 'timeout', or 'error')
        winner (Optional[int]): Winner's player ID (0 or 1) for timeout cases
        rewards (Optional[Tuple[float, float]]): Final bankrolls (bot0_reward, bot1_reward)
        error (Optional[str]): Error message for error status

    Returns:
        Dict[str, Any]: Standardized match result containing:
            - status: 'completed', 'timeout', or 'error'
            - result: 'win' (player 0 won), 'loss' (player 1 won), 'tie', or 'invalid'
            - bot0_reward/bot1_reward: Final bankroll amounts (if available)
            - error: Error message (if status is 'error')
    """
    result = {"status": status}

    # Add result field for completed matches
    if status == "completed" and rewards:
        # Determine winner based on rewards
        if rewards[0] > rewards[1]:
            result["result"] = "win"  # Player 0 won
        elif rewards[1] > rewards[0]:
            result["result"] = "loss"  # Player 1 won
        else:
            result["result"] = "tie"  # Equal rewards
    elif status == "timeout" and winner is not None:
        # Convert timeout winner to win/loss
        result["result"] = "win" if winner == 0 else "loss"
    else:
        result["result"] = "invalid"

    # Add rewards if available
    if rewards:
        result["bot0_reward"] = rewards[0]
        result["bot1_reward"] = rewards[1]

    if error:
        result["error"] = error

    return result


def log_game_state(logger: logging.Logger, obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
    """
    Log the current game state.

    Args:
        logger (logging.Logger): The logger object to use for logging.
        obs0 (Dict[str, Any]): Observation for the first agent.
        obs1 (Dict[str, Any]): Observation for the second agent.
    """
    logger.debug("#####################")
    logger.debug(f"Turn: {obs0['acting_agent']}")
    logger.debug(f"Bot0 cards: {obs0['my_cards']}, Bot1 cards: {obs1['my_cards']}")
    logger.debug(f"Community cards: {obs0['community_cards']}")
    logger.debug(f"Bot0 bet: {obs0['my_bet']}, Bot1 bet: {obs1['my_bet']}")
    logger.debug("#####################")


def format_bankroll_log(game_number: int, bankrolls: list) -> str:
    """Format bankroll data as a JSON string for logging"""
    bankroll_data = {
        "type": "bankroll_update",
        "game_number": game_number,
        "bot0_bankroll": int(bankrolls[0]),
        "bot1_bankroll": int(bankrolls[1]),
    }
    return json.dumps(bankroll_data)
