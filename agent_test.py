"""
Basic Test Suite for PlayerAgent which checks that it never does an invalid action
"""

import importlib.util
import multiprocessing
import os
import sys
import time
import logging
from logging import getLogger
from typing import Optional, Type

from agents.agent import Agent
from agents.test_agents import AllInAgent, CallingStationAgent, FoldAgent, RandomAgent
from match import run_api_match

NUM_HANDS = 5
TIME_PER_HAND = 5


def verify_submission() -> Optional[str]:
    """
    Verify that the submission contains required files and can be imported

    Args:
        submission_dir: Path to the submission directory

    Returns:
        Optional[str]: Error message if verification fails, None if successful
    """
    if not os.path.isdir("submission"):
        return "Submission directory not found"

    if not os.path.isfile("submission/player.py"):
        return "Required file 'player.py' not found in submission directory"

    try:
        spec = importlib.util.spec_from_file_location("player", "submission/player.py")
        if spec is None or spec.loader is None:
            return "Could not load player.py"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "PlayerAgent"):
            return "player.py does not contain PlayerAgent class"

        if not issubclass(module.PlayerAgent, Agent):
            return "PlayerAgent must inherit from Agent class"

    except Exception as e:
        return f"Error importing PlayerAgent: {str(e)}"

    return None


def get_player_agent() -> Optional[Type[Agent]]:
    """
    Import and return the PlayerAgent class

    Returns:
        Optional[Type[Agent]]: PlayerAgent class if successful, None if import fails
    """
    try:
        from submission.player import PlayerAgent

        return PlayerAgent
    except ImportError:
        return None


def run_test_match(test_agent_class: Agent, logger):
    """
    Run a match between PlayerAgent and a test agent using the API interface

    Args:
        test_agent_class (Agent): The test agent class to play against
        logger: Logger instance

    Returns:
        dict: Match results

    Raises:
        RuntimeError: If there are initialization or runtime errors
    """
    PlayerAgent = get_player_agent()
    if PlayerAgent is None:
        raise RuntimeError("Could not import PlayerAgent")

    process0 = multiprocessing.Process(target=PlayerAgent.run, args=(False, 8000))
    process1 = multiprocessing.Process(target=test_agent_class.run, args=(False, 8001))

    try:
        process0.start()
        process1.start()

        result = run_api_match("http://127.0.0.1:8000", "http://127.0.0.1:8001", logger, num_hands=NUM_HANDS, csv_path=f"./match_{test_agent_class.__name__}.csv")

        return result

    finally:
        process0.terminate()
        process1.terminate()
        process0.join()
        process1.join()


def main():
    """
    Runs a test suite of games between PlayerAgent and various test agents to verify:
    1. The submission contains required files
    2. The agent can be imported successfully
    3. The agent can be initialized and run as an API server
    4. The agent can play full games without crashing
    5. The agent responds within time limits
    
    Returns:
        dict: Test results containing:
            - verification_error: str or None
            - games_completed: int
            - runtime_errors: int
            - timeout_errors: int
            - passed: bool
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = getLogger(__name__)

    verification_error = verify_submission()
    if verification_error:
        print(f"Submission verification failed: {verification_error}")
        return {
            "verification_error": verification_error,
            "games_completed": 0,
            "runtime_errors": 0,
            "timeout_errors": 0,
            "passed": False
        }

    test_results = {"games_completed": 0, "runtime_errors": 0, "timeout_errors": 0}

    test_agents = [AllInAgent, FoldAgent, CallingStationAgent, RandomAgent]

    for test_agent_class in test_agents:
        print(f"\nTesting user bot against {test_agent_class.__name__}")
        print("-" * 50)

        start_time = time.time()

        try:
            result = run_test_match(test_agent_class, logger)

            if result["status"] == "completed":
                test_results["games_completed"] += NUM_HANDS
                print(f"✓ Completed {NUM_HANDS} games successfully")
            elif result["status"] == "timeout":
                test_results["timeout_errors"] += 1
                print(f"✗ Failed: Time limit exceeded")
            else:
                test_results["runtime_errors"] += 1
                print(f"✗ Failed: Runtime error")
                print(f"  {result.get('error', 'Unknown error')}")

        except Exception as e:
            test_results["runtime_errors"] += 1
            print(f"✗ Failed: Runtime error")
            print(f"  {str(e)}")
            continue

        end_time = time.time()
        time_per_hand = (end_time - start_time) / NUM_HANDS

        if time_per_hand > TIME_PER_HAND:
            test_results["timeout_errors"] += 1
            print(f"✗ Time limit exceeded: {time_per_hand:.2f}s per hand (limit: {TIME_PER_HAND}s)")
        else:
            print(f"✓ Time check passed: {time_per_hand:.2f}s per hand")

    print("\nTest Suite Summary")
    print("-" * 50)
    print(f"Games completed successfully: {test_results['games_completed']}")
    print(f"Runtime errors encountered: {test_results['runtime_errors']}")
    print(f"Time limit violations: {test_results['timeout_errors']}")

    test_results["verification_error"] = None
    test_results["passed"] = (
        test_results["games_completed"] > 0 and
        test_results["runtime_errors"] == 0 and
        test_results["timeout_errors"] == 0
    )
    
    return test_results


if __name__ == "__main__":
    results = main()
    if not results["passed"]:
        sys.exit(1)
