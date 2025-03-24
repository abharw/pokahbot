import logging
import multiprocessing
import json
import importlib

from match import run_api_match

def load_agent_class(file_path):
    """
    Dynamically imports and returns an agent class from a string path.
    Example: 'agents.test_agents.AllInAgent' -> AllInAgent class
    """
    module_path, class_name = file_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    # Load configuration
    with open('agent_config.json', 'r') as f:
        config = json.load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load agent classes dynamically
    bot0_class = load_agent_class(config['bot0']['file_path'])
    bot1_class = load_agent_class(config['bot1']['file_path'])

    # Create processes using the configuration
    process0 = multiprocessing.Process(
        target=bot0_class.run,
        args=(False, config['bot0']['port']),
        kwargs = {"player_id": config['bot0']['player_id']}
    )
    process1 = multiprocessing.Process(
        target=bot1_class.run,
        args=(False, config['bot1']['port']),
        kwargs = {"player_id": config['bot1']['player_id']}
    )

    process0.start()
    process1.start()

    logger.info("Starting API-based match")
    result = run_api_match(
        f"http://localhost:{config['bot0']['port']}",
        f"http://localhost:{config['bot1']['port']}",
        logger,
        csv_path=config['match_settings']['csv_output_path'],
        team_0_name=bot0_class.__name__,
        team_1_name=bot1_class.__name__
    )
    logger.info(f"Match result: {result}")

    # Clean up processes
    process0.terminate()
    process1.terminate()
    process0.join()
    process1.join()


if __name__ == "__main__":
    main()
