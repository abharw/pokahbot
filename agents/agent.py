import logging
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

# I used a typedDict instead of a pydantic model because it
# was giving me issues.
class Observation(TypedDict):
    street: int
    acting_agent: int
    my_cards: List[int]
    community_cards: List[int]
    my_bet: int
    opp_bet: int
    opp_discarded_card: int
    opp_drawn_card: int
    min_raise: int
    max_raise: int
    valid_actions: List[int]


class ActionRequest(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Any


class ObservationRequest(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Any


class ActionResponse(BaseModel):
    action: Tuple[int, int, int]


class Agent(ABC):
    def __init__(self, stream: bool = False, player_id: str = None):
        self.app = FastAPI()
        self.player_id = player_id
        self.logger = self._setup_logger(stream)
        self.add_routes()

    def _setup_logger(self, stream: bool = False) -> logging.Logger:
        """Set up a logger for this agent instance"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicate logging
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Set up file logging in a local agent_logs directory
        match_id = os.getenv("MATCH_ID", "unknown")
        player_id = os.getenv("PLAYER_ID", "unknown")
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"match_{match_id}_{player_id}.log")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if stream:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    @abstractmethod
    def __name__(self):
        """Return the name of the agent. Must be implemented by subclasses."""
        pass

    def get_bot_action(self, observation, reward, terminated, truncated, info) -> tuple[int, int]:
        """
        Given the current state, return the action to take.
        """
        try:
            return self.act(observation, reward, terminated, truncated, info)
        except Exception as e:
            self.logger.error(f"Bot raised an error during act: {str(e)}.\n{traceback.format_exc()}")
            print(f"Bot raised an error during act: {str(e)}.\n{traceback.format_exc()}")

    def do_bot_observation(self, observation, reward, terminated, truncated, info):
        try:
            self.observe(observation, reward, terminated, truncated, info)
        except Exception as e:
            self.logger.error(f"Bot raised an error during observe: {str(e)}.\n{traceback.format_exc()}")
            print(f"Bot raised an error during observe: {str(e)}.\n{traceback.format_exc()}")


    @abstractmethod
    def act(self, observation, reward, terminated, truncated, info) -> tuple[int, int]:
        """
        Given the current state, return the action to take.

        Args:
            reward (int)  : 0 if terminated is false, or the profit / loss of the game
            #TODO: add the types of the arguments
        Returns:
            action (Tuple[int, int]) : (cumulative amount to bet, index of the card to discard)
        """
        pass

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        """
        Observe the result of your action. However, it's not your turn.
        """
        pass

    def add_routes(self):
        @self.app.get("/get_action")
        async def get_action(request: ActionRequest) -> ActionResponse:
            """
            API endpoint to get an action based on the current game state.
            """
            self.logger.debug(f"ActionRequest: {request}")
            try:
                action = self.get_bot_action(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
                self.logger.debug(f"Action taken: {action}")
                return ActionResponse(action=action)
            except Exception as e:
                self.logger.error(f"Error in get_action: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/post_observation")
        async def post_observation(request: ObservationRequest) -> None:
            """
            API endpoint to send the observation to the bot
            """
            self.logger.debug(f"Observation: {request}")
            try:
                self.do_bot_observation(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
            except Exception as e:
                self.logger.error(f"Error in post_observation: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    @classmethod
    def run(cls, stream: bool = False, port: int = 8000, host: str = "0.0.0.0", player_id: str = None):
        """Run an API-based bot on a specified port."""
        if player_id is not None:
            os.environ["PLAYER_ID"] = player_id
        bot = cls(stream)
        bot.logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(bot.app, host=host, port=port, log_level="info", access_log=False)
