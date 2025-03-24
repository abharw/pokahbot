# Advanced Poker AI Bot Implementation

This repository contains a neural network-based poker AI for the modified Texas Hold'em tournament. The implementation uses TensorFlow to create a policy network that makes intelligent poker decisions.

## Project Structure

```
├── submission/               # Tournament submission folder
│   ├── player.py             # Main bot entry point
│   ├── model.py              # Neural network model architecture
│   ├── poker_utils.py        # Poker game utilities and encoders
│   └── model/                # Pre-trained model weights (created during training)
│       └── poker_model.*     # SavedModel files
├── train_model.py            # Training script (not part of submission)
└── README.md                 # This file
```

## Implementation Details

### 1. Neural Network Approach

The bot uses a deep neural network architecture with:
- Input layer that encodes cards, game state, and betting information
- Hidden layers with ReLU activation, batch normalization, and dropout
- Multiple output heads for:
  - Action selection (fold, raise, check, call)
  - Raise sizing
  - Discard decision

### 2. Card and Game State Encoding

- Cards are one-hot encoded based on the 27-card deck
- Game state includes:
  - Current street
  - Betting information
  - Hand strength evaluation
  - Discard information
  - Opponent modeling

### 3. Decision Making Process

For each decision, the bot:
1. Encodes the current game state
2. Runs it through the neural network
3. Applies a mask to filter out invalid actions
4. Selects the highest probability action
5. Determines raise amount and discard strategy if applicable

### 4. Time Management

The bot includes careful time management:
- Adjusts thinking time based on remaining time budget
- Falls back to simpler rule-based decisions when time is critical
- Tracks and analyzes decision time to avoid timeouts

### 5. Opponent Modeling

The bot builds a model of the opponent's play style:
- Tracks fold and raise frequencies
- Calculates aggression factor
- Adapts strategy based on opponent tendencies

## Training the Model

The included training script uses self-play reinforcement learning to train the model:

```bash
python train_model.py --episodes 10000 --batch_size 64 --learning_rate 1e-4
```

Key training parameters:
- `--episodes`: Number of episodes to train (default: 10000)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--gamma`: Reward discount factor (default: 0.99)
- `--epsilon`: Exploration rate (default: 0.1)
- `--checkpoint_interval`: Save model every N episodes (default: 1000)
- `--load_checkpoint`: Load existing checkpoint (optional)

## Hand Strength Evaluation

The bot evaluates hand strength using:
1. Current hole cards
2. Visible community cards
3. Knowledge of the 27-card deck structure
4. Special handling for straights, flushes, and other hand types

## Strategy Considerations

The bot uses several strategic elements:
- **Position-based adjustments**: Different strategies for early vs. late position
- **Pot odds calculation**: Compare pot odds to hand strength for calling decisions
- **Bluffing**: Occasional bluffs with weak hands based on configurable thresholds
- **Value betting**: Stronger bets with high-value hands
- **Discard strategy**: Evaluates individual cards and discards weaker ones

## Performance Optimization

- The model uses `tf.function` decorators for faster inference
- Decision-making time is carefully managed to avoid timeouts
- Fallback to rule-based decisions when time is critical

## Additional Considerations

1. **Bankroll Management**: The bot tracks performance to identify and adjust underperforming strategies
2. **Debug Mode**: Extensive logging for analyzing decisions (disabled in tournament play)
3. **Error Handling**: Robust error handling to avoid crashes during tournament play

## Usage

1. Train the model using the training script
2. The trained model weights will be saved to `submission/model/`
3. Submit the entire `submission/` folder to the tournament platform

## Requirements

- TensorFlow 2.x
- NumPy
- Python 3.12