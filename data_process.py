import pandas as pd
from collections import defaultdict

def analyze_poker_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Extract relevant data
    player_1_wins = 0
    total_games = df['hand_number'].nunique()
    player_1_never_raised = set()
    player_1_win_no_raise = 0
    
    # Tracking opponent tendencies
    opponent_raises = defaultdict(int)
    opponent_all_in = defaultdict(int)
    opponent_calls_strong = defaultdict(int)
    opponent_actions = []
    
    player_1_aggression = 0  # Measure of how loose or tight they are
    opponent_aggression = 0
    player_1_raises = 0
    opponent_raises_count = 0
    
    for hand_number, hand_data in df.groupby('hand_number'):
        # Determine winner (bankroll increase for Player 1)
        final_row = hand_data.iloc[-1]
        if final_row['team_1_bankroll'] > 0:
            player_1_wins += 1
        
        # Track if Player 1 raised
        player_1_raised = any(
            (hand_data['active_team'] == 1) & (hand_data['action_type'] == 'RAISE')
        )
        if not player_1_raised:
            player_1_never_raised.add(hand_number)
            if final_row['team_1_bankroll'] > 0:
                player_1_win_no_raise += 1
        
        # Track player 1's raise count
        player_1_raises += hand_data[(hand_data['active_team'] == 1) & (hand_data['action_type'] == 'RAISE')].shape[0]
        
        # Track opponent tendencies
        for _, row in hand_data.iterrows():
            if row['active_team'] == 0:
                if row['action_type'] == 'RAISE':
                    opponent_raises[row['team_0_cards']] += 1
                    opponent_raises_count += 1
                elif row['action_type'] == 'ALL-IN':
                    opponent_all_in[row['team_0_cards']] += 1
                elif row['action_type'] == 'CALL' and row['team_1_bet'] > 50:  # Strong bet threshold
                    opponent_calls_strong[row['team_0_cards']] += 1
                opponent_actions.append(row['action_type'])
                opponent_aggression += 1 if row['action_type'] in ['RAISE', 'ALL-IN'] else 0
    
    # Calculate statistics
    winrate = player_1_wins / total_games
    winrate_no_raise = player_1_win_no_raise / len(player_1_never_raised) if player_1_never_raised else 0
    player_1_tightness = 1 - (player_1_raises / df[df['active_team'] == 1].shape[0])
    opponent_tightness = 1 - (opponent_raises_count / len(opponent_actions))
    
    return {
        'Player 1 Win Rate': winrate,
        'Win Rate Without Raising': winrate_no_raise,
        'Opponent Raise Hands': dict(opponent_raises),
        'Opponent All-In Hands': dict(opponent_all_in),
        'Opponent Calls Strong Bet Hands': dict(opponent_calls_strong),
        'Opponent Consistency in Aggression': opponent_aggression / len(opponent_actions),
        'Player 1 Tightness Score': player_1_tightness,
        'Opponent Tightness Score': opponent_tightness
    }

# Example usage:
results = analyze_poker_csv('poker_game_log.csv')
print(results)
