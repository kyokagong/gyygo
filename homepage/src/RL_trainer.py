
import numpy as np
import homepage.src.go as go
from homepage.src.go import GameState
from homepage.src.Player import ProcessState
import tensorflow as tf

###

def flatten_idx(position, size):
    (x, y) = position
    return x * size + y

def decode_move(self,index, size):
    x = int(index / size)
    y = index % size
    return (x, y)

"""making a batch train set by RL(self play)"""
# mode is either policy or value
def make_training_pairs4RL(player, opponent, features, mini_batch_size, mode='policy', board_size=19):

    def do_move(states, moves, X_list, y_list, player_color):
        bsize_flat = bsize * bsize
        for st, mv, X, y in zip(states, moves, X_list, y_list):
            # Only do more moves if not end of game already
            if not st.is_end_of_game:
                if st.current_player == player_color and mv is not go.PASS_MOVE:
                    # Convert move to one-hot
                    state_1hot = processor.state_to_tensor(st)
                    move_1hot = np.zeros(bsize_flat)
                    move_1hot[flatten_idx(mv, bsize)] = 1
                    X.append(state_1hot)
                    y.append(move_1hot)
                st.do_move(mv)
        return states, X_list, y_list

    # Lists of game training pairs (1-hot)
    X_list = [list() for _ in range(mini_batch_size)]
    y_list = [list() for _ in range(mini_batch_size)]
    processor = ProcessState(features)
    # bsize = player.policy.model.input_shape[-1]
    bsize = board_size
    states = [GameState(size=board_size) for i in range(mini_batch_size)]
    # Randomly choose who goes first (i.e. color of 'player')
    player_color = np.random.choice([go.BLACK, go.WHITE])
    player1, player2 = (player, opponent) if player_color == go.BLACK else \
        (opponent, player)

    shou = 0
    while True:
        # Get moves (batch)
        moves_black = player1.get_moves(states)
        # Do moves (black)
        states, X_list, y_list = do_move(states, moves_black, X_list, y_list, player_color)
        # Do moves (white)
        moves_white = player2.get_moves(states)
        # print("step: %s, black: %s, white: %s " % (shou, moves_black, moves_white))
        shou += 1
        states, X_list, y_list = do_move(states, moves_white, X_list, y_list, player_color)
        # If all games have ended, we're done. Get winners.
        done = [st.is_end_of_game for st in states]
        if all(done):
            break
    won_game_list = [st.get_winner() == player_color for st in states]
    # Concatenate tensors across turns within each game
    if mode == 'value':
        for i in range(mini_batch_size):
            X_list[i] = np.concatenate(X_list[i], axis=0)
            y_list[i] = np.ones((X_list[i].shape[0], 1))
            if not won_game_list[i]:
                y_list[i][:] = -1
    else:
        for i in range(mini_batch_size):
            X_list[i] = np.concatenate(X_list[i], axis=0)
            y_list[i] = np.vstack(y_list[i])

    return X_list, y_list, won_game_list

def train_rl_batch(player, X_list, y_list, won_game_list, lr):
    """Given the outcomes of a mini-batch of play against a fixed opponent,
        update the weights with reinforcement learning.

        Args:
        player -- player object with policy weights to be updated
        X_list -- List of one-hot encoded states.
        y_list -- List of one-hot encoded actions (to pair with X_list).
        winners -- List of winners corresponding to each item in
                training_pairs_list
        lr -- Keras learning rate

        Return:
        player -- same player, with updated weights.
        """

    for X, y, won_game in zip(X_list, y_list, won_game_list):
        # Update weights in + direction if player won, and - direction if player lost.
        # Setting learning rate negative is hack for negative weights update.
        if won_game:
            lr = lr
        else:
            lr = -lr
        player.model.fit(X, y, epoch=1, batch_size=X.shape[0], lr=lr)


def train_rl_value_batch(value_model, X_list, y_list, lr, verbose=False):
    for X, y in zip(X_list, y_list):
        value_model.fit(X,y, epoch=1, batch_size=X.shape[0], lr=lr, verbose=verbose)


