from homepage.src.util import sgf_iter_states,_parse_sgf_move
import sgf
import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0

class State:
    def __init__(self, current_player_color=BLACK):
        self.board = np.zeros((19,19))
        self.current_player = current_player_color

    def get_state(self,move, color):
        self.board[move[0], move[1]] = color
        planes = np.zeros((19, 19, 3))
        planes[:, :, 0] = state.board == self.current_player  # own stone
        planes[:, :, 1] = state.board == -self.current_player  # opponent stone
        planes[:, :, 2] = state.board == EMPTY  # empty space
        return planes.reshape([1,19,19,3])

def convert_move2array(move):
    move_arr = np.zeros((1,19*19))
    ind = move[0]*19 + move[1]
    move_arr[0,ind] = 1
    return move_arr

def parse_sgf2tensor_input(sgf_file):
    collection = sgf.parse(sgf_file)
    game = collection[0]
    root = game.root
    print(root.current_prop_value[0])
    state_tensor = []
    move_tensor = []
    for node in game.rest:
        props = node.properties
        if 'W' in props:
            move = _parse_sgf_move(props['W'][0])
            player = WHITE
        elif 'B' in props:
            move = _parse_sgf_move(props['B'][0])
            player = BLACK
        state_tensor.append(state.get_state(move, player))
        move_tensor.append(convert_move2array(move))
    input_x = np.concatenate(state_tensor)
    input_y = np.concatenate(move_tensor)
    print(input_y.shape)

if __name__ == '__main__':

    state = State()

    sgf_file = open("../static/20160312-Lee-Sedol-vs-AlphaGo.sgf").read()
    parse_sgf2tensor_input(sgf_file)