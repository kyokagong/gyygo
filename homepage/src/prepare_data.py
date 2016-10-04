
import numpy as np
import h5py
import os
import sgf

from homepage.src.util import sgf_iter_states,_parse_sgf_move


WHITE = -1
BLACK = +1
EMPTY = 0


class State:
    def __init__(self, current_player_color=BLACK):
        self.board = np.zeros((19,19), dtype=np.float32)
        self.current_player = current_player_color

    def get_state(self,move, color):
        self.board[move[0], move[1]] = color
        planes = np.zeros((19, 19, 3), dtype=np.float32)
        planes[:, :, 0] = self.board == self.current_player  # own stone
        planes[:, :, 1] = self.board == -self.current_player  # opponent stone
        planes[:, :, 2] = self.board == EMPTY  # empty space
        return planes.reshape([1,19,19,3])

class Prepare:
    def __init__(self):
        self.num_files = 0

        self.np_state_node_list = []
        self.np_move_node_list = []

    def get_data(self, file):
        try:
            sgf_str = open(file, 'r').read()
            (game, winner) = self.parse_sgf(sgf_str)
            self.convert_state(game, winner)
            self.num_files += 1
        except Exception as e:
            print(e)

    def save_data_node_as_hdf5(self, output_dir):
        f = h5py.File(output_dir, "w")
        dat_x = np.concatenate(self.np_state_node_list)
        dat_y = np.concatenate(self.np_move_node_list)

        f.create_dataset("dat_x", data=dat_x)
        f.create_dataset("dat_y", data=dat_y)


    def convert_state(self, game, winner):
        def convert_move2array(move):
            move_arr = np.zeros((1,19*19))
            ind = move[0]*19 + move[1]
            move_arr[0,ind] = 1
            return move_arr

        state = State(winner)
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
            if player == winner:
                state_tensor.append(state.get_state(move, player))
                move_tensor.append(convert_move2array(move))
        state_node = np.concatenate(state_tensor)
        move_node = np.concatenate(move_tensor)
        self.np_state_node_list.append(state_node)
        self.np_move_node_list.append(move_node)

    def parse_sgf(self, sgf_string):
            collection = sgf.parse(sgf_string)
            game = collection[0]
            re = game.root.properties["RE"][0]
            winner = self.get_winner(re)
            return (game, winner)


    def get_winner(self, result):
        print(result[0])
        if result[0] == 'W':
            winner = WHITE
        elif result[0] == "B":
            winner = BLACK
        else:
            raise Exception("no winner")
        return winner

    def search_data(self, file):
        if self.num_files < 5000:
            if os.path.isfile(file):
                self.get_data(file)
            else:
                for f in os.listdir(file):
                    self.search_data(file+"/"+f)



def prepare_data():

    pass

if __name__ == '__main__':
    file_dir = "../data/"
    output_dir = "../train_data/train_data1.hdf5"

    p = Prepare()
    p.search_data(file_dir)
    p.save_data_node_as_hdf5(output_dir)