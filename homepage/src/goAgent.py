
import random
import numpy as np
from homepage.src.go import GameState
from homepage.src.nn_utils import CNN, ConvLayer, PoolLayer, ConvPoolLayer, error_rate, get_predict_by_batch
from homepage.src.Player import Player, MCTS_Player
import h5py


EMPTY = 0
WHITE = -1
BLACK = +1
BOARD_SIZE = 19

BEST_WEIGHTS_PATH = "homepage/weights/rl_weights19.hdf5"
BEST_WEIGHTS_PATH2 = "homepage/weights/rl_value_weights0.hdf5"

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


class TestGo():
    def __init__(self, user_color=BLACK):

        # agent is temperately white
        self.user_color = BLACK
        self.agent_color = WHITE if self.user_color == BLACK else BLACK
        self.agent = MCTS_Player(BEST_WEIGHTS_PATH,BEST_WEIGHTS_PATH2,player_color=WHITE)
        self.game_state = GameState()
        # self.set_default_agent()

    # def set_default_agent(self):
    #     self.agent.model.save_weights(BEST_WEIGHTS_PATH)

    def test_init(self, x, y):
        self.x = x
        self.y = y

    def user_move(self,move):
        self.game_state.do_move(move, self.user_color)
        print(self.game_state.board)

    def agent_move(self):
        move = self.agent.get_move(self.game_state,n_search=2)
        self.game_state.do_move(move, self.agent_color)
        return move

    def randomMove(self, rival_x, rival_y):
        self.board[rival_x,rival_y] = BLACK
        is_continue = True
        while is_continue:
            x = random.randint(1,19)
            y = random.randint(1,19)
            if self.board[x,y] == EMPTY:
                self.board[x,y] = WHITE
                is_continue = False
        return (x, y)


    def decode_move(self,index):
        x = int(index / BOARD_SIZE)
        y = index % BOARD_SIZE
        return (x, y)

