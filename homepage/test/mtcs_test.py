
from homepage.src.nn_utils import get_cnn_value_model, get_cnn_model
from homepage.src.Player import Player, Value
from homepage.src.mcts import MCTS, ParallelMCTS
from homepage.src.go import GameState
import h5py
import time
import asyncio

if __name__ == '__main__':

    weights_set4p = h5py.File("homepage/weights/rl_weights19.hdf5", 'r')
    weights_set4v = h5py.File("homepage/weights/rl_value_weights0.hdf5", 'r')

    policy = Player()
    policy.model.load_weights(weights_set4p)

    value = Value()
    value.model.load_weights(weights_set4v)

    go_state = GameState()

    mcts = ParallelMCTS(value.get_value, policy.policy.eval_state, policy.policy.eval_state, n_search=5, n_executors=2)

    start_time = time.time()
    moves = mcts.parallel_get_move(go_state)
    end_time = time.time()
    print(end_time-start_time)
    print(moves)