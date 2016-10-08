
from homepage.src.nn_utils import get_cnn_model, get_cnn_value_model
from homepage.src.mcts import MCTS
import homepage.src.go as go
import numpy as np
import h5py

EMPTY = 0
WHITE = -1
BLACK = +1
BOARD_SIZE = 19

def flatten_idx(position, size):
    (x, y) = position
    return x * size + y

# this code is revised by the one i mentioned in readme
class BasePolicy:
    def _select_moves_and_normalize(self, nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        return zip(moves, distribution.tolist())

    def batch_eval_state(self, states, moves_lists=None):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.

        Analogous to [eval_state(s) for s in states]forward

        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []
        state_size = states[0].size
        if not all([st.size == state_size for st in states]):
            raise ValueError("all states must have the same size")
        # concatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)
        # pass all input through the network at once (backend makes use of batches if len(states) is large)
        network_output = self.model.get_predict(nn_input)

        # default move lists to all legal moves
        moves_lists = moves_lists or [st.get_legal_moves() for st in states]
        results = [None] * n_states
        for i in range(n_states):
            results[i] = self._select_moves_and_normalize(network_output[i], moves_lists[i], state_size)
        return results

    def eval_state(self, state, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.model.get_predict(tensor)[0]
        moves = moves or state.get_legal_moves()
        return self._select_moves_and_normalize(network_output, moves, state.size)


class GreedyPolicyPlayer(BasePolicy):
    """A player that uses a greedy policy (i.e. chooses the highest probability
    move each turn)
    """
    def __init__(self, model, process_state):
        self.model = model
        self.preprocessor = process_state

    def get_move(self, state):
        sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
        if len(sensible_moves) > 0:
            move_probs = self.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=lambda a_p:a_p[1])
            return max_prob[0]
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves() if not st.is_eye(move, st.current_player)] for st in states]
        all_moves_distributions = self.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            move_probs = list(move_probs)
            if len(move_probs) == 0:
                move_list[i] = go.PASS_MOVE
            else:
            # this 'else' clause is identical to ProbabilisticPolicyPlayer.get_move
                max_prob = max(move_probs, key=lambda m_p:m_p[1])
                move_list[i] = max_prob[0]
        return move_list


class ProbabilisticPolicyPlayer(BasePolicy):
    """A player that samples a move in proportion to the probability given by the
    policy.

    By manipulating the 'temperature', moves can be pushed towards totally random
    (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, model, process_state, temperature=1.0):
        assert(temperature > 0.0)
        self.model = model
        self.preprocessor = process_state
        self.beta = 1.0 / temperature

    def get_move(self, state):
        sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
        if len(sensible_moves) > 0:
            move_probs = self.eval_state(state, sensible_moves)
            moves, probabilities = zip(*move_probs)
            probabilities = np.array(probabilities)
            probabilities = probabilities ** self.beta
            probabilities = probabilities / probabilities.sum()
            choice_idx = np.random.choice(len(moves), p=probabilities)
            return moves[choice_idx]
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves() if not st.is_eye(move, st.current_player)] for st in states]
        all_moves_distributions = self.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            move_probs = list(move_probs)
            if len(move_probs) == 0:
                move_list[i] = go.PASS_MOVE
            else:
                # this 'else' clause is identical to ProbabilisticPolicyPlayer.get_move
                moves, probabilities = zip(*move_probs)
                probabilities = np.array(probabilities)
                probabilities = probabilities ** self.beta
                probabilities = probabilities / probabilities.sum()
                choice_idx = np.random.choice(len(moves), p=probabilities)
                move_list[i] = moves[choice_idx]
        return move_list


class Player:
    def __init__(self, player_color=None, feature_list=["board"]):
        self.model = get_cnn_model()

        self.player_color = player_color
        self.process_state = ProcessState(feature_list)
        self.policy = ProbabilisticPolicyPlayer(self.model, self.process_state)

    def get_move(self, state):
        return self.policy.get_move(state)

    def get_moves(self, states):

        return self.policy.get_moves(states)

    def decode_move(self,index):
        x = int(index / BOARD_SIZE)
        y = index % BOARD_SIZE
        return (x, y)

    def set_color(self, player_color):
        self.player_color = player_color


class Value:
    def __init__(self, player_color=None, feature_list=["board"]):
        self.model = get_cnn_value_model()

        self.process_state = ProcessState(feature_list)

    def get_value(self, state):
        nn_input = self.process_state.state_to_tensor(state)
        return self.model.get_predict(nn_input)

class MCTS_Player:
    def __init__(self, policy_weigths_dir, value_weights_dir, player_color=None, feature_list=["board"]
                 ):
        self.model = get_cnn_model()

        self.player_color = player_color
        self.process_state = ProcessState(feature_list)

        weights_set4p = h5py.File(policy_weigths_dir, 'r')
        weights_set4v = h5py.File(value_weights_dir, 'r')

        self.policy = Player()
        self.policy.model.load_weights(weights_set4p)

        self.value = Value()
        self.value.model.load_weights(weights_set4v)



    def get_move(self, state, lmbda=0.5, c_puct=5, rollout_limit=500, playout_depth=20, n_search=10):
        policy = MCTS(self.value.get_value, self.policy.policy.eval_state, self.policy.policy.eval_state,
                           lmbda, c_puct, rollout_limit, playout_depth, n_search)
        return policy.get_move(state)


def get_borad_feature2tensor(state):
    """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    # WHITE = -1, BLACK = 1 if current player is BLACK, vice versa
    planes = np.zeros((19, 19, 3), dtype=np.float32)
    planes[:, :, 0] = state.board == state.current_player  # own stone
    planes[:, :, 1] = state.board == -state.current_player  # opponent stone
    planes[:, :, 2] = state.board == EMPTY  # empty space
    return planes


## this features contain various method that compute for representing Go. the default one is make what people always see(board) as the input feature
FEATURES = {
    'board':{
        'size':3,
        "function":get_borad_feature2tensor
    }
}

# this code is revised by the one i mentioned in readme
class ProcessState:
    def __init__(self, features_list):
        self.output_dim = 0
        self.processors = [None] * len(features_list)

        for f_ind in range(len(features_list)):
            feature = features_list[f_ind].lower()
            if feature in FEATURES:
                self.processors[f_ind] = FEATURES[feature]['function']
                self.output_dim += FEATURES[feature]['size']
            else:
                raise ValueError("uknown feature: %s" % feature)


    def state_to_tensor(self, state):

        feat_tensors = [proc(state) for proc in self.processors]
        f, s = self.output_dim, state.size
        return np.concatenate(feat_tensors).reshape((1, s, s, f))