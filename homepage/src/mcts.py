import numpy as np


class TreeNode(object):
    """Tree Representation of MCTS that covers Selection, Expansion, Evaluation, and backUp (aka 'update()')
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.nVisits = 0
        self.Q_value = 0
        self.u_value = prior_p
        self.children = {}
        self.P = prior_p

    def expansion(self, actions):
        """Expand subtree - a dictionary with a tuple of (x,y) position as keys, TreeNode object as values

        Keyword arguments:
        Output from policy function - a list of tuples of (x, y) position and prior probability

        Returns:
        None
        """
        for action, prob in actions:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def selection(self):
        """Select among subtree to get the position that gives maximum action value Q plus bonus u(P)

        Keyword arguments:
        None.

        Returns:
        a tuple of (action, next_node)
        """
        # python3 is use items instead of iteritems
        return max(self.children.items(), key=lambda a_n: a_n[1].toValue())

    def isLeaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)
        """
        return self.children == {}

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation

        Arguments:
        value of traversed subtree evaluation

        Returns:
        None
        """
        # count visit
        self.nVisits += 1
        # update Q
        mean_V = self.Q_value * (self.nVisits - 1)
        self.Q_value = (mean_V + leaf_value) / self.nVisits
        # update u (note that u is not normalized to be a distribution)
        self.u_value = c_puct * self.P * np.sqrt(self.parent.nVisits) / (1 + self.nVisits)

    def toValue(self):
        """Return action value Q plus bonus u(P)
        """
        return self.Q_value + self.u_value


class MCTS(object):
    """Monte Carlo tree search, takes an input of game state, value network function, policy network function,
    rollout policy function. get_move outputs an action after lookahead search is complete.

    The value function should take in a state and output a number in [-1, 1]
    The policy and rollout functions should take in a state and output a list of (action,prob) tuples where
        action is an (x,y) tuple

    lmbda and c_puct are hyperparameters. 0 <= lmbda <= 1 controls the relative weight of the value network and fast
    rollouts in determining the value of a leaf node. 0 < c_puct < inf controls how quickly exploration converges
    to the maximum-value policy
    """

    def __init__(self, value_network, policy_network, rollout_policy, lmbda=0.5, c_puct=5, rollout_limit=500, playout_depth=20, n_search=10):
        self.root = TreeNode(None, 1.0)
        self._value = value_network
        self._policy = policy_network
        self._rollout = rollout_policy
        self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_search = n_search

    def _DFS(self, nDepth, treenode, state):
        """Monte Carlo tree search over a certain depth per simulation, at the end of simulation,
        the action values and visits of counts of traversed treenode are updated.

        Keyword arguments:
        Initial GameState object
        Initial TreeNode object
        Search Depth

        Returns:
        None
        """

        visited = [None] * nDepth

        # Playout to nDepth moves using the full policy network
        for index in range(nDepth):
            action_probs = self._policy(state)
            # check for end of game
            action_probs = list(action_probs)
            if len(action_probs) == 0:
                break
            treenode.expansion(action_probs)
            action, treenode = treenode.selection()
            state.do_move(action)
            visited[index] = treenode

        # leaf evaluation
        v = self._value(state)
        z = self._evaluate_rollout(state, self._rollout_limit)
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        # update value and visit count of nodes in this traversal
        # Note: it is important that this happens from the root downward
        # so that 'parent' visit counts are correct
        for node in visited:
            node.update(leaf_value, self._c_puct)

    def _evaluate_rollout(self, state, limit):
        """Use the rollout policy to play until the end of the game, get the winner (or 0 if tie)
        """
        for i in range(limit):
            action_probs = self._rollout(state)
            action_probs = list(action_probs)
            if len(action_probs) == 0:
                break
            max_action = max(action_probs, key=lambda a_p: a_p[1])[0]
            state.do_move(max_action)
        else:
            # if no break from the loop
            print("WARNING: rollout reached move limit")
        return state.get_winner()

    def get_move(self, state):
        """After running simulations for a certain number of times, when the search is complete, an action is selected
        from root state

        Keyword arguments:
        Number of Simulations

        Returns:
        action -- a tuple of (x, y)
        """
        action_probs = self._policy(state)
        self.root.expansion(action_probs)

        for n in range(0, self._n_search):
            state_copy = state.copy()
            self._DFS(self._L, self.root, state_copy)

        # chosen action is the *most visited child*, not the highest-value
        # (note that they are the same as self._n_search gets large)
        return max(self.root.children.items(), key=lambda a_n: a_n[1].nVisits)[0]

    def update_with_move(self, last_move):
        """step forward in the tree and discard everything that isn't still reachable
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
            # siblings of root will be garbage-collected because they are no longer reachable
        else:
            self.root = TreeNode(None, 1.0)


# future part
class ParallelMCTS(MCTS):
    pass


