
from homepage.src.nn_utils import CNN, ConvPoolLayer
import numpy as np

DEFAULTS = {
            'board':19,
            'action':361,
            'channel':1,
        }

class CNNPolicy():

    def select_move(self, action_pred):
        action_ind = np.argmax(action_pred)
        move_x = action_ind // DEFAULTS['board']
        move_y = action_ind % DEFAULTS['board']
        return (move_x, move_y)

    def eval_state_(self,states):
        actions_pred = self.cnn.get_predict().eval(feed_dict={self.cnn.input_node:states})


    def create_graph(self):


        params = DEFAULTS

        self.cnn = CNN(params['board'],params['channel'],params['action'])

        next_cores_size = 8

        self.cnn.add(ConvPoolLayer(filter_shape=[3,3,params['channel'],next_cores_size]))
        for i in range(1,4):
            self.cnn.add(ConvPoolLayer(filter_shape=[3,3,next_cores_size,next_cores_size*2]))
            next_cores_size *= 2

        self.cnn.make_net()

    def fit(self, batch_input_x, batch_intpu_y, batch_size):
        self.cnn.fit(batch_input_x,batch_intpu_y,1)