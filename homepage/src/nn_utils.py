import tensorflow as tf
import numpy as np
import h5py
import math
from homepage.src.tf_utils import SessionHandler

VALID_METHOD = "VALID"
SAME_METHOD = "SAME"

LAYER_TYPE = ["convpool", "conv", "pool", "fc"]

IS_GPU = False

INIT_VAL = 0.1

"""the basic layer class"""
class CommonLayer(object):
    def get_filter(self):
        return self.filter_shape

    ## padding method is either VALID or SAME
    def get_padding_method(self):
        return self.padding

    def get_output(self, input):
        pass

    def set_weights(self, weights):
        self.init_weights = weights

    def init_net(self):
        if IS_GPU:
            self._init_net()
        else:
            with tf.device("/cpu:0"):
                self._init_net()


class ConvLayer(CommonLayer):
    def __init__(self, filter_shape, padding="SAME", weights=None):
        self.layer_type = "conv"

        self.filter_shape = filter_shape
        self.padding = padding

        self.init_weights = weights

        self.update_weights_op = None

    def set_init_values(self, weights=None):
        self.init_weights = weights

    def _init_net(self):
        if self.init_weights is None:
            self.init_weights = tf.random_uniform(self.filter_shape,
                              minval = -INIT_VAL,
                              maxval = INIT_VAL
                            )
        self.conv_weights = tf.Variable(self.init_weights)

    def get_output(self, input):
        self.input = input
        conv = tf.nn.conv2d(input=self.input,
                        filter=self.conv_weights,
                        strides=[1, 1, 1, 1],
                        padding=self.padding)

        # relu = tf.nn.relu(conv)
        return conv

    ## padding method is either VALID or SAME
    def get_padding_method(self):
        return self.padding

    def get_filter(self):
        return self.filter_shape

    def get_weights(self):
        return self.conv_weights

    def set_weights(self, weights):
        return self.conv_weights.assign(weights)

    def set_placeholder_weights(self, weights):
        if self.update_weights_op is None:
            self.update_placeholder_weights = tf.placeholder(self.init_weights.dtype,
                                                             shape=self.conv_weights.get_shape())
            self.update_weights_op = self.conv_weights.assign(self.update_placeholder_weights)
        SessionHandler().get_session().run(self.update_weights_op,
                                           {self.update_placeholder_weights:weights})


class PoolLayer(CommonLayer):
    def __init__(self, padding="SAME"):
        self.layer_type = "pool"
        self.padding = padding


    def init_net(self):
        pass

    def get_output(self, input):
        self.input = input
        pool = tf.nn.max_pool(self.input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=self.padding)
        return pool

    ## padding method is either VALID or SAME
    def get_padding_method(self):
        return self.padding


class ConvPoolLayer(ConvLayer):
    def __init__(self,filter_shape, weights=None, is_add_bias=False):
        self.filter_shape = filter_shape
        self.layer_type = "convpool"
        self.padding = "SAME"
        self.init_weights = weights


    def _init_net(self):
        if self.init_weights is None:
            self.init_weights = tf.random_uniform(self.filter_shape,
                              minval = -INIT_VAL,
                              maxval = INIT_VAL
                            )
        self.conv_weights = tf.Variable(self.init_weights)

    def get_output(self, input):
        self.input = input
        conv = tf.nn.conv2d(input=self.input,
                        filter=self.conv_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

        return pool


class FullForwardLayer(CommonLayer):
    def __init__(self, input_size, output_size, weights=None, biases=None):
        self.layer_type = "fc"
        self.input_size = input_size
        self.output_size = output_size

        self.init_fc_weights = weights
        self.init_fc_biases = biases

        self.update_fc_weights_op = None
        self.update_fc_biases_op = None

    def set_init_values(self, weights=None, biases=None):
        self.init_fc_weights = weights
        self.init_fc_biases = biases

    def init_net(self):
        if self.init_fc_weights is None:
            # fully connected, depth 512
            self.init_fc_weights = tf.random_uniform([self.input_size, self.output_size],
                                                minval = -INIT_VAL,
                                                maxval = INIT_VAL)

        if self.init_fc_biases is None:
            self.init_fc_biases = tf.constant(0.0, shape=[self.output_size])

        self.fc_weights = tf.Variable(self.init_fc_weights)
        self.fc_biases = tf.Variable(self.init_fc_biases)

    def get_output(self, input):
        self.input = input
        return tf.matmul(self.input, self.fc_weights) + self.fc_biases

    def get_weights(self):
        return self.fc_weights

    def get_biases(self):
        return self.fc_biases

    def set_weights(self, weights):
        return self.fc_weights.assign(weights)

    def set_biases(self, biases):
        return self.fc_biases.assign(biases)

    def set_placeholder_weights(self, weights):
        if self.update_fc_weights_op is None:
            self.update_placeholder_weights = tf.placeholder(self.init_fc_weights.dtype,
                                                             shape=self.fc_weights.get_shape())
            self.update_fc_weights_op = self.fc_weights.assign(self.update_placeholder_weights)
        SessionHandler().get_session().run(self.update_fc_weights_op,
                                           {self.update_placeholder_weights:weights})
    def set_placeholder_biases(self, biases):
        if self.update_fc_biases_op is None:
            self.update_placeholder_biases = tf.placeholder(self.init_fc_biases.dtype,
                                                             shape=self.fc_biases.get_shape())
            self.update_fc_biases_op = self.fc_biases.assign(self.update_placeholder_biases)
        SessionHandler().get_session().run(self.update_fc_biases_op,
                                           {self.update_placeholder_biases:biases})


class ShortcutLayer():
    """
        Shortcut connection layer.
    """
    def __init__(self):
        pass

class CNN(object):
    """a Convolution neural network model
        Args:
        # IMAGE_SIZE
        # NUM_CHANNELS
        # NUM_LABELS
        # BATCH_SIZE=None
        # a parameters above are used for defining the image input node
        # in the first time, i want to make a model like keras, however,
        # i found it a lit of bit difficult. so, i just make a cnn model instead.
        # in the near future, i would like to make a residual cnn
    """
    def __init__(self, IMAGE_SIZE, NUM_CHANNELS, NUM_LABELS, BATCH_SIZE=None):
        self.image_size = IMAGE_SIZE
        self.label_size = NUM_LABELS
        self.batch_size = BATCH_SIZE

        self.last_layer_size = IMAGE_SIZE
        self.last_core_size = NUM_CHANNELS

        self.num_layer = 0
        self.layer_map = {}

        self.weight_list = []
        self.bias_list = []

        self.input_node = tf.placeholder(tf.float32,
                                    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        self.label_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, NUM_LABELS))

        # initial learning rate
        self.lr_value = 0.01
        self.placeholder_mode = False

        # self.fc1_weights = None
        # self.fc1_biases = None
        # self.fc2_weights = None
        # self.fc2_biases = None

        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.gobal_step = tf.Variable(0, trainable=False)
        self.decay_size = tf.placeholder(dtype=tf.int32)
        self.momentum = tf.Variable(0.8)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

    ## add convpool layer
    def add(self, layer):
        self.layer_map[self.num_layer] = layer
        # self.compute_last_layer_size(layer)
        self.num_layer += 1

        # if layer.layer_type == LAYER_TYPE[1] or layer.layer_type == LAYER_TYPE[0]:
        #     self.weight_list.append(layer.conv_weights)
        # elif layer.layer_type == LAYER_TYPE[3]:
        #     self.weight_list.append(layer.fc_weights)
        #     self.bias_list.append(layer.fc_biases)


    ## actually init session
    def make_net(self, mode="softmax"):
        if IS_GPU:
            self.create_net(mode)
        else:
            with tf.device("/cpu:0"):
                self.create_net(mode)

    def load_init_weights(self, hdf5_dset):
        weight_num = 0
        bias_num = 0
        for i in range(0,self.num_layer):
            if self.layer_map[i].layer_type != LAYER_TYPE[2] and self.layer_map[i].layer_type != LAYER_TYPE[3]:
                self.layer_map[i].set_init_values(hdf5_dset['weight_%s'%(weight_num)][:])
                weight_num += 1
            elif self.layer_map[i].layer_type == LAYER_TYPE[3]:
                self.layer_map[i].set_init_values(hdf5_dset['weight_%s'%(weight_num)][:],
                                                  hdf5_dset['bias_%s'%(bias_num)][:])
                weight_num += 1
                bias_num += 1

    def load_weights(self, hdf5_dset):
        weight_num = 0
        bias_num = 0
        for i in range(0,self.num_layer):
            if self.layer_map[i].layer_type != LAYER_TYPE[2] and self.layer_map[i].layer_type != LAYER_TYPE[3]:
                self.layer_map[i].set_placeholder_weights(hdf5_dset['weight_%s'%(weight_num)][:])
                weight_num += 1
            elif self.layer_map[i].layer_type == LAYER_TYPE[3]:
                self.layer_map[i].set_placeholder_weights(hdf5_dset['weight_%s'%(weight_num)][:])
                self.layer_map[i].set_placeholder_biases(hdf5_dset['bias_%s'%(bias_num)][:])
                weight_num += 1
                bias_num += 1

    def set_learning_rate(self, lr):
        self.lr_value = lr

    # initialize every layer
    def compile_net(self):
        for i in range(self.num_layer):
            self.layer_map[i].init_net()

    # in order to use GPU or CPU selectively
    def create_net(self, mode="Softmax"):
            self.compile_net()

            last_layer_type = ""
            for i in range(self.num_layer):
                if i == 0 and self.layer_map[i].layer_type != LAYER_TYPE[2]:
                    next_input = tf.nn.relu(self.layer_map[i].get_output(self.input_node))
                    last_layer_type = self.layer_map[i].layer_type
                elif i == (self.num_layer-1):
                    next_input = self.layer_map[i].get_output(next_input)
                elif i < (self.num_layer-1) and (last_layer_type == LAYER_TYPE[1] or last_layer_type == LAYER_TYPE[0]
                                                 or last_layer_type == LAYER_TYPE[2]) \
                    and self.layer_map[i].layer_type == LAYER_TYPE[3]:
                    shape = next_input.get_shape().as_list()
                    reshape = tf.reshape(
                        next_input,
                        [-1, shape[1] * shape[2] * shape[3]])
                    next_input = tf.nn.relu(self.layer_map[i].get_output(reshape))
                elif self.layer_map[i].layer_type == LAYER_TYPE[2]:
                    next_input = self.layer_map[i].get_output(next_input)
                else:
                    next_input = tf.nn.relu(self.layer_map[i].get_output(next_input))
                last_layer_type = self.layer_map[i].layer_type

            self.logits = next_input

            if mode == "softmax":
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                self.logits, self.label_node))
                self.pred = tf.nn.softmax(self.logits)
            elif mode == "regression":
                self.loss = tf.reduce_mean((self.label_node - self.logits)**2)
                self.pred = self.logits

            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum
                                        ).minimize(self.loss,
                                                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                                                        global_step=None)



    def fit(self, inputX, inputY, epoch, batch_size, lr=0.01, decay_size=20, verbose=False):
        lost = 0
        for i in range(epoch):
            feed_dict = {self.input_node: inputX,
                         self.label_node: inputY,
                         self.batch_size: batch_size,
                         self.decay_size: decay_size,
                         self.learning_rate: lr}

            [ _, l ] = SessionHandler().get_session().run([self.optimizer, self.loss],
                                            feed_dict=feed_dict)
            lost += l
            if verbose:
                print("loss is "+str(l))
        return lost


    def get_predict(self, input):
        return self.pred.eval(session=SessionHandler().get_session(),
                              feed_dict = {self.input_node:input})


    def save_weights(self, file_name):
        with h5py.File(file_name, 'w') as f:
            weight_num = 0
            bias_num = 0
            for i in range(0,self.num_layer):
                if self.layer_map[i].layer_type != LAYER_TYPE[2] and self.layer_map[i].layer_type != LAYER_TYPE[3]:
                    weight =  self.layer_map[i].conv_weights.eval()
                    name = "weight_%s" % str(weight_num)
                    f.create_dataset(name, data=weight)
                    weight_num += 1

                elif self.layer_map[i].layer_type == LAYER_TYPE[3]:
                    weight = self.layer_map[i].fc_weights.eval()
                    bias = self.layer_map[i].fc_biases.eval()
                    f.create_dataset("weight_%s" % str(weight_num), data=weight)
                    f.create_dataset("bias_%s"%str(bias_num), data=bias)
                    weight_num += 1
                    bias_num += 1

    def release(self):
        self.session.close()

    def init_vars(self):
        SessionHandler().initialize_variables()

#
def get_predict_by_batch(model, test_x, batch_size):
    iter = math.ceil(test_x.shape[0] / batch_size)
    pred_list = []

    for i in range(iter-1):
        batch_ind = i * batch_size
        batch_ind_plus1 = (i+1) * batch_size
        batch_x = test_x[batch_ind:batch_ind_plus1,:,:,:]
        batch_pred = model.get_predict(batch_x)
        pred_list.append(batch_pred)
    batch_pred = model.get_predict(test_x[batch_ind_plus1:,:,:,:])
    pred_list.append(batch_pred)

    pred = np.concatenate(pred_list)
    return pred

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])


def get_cnn_model():
    """
    get the policy model.   i would like to make it like keras, just need input json.
        this function is totally handmake, it was so ugly
    Returns:

    """
    try:
        if not SessionHandler().is_session_init:
            SessionHandler().set_default()
    except:
        SessionHandler().set_default()

    state_size = 19
    cnn = CNN(state_size, 3 ,state_size**2)

    next_core_size = 4
    cnn.add(ConvLayer([2,2,3,next_core_size],"VALID"))
    for i in range(5):
        cnn.add(ConvLayer([3,3,next_core_size,next_core_size*2],"SAME"))
        next_core_size *=2
    cnn.add(ConvLayer([1,1,next_core_size,next_core_size],"SAME"))
    cnn.add(FullForwardLayer(128*18*18,512))
    cnn.add(FullForwardLayer(512,361))

    cnn.make_net()
    cnn.init_vars()
    return cnn

# like above
def get_cnn_value_model(weight_dir=None):
    try:
        if not SessionHandler().is_session_init:
            SessionHandler().set_default()
    except:
        SessionHandler().set_default()

    weights_dict = {}

    if weight_dir is None:
        for i in range(8):
            weights_dict["weight_%s"%i] = None
        for i in range(2):
            weights_dict["bias_%s"%i] = None
    else:
        dset = h5py.File(weight_dir, 'r')
        for name in dset:
            weights_dict[name] = dset[name][:]
        dset.close()

    state_size = 19
    cnn = CNN(state_size, 3 ,1)

    next_core_size = 4
    cnn.add(ConvLayer([2,2,3,next_core_size],"VALID", weights=weights_dict['weight_0']))
    for i in range(5):
        cnn.add(ConvLayer([3,3,next_core_size,next_core_size*2],"SAME", weights=weights_dict['weight_%s'%(i+1)]))
        next_core_size *=2
    cnn.add(ConvLayer([1,1,next_core_size,next_core_size],"SAME" ,weights=weights_dict['weight_%s'%(i+2)]))
    cnn.add(FullForwardLayer(128*18*18,512, weights=weights_dict['weight_%s'%(i+3)], biases=weights_dict['bias_0']))
    cnn.add(FullForwardLayer(512,1))

    cnn.make_net("regression")
    cnn.init_vars()
    return cnn

# for testing the class made by tensorflow
if __name__ == '__main__':
    SessionHandler().set_default()

    digit = np.loadtxt("../static/digitInputOutput.txt",dtype=np.float32)
    np.random.shuffle(digit)

    train_size = 4000
    X = digit[:train_size,0:400].reshape((train_size,20,20,1))
    Y = digit[:train_size,400:]

    test_X = digit[train_size:,0:400].reshape((5000-train_size,20,20,1))
    test_Y = digit[train_size:,400:]

    NUM_CHANNELS = 1
    BATCH_SIZE = 200

    dset = h5py.File('../weights/test_w.hdf5','r')
    weights_0 = dset['weight_0'][:]


    cnn = CNN(20,NUM_CHANNELS,10)
    next_cores_size = 8
    cnn.add(ConvLayer(filter_shape=[3,3,NUM_CHANNELS,8]))
    cnn.add(ConvLayer(filter_shape=[3,3,8,16]))
    cnn.add(ConvLayer(filter_shape=[3,3,16,32]))
    cnn.add(PoolLayer())
    cnn.add(FullForwardLayer(32*10*10,512))
    cnn.add(FullForwardLayer(512,10))

    cnn.make_net()
    cnn.init_vars()
    # fc1_pre = cnn.fc1_weights.eval()


    for epoch in range(1):
            print("epoch %d" % epoch)
            for i in range(20):
                tmpX = X[i*200:(i+1)*200,:,:,:].reshape((200,20,20,1))
                tmpY = Y[i*200:(i+1)*200,:]

                cnn.fit(tmpX,tmpY,1,BATCH_SIZE, decay_size=20)
            pred = get_predict_by_batch(cnn, test_X, BATCH_SIZE)
            er = error_rate(pred, test_Y)
            print("test error: %s" % str(er))


    cnn.load_weights(dset)
    # fc1_pre = cnn.fc1_weights.eval()
    tf.get_default_graph().finalize()

    for epoch in range(10):
            print("epoch %d" % epoch)
            for i in range(20):
                tmpX = X[i*200:(i+1)*200,:,:,:].reshape((200,20,20,1))
                tmpY = Y[i*200:(i+1)*200,:]

                cnn.fit(tmpX,tmpY,1,BATCH_SIZE, decay_size=20, lr=0.005)
                # print(tf.get_default_graph().)
            pred = get_predict_by_batch(cnn, test_X, BATCH_SIZE)
            er = error_rate(pred, test_Y)
            print("test error: %s" % str(er))

