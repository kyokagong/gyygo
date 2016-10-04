

## train CNN and save the weights

from homepage.src.nn_utils import CNN, ConvLayer, PoolLayer, ConvPoolLayer, error_rate, get_predict_by_batch, get_cnn_value_model
from homepage.src.tf_utils import SessionHandler
import h5py
import math
import numpy as np
from homepage.src.RL_trainer import make_training_pairs4RL, train_rl_batch, train_rl_value_batch
from homepage.src.Player import Player
import tensorflow as tf

PLAYER_POOL_SIZE = 20

def train_cnn():
    state_size = 19
    cnn = CNN(state_size, 3 ,state_size**2)

    next_core_size = 4

    cnn.add(ConvLayer([2,2,3,next_core_size],"VALID"))
    for i in range(5):
        cnn.add(ConvLayer([3,3,next_core_size,next_core_size*2],"SAME"))
        next_core_size *=2
    cnn.add(ConvLayer([1,1,next_core_size,next_core_size],"SAME"))
    cnn.make_net()

    f = h5py.File("homepage/train_data/train_data1.hdf5","r")
    dat_x = f['dat_x'][:]
    dat_y = f['dat_y'][:]
    sample_size = dat_x.shape[0]

    index = np.arange(sample_size)
    np.random.shuffle(index)

    dat_x = dat_x[index,:,:,:]
    dat_y = dat_y[index,:]

    ran_ind = math.ceil(sample_size*0.9)
    train_x, test_x = dat_x[0:ran_ind,:,:,:], dat_x[ran_ind:,:,:,:]
    train_y, test_y = dat_y[0:ran_ind,:], dat_y[ran_ind:,:]


    batch_size = 128

    iter = math.ceil(train_x.shape[0] / batch_size)

    for epoch in range(10):
        lost = 0
        for i in range(iter-1):
            batch_ind = i * batch_size
            batch_ind_plus1 = (i+1) * batch_size
            batch_x = train_x[batch_ind:batch_ind_plus1,:,:,:]
            batch_y = train_y[batch_ind:batch_ind_plus1,:]

            l = cnn.fit(batch_x, batch_y, 1, batch_size, False)
            lost += l
        print(lost/(iter-1))
        pred = get_predict_by_batch(cnn, test_x, batch_size)
        er = error_rate(pred, test_y)
        print("epoch: %s, test error: %s" % (str(epoch), str(er)))
    # cnn.save_weights("homepage/weights/cnn_weights1.hdf5")


def test_cnn():
    state_size = 19
    cnn = CNN(state_size, 3 ,state_size**2)

    next_core_size = 4

    dset = h5py.File("homepage/weights/cnn_weights3.hdf5","r")

    cnn.add(ConvLayer([2,2,3,next_core_size],"VALID", weights=dset['weight_0'][:]))
    for i in range(5):
        cnn.add(ConvLayer([3,3,next_core_size,next_core_size*2],"SAME", weights=dset['weight_%s'%(i+1)][:]))
        next_core_size *=2
    cnn.add(ConvLayer([1,1,next_core_size,next_core_size],"SAME", weights=dset['weight_%s'%(6)][:]))
    cnn.make_net(fc1_weights=dset['weight_7'][:],fc1_biases=dset['bias_0'][:],fc2_weights=dset['weight_8'][:],fc2_biases=dset['bias_1'][:])


    f = h5py.File("homepage/train_data/train_data1.hdf5","r")
    dat_x = f['dat_x'][:]
    dat_y = f['dat_y'][:]

    batch_size = 128

    pred = get_predict_by_batch(cnn, dat_x, batch_size)
    er = error_rate(pred, dat_y)
    print(er)

def get_cnn_model():
    state_size = 19
    cnn = CNN(state_size, 3 ,state_size**2)

    next_core_size = 4
    cnn.add(ConvLayer([2,2,3,next_core_size],"VALID"))
    for i in range(5):
        cnn.add(ConvLayer([3,3,next_core_size,next_core_size*2],"SAME"))
        next_core_size *=2
    cnn.add(ConvLayer([1,1,next_core_size,next_core_size],"SAME"))
    return cnn

def rl_train(mode='policy'):
    player_weights_pool = []
    player = Player()
    opponent = Player()
    value_net = get_cnn_value_model("homepage/weights/rl_weights19.hdf5")

    default_weights_path = "homepage/weights/rl_weights_%s.hdf5" % "default"
    player_weights_pool.append(default_weights_path)

    player_weights = h5py.File(player_weights_pool[-1],'r')
    player.model.load_weights(player_weights)
    player_weights.close()
    feature_list = ['board']

    iter = 200
    lr = 0.03
    mini_batch_size = 1
    base = 10

    for i in range(iter):
        print("iter:%s, learning rate:%s" % (i,lr))

        weight_num = i % PLAYER_POOL_SIZE
        opponent_weights = np.random.choice(player_weights_pool)

        dset = h5py.File(opponent_weights,'r')
        # for name in dset:
        #     print(name)
        opponent.model.load_weights(dset)
        dset.close()

        X_list, y_list, winner_list = make_training_pairs4RL(player, opponent, feature_list, mini_batch_size,mode='value')
        if mode == 'policy':
            train_rl_batch(player, X_list, y_list, winner_list, lr)
            next_weights_path = "homepage/weights/rl_weights%s.hdf5" % (weight_num)
            player.model.save_weights(next_weights_path)

            player_weights_pool.append(next_weights_path)
            if len(player_weights_pool) > 20:
                player_weights_pool = player_weights_pool[1:]

        elif mode == 'value':
            train_rl_value_batch(value_net, X_list, y_list, lr, True)

        if i % base == 0:
            lr *= 0.9

    if mode == 'value':
        value_net.save_weights("homepage/weights/rl_value_weights0.hdf5")

    a = 1

if __name__ == '__main__':
    # train_cnn()
    # test_cnn()
    rl_train('value')