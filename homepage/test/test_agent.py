from homepage.src.AgentHandler import AgentHandler
from homepage.src.nn_utils import FullForwardLayer, CNN
from homepage.src.tf_utils import SessionHandler
from homepage.src.goAgent import TestGo
import numpy as np

def main():
    try:
        if not SessionHandler().is_session_init:
            SessionHandler().set_default()
    except:
        SessionHandler().set_default()

    model = CNN(None, None,None)
    model.add(FullForwardLayer(784,1))
    model.make_net("regression")
    model.init_vars()

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))

    model.fit(data,labels,100,100,verbose=True)

if __name__ == '__main__':
    main()