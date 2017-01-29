import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwonLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
