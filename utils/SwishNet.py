import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv1D, Multiply, Add, Concatenate, GlobalAveragePooling1D, Activation


# ====================================== SwishNet ======================================





def __causal_gated_conv1D(x = None, filters = 16, length = 6, strides = 1):
    def causal_gated_conv1D(x, filters, length, strides):
        x_sigm = Conv1D(filters = filters //2,
                       kernel_size = length,
                       dilation_rate = strides,
                       strides = 1,
                       padding = "causal",
                       activation = "sigmoid")(x)
        
        x_tanh = Conv1D(filters = filters //2,
                       kernel_size = length,
                       dilation_rate = strides,
                       strides = 1,
                       padding = "causal",
                       activation = "tanh")(x)
        
        x_out = Multiply()([x_sigm, x_tanh])
        
        return x_out
    
    if x is None:
        return lambda _x: causal_gated_conv1D(x=_x, filters=filters, length=length, strides=strides)
    else:
        return causal_gated_conv1D(x=x, filters=filters, length=length, strides=strides)
    
    
    
    
    
# ======================================================================================
    
    
    
    
    
def SwishNet(input_shape, classes, width_multiply=1):
    
    x_input = Input(shape = input_shape)
    
    # 1 block
    x_up = __causal_gated_conv1D(filters=16 * width_multiply, length=3)(x_input)
    x_down = __causal_gated_conv1D(filters=16 * width_multiply, length=6)(x_input)
    x = Concatenate()([x_up, x_down])
    
    # 2 block
    
    x_up = __causal_gated_conv1D(filters=8 * width_multiply, length=3)(x)
    x_down = __causal_gated_conv1D(filters=8 * width_multiply, length=6)(x)
    x = Concatenate()([x_up, x_down])
    
    # 3 block
    x_up = __causal_gated_conv1D(filters=8 * width_multiply, length=3)(x)
    x_down = __causal_gated_conv1D(filters=8 * width_multiply, length=6)(x)
    x_concat = Concatenate()([x_up, x_down])
    
    x = Add()([x, x_concat])
    
    # 4 block
    x_loop1 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=3)(x)
    x = Add()([x, x_loop1])

    # 5 block
    x_loop2 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(x)
    x = Add()([x, x_loop2])

    # 6 block
    x_loop3 = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(x)
    x = Add()([x, x_loop3])

    # 7 block
    x_forward = __causal_gated_conv1D(filters=16 * width_multiply, length=3, strides=2)(x)

    # 8 block
    x_loop4 = __causal_gated_conv1D(filters=32 * width_multiply, length=3, strides=2)(x)

    # output
    x = Concatenate()([x_loop2, x_loop3, x_forward, x_loop4])
    x = Conv1D(filters=classes, kernel_size=1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Activation("softmax")(x)

    model = Model(inputs=x_input, outputs=x)

    return model




# ======================================================================================