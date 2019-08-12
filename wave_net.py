# Offset indexing by one for causal conv
# Calculate receptive field and slice rather than using padding="same"
# Don't replicate sequence net component
# ResNet blocks
# Skip connections

#%% 

import utils
import tensorflow as tf
import numpy as np

from tensorflow import keras

import utils

import gzip

#%%

def res_block(original_input, channels_out, kernel_size, padding="valid", dilation_rate = 2):

    net = keras.layers.Conv1D(channels_out, 
                              kernel_size, 
                              activation="elu", 
                              padding=padding, 
                              dilation_rate=dilation_rate)(original_input)
    
    net = keras.layers.Conv1D(channels_out, 
                              kernel_size, 
                              activation="elu", 
                              padding=padding, 
                              dilation_rate=dilation_rate)(net)

    skip = original_input
    if padding == "valid":
        to_trim = (kernel_size - 1) * dilation_rate
        assert(to_trim % 2 == 0)
        to_trim /= 2
        #skip = original_input[:,to_trim:-to_trim,:]
        skip = keras.layers.Cropping1D(to_trim)(skip)
    
    if channels_out != original_input.shape[-1]:
        skip = keras.layers.Conv1D(channels_out, 1, activation="elu")(skip)

    print(net)
    print(skip)
    return(keras.layers.Add()((net, skip)))

def my_model(causal_net, noncausal_net, num_layers = 7, num_channels = 32, kernel_size = 7, dilation_rate = 2):
    
    #causal_net = tf.pad( causal_net[:,:-1,:], tf.constant([(0, 0,), (1, 0), (0, 0)] ) )
    causal_net = keras.layers.Cropping1D((0,1))(causal_net)
    causal_net = keras.layers.ZeroPadding1D((1,0))(causal_net)

    for i in range(num_layers):
        print(i)
        concat = keras.layers.concatenate([causal_net, noncausal_net])
        causal_net = res_block(concat, 
                            num_channels, 
                            kernel_size, 
                            padding="causal", 
                            dilation_rate = dilation_rate)
        noncausal_net = res_block(noncausal_net, 
                            num_channels, 
                            kernel_size, 
                            padding="same", 
                            dilation_rate = dilation_rate)

    concat = keras.layers.concatenate([causal_net, noncausal_net])
    output = keras.layers.Conv1D(1, 
                                 kernel_size, 
                                 activation=None, 
                                 padding="causal", 
                                 dilation_rate=dilation_rate)
    return(output)

causal_net = keras.layers.Input((None, 1))
noncausal_net = keras.layers.Input((None, 4))

output = my_model(causal_net, noncausal_net)

model = keras.Model(inputs=[causal_net,noncausal_net], outputs = output)

context_length = 10
a = np.random.rand(3, context_length, 4).astype(np.float32)
b = np.random.rand(3, context_length, 4).astype(np.float32)

model([a,b])

#%%
optimizer = keras.optimizers.Adam()

print_every=1

avg_loss = keras.metrics.Mean(name='loss', dtype=tf.float32)
avg_acc = keras.metrics.Mean(name='acc', dtype=tf.float32)
for ((is_exon, one_hot), is_exon) in get_gene(): 
    with tf.GradientTape() as tape: 
        #pad = dilation_rate - (is_exon.shape[1] % dilation_rate)
        #if pad != 0:
        #    is_exon = np.pad( is_exon, ((0,0),(0,pad),(0,0)), "constant")
        #    one_hot = np.pad( one_hot, ((0,0),(0,pad),(0,0)), "constant")
        pred = model(is_exon, one_hot) 
        loss = keras.losses.binary_crossentropy(is_exon, pred, from_logits=True)
        avg_loss.update_state(loss)
        acc = keras.metrics.binary_accuracy(is_exon, pred, threshold=0.)
        avg_acc.update_state(acc)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if tf.equal(optimizer.iterations % print_every, 0):
        print("\r%i %f %f" % (optimizer.iterations, avg_loss.result(), avg_acc.result()), end='', flush=True)
avg_loss.reset_states()
avg_acc.reset_states()

