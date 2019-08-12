#%%
import utils

import utils
import tensorflow as tf
import numpy as np

from smith_waterman_layer import SmithWatermanLayer

from tensorflow import keras

import gzip

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

genome = utils.get_fasta("hg38.fa.gz")
bed_file = "ENCFF348ZZP.bed.gz"

def get_random_range(start, end):
    if (end - start) <= context_length: 
        midpoint = int(.5 * (start + end))
        return( midpoint - context_length/2, midpoint + context_length/2 )
    else: 
        left = np.random.randint(start, end - context_length)
        return( left, left + context_length )

def read_bed():
    prev_end = 0
    prev_chrom = ""
    with gzip.open(bed_file,"r") as f: 
        for l in f: 
            ss = l.decode().strip().split()
            (chrom, start, end) = ss[0:3]
            start = int(start)
            end = int(end)
            l,r = get_random_range(start, end)
            seq = utils.fetch_sequence(genome, chrom, l, r, "+")
            yield(seq, 1) # positive example (randomly within peak)

            if prev_chrom == chrom: 
                l,r = get_random_range(prev_end, start)
                seq = utils.fetch_sequence(genome, chrom, l, r, "+")
                yield(seq, 0) # negative example randomly inbetween peaks
            
            prev_chrom = chrom
            prev_end = end

def get_xy(batch_size=100):
    sequences = []
    labels = []
    for (seq,label) in read_bed(): 
        if seq is None: continue
        sequences.append( utils.one_hot(seq) )
        labels.append(label)
        if len(labels) >= batch_size: 
            yield(np.array(sequences), np.array(labels))
            sequences=[]
            labels=[]

#%%
def train(model, optimizer, batch_size=100, epochs=20, print_every=10):
    
    for epoch_index in range(epochs):
        avg_loss = keras.metrics.Mean(name='loss', dtype=tf.float32)
        avg_acc = keras.metrics.Mean(name='acc', dtype=tf.float32)
        for (x,y) in get_xy(batch_size=batch_size): 
            with tf.GradientTape() as tape: 
                pred = tf.squeeze( model(x) )
                loss = keras.losses.binary_crossentropy(y, pred, from_logits=True)
                avg_loss.update_state(loss)
                acc = keras.metrics.binary_accuracy(y, pred, threshold=0.)
                avg_acc.update_state(acc)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if tf.equal(optimizer.iterations % print_every, 0):
                print("\r%i %f %f" % (optimizer.iterations, avg_loss.result(), avg_acc.result()), end='', flush=True)
        print("\rEPOCH %i %f %f" % (epoch_index, avg_loss.result(), avg_acc.result()))
        avg_loss.reset_states()
        avg_acc.reset_states()

#%% Standard CNN. Note SOTA uses more like context=1k bp
context_length = 200

model = keras.Sequential([keras.layers.Conv1D(32, 7, activation="elu", input_shape=(context_length, 4)), 
                    keras.layers.MaxPool1D(4),
                    keras.layers.Conv1D(32, 5, activation="elu"), 
                    keras.layers.MaxPool1D(3),
                    keras.layers.Conv1D(32, 3, activation="elu"),
                    keras.layers.MaxPool1D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, "elu"),
                    keras.layers.Dense(1)])

train(model, optimizer = keras.optimizers.Adam())

#%% With SmithWatermanLayer
model = keras.Sequential([SmithWatermanLayer(32, 7), 
                    keras.layers.MaxPool1D(4),
                    keras.layers.Conv1D(32, 5, activation="elu"), 
                    keras.layers.MaxPool1D(3),
                    keras.layers.Conv1D(32, 3, activation="elu"),
                    keras.layers.MaxPool1D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, "elu"),
                    keras.layers.Dense(1)])

train(model, optimizer = keras.optimizers.Adam(), batch_size=512, print_every=1)
