
class CausalConv1D(keras.layers.Conv1D):
    def __init__(self, 
                filters,
               kernel_size,
               **kwargs):
        assert(not "padding" in kwargs) 
        assert(not "data_format" in kwargs) # data_format must be channels_last

        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_last',
            **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0] + 1
        #inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (padding, 0), (0, 0)]))
        inputs = inputs[:,:-1,:]
        return super(CausalConv1D, self).call(inputs)

class ResBlock(keras.layers.Layer):
    
    def __init__(self, channels_in, channels_out, kernel_size, padding="valid", dilation_rate = 2):
        super(ResBlock, self).__init__()
        self.conv1 = keras.layers.Conv1D(channels_out, kernel_size, activation="elu", padding=padding, dilation_rate=dilation_rate)
        self.conv2 = keras.layers.Conv1D(channels_out, kernel_size, activation="elu", padding=padding, dilation_rate=dilation_rate)

        self.downsample = keras.layers.Conv1D(channels_out, 1, activation="elu") if channels_out != channels_in else None

        self.add = keras.layers.Add()

    def call(self, g):
        net = self.conv1(g)
        net = self.conv2(net)
        if self.conv1.padding == "valid":
            to_trim = (self.conv1.kernel_size[0] - 1) * self.conv1.dilation_rate[0] 
            assert(to_trim % 2 == 0)
            to_trim /= 2
            g = g[:,to_trim:-to_trim,:]
        skip = g if self.downsample is None else self.downsample(g)
        net = self.add((net, skip))
        return(net)

class MyModel(keras.Model):

    def __init__(self, num_layers = 7, num_channels = 32, kernel_size = 7, dilation_rate = 2):
        super(MyModel, self).__init__()
        self.concats = []
        self.causal_convs = []
        self.noncausal_convs = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.concats.append( keras.layers.Concatenate() )
            channels_in = 5 if i==0 else num_channels*2
            self.causal_convs.append( ResBlock(channels_in, num_channels, kernel_size, padding="causal", dilation_rate = dilation_rate) )
            #self.causal_convs.append( CausalConv1D(32, 7, activation="elu", dilation_rate=dilation_rate) )
            #self.noncausal_convs.append( keras.layers.Conv1D(32, 7, activation="elu", padding="same", dilation_rate=dilation_rate) )
            channels_in = 4 if i==0 else num_channels
            self.noncausal_convs.append( ResBlock(channels_in, num_channels, kernel_size, padding="same", dilation_rate = dilation_rate) )
            
        self.concats.append( keras.layers.Concatenate() )
        self.causal_convs.append( keras.layers.Conv1D(1, kernel_size, activation=None, padding="causal", dilation_rate=dilation_rate) )
        #self.causal_convs.append( CausalConv1D(1, 7, activation=None, dilation_rate=dilation_rate) )
        
    def call(self, causal_net, noncausal_net):
        
        causal_net = tf.pad( causal_net[:,:-1,:], tf.constant([(0, 0,), (1, 0), (0, 0)] ) )

        for i in range(self.num_layers):
            concat = self.concats[i]([causal_net, noncausal_net])
            causal_net = self.causal_convs[i](concat)
            noncausal_net = self.noncausal_convs[i](noncausal_net)

        concat = self.concats[self.num_layers]([causal_net, noncausal_net])
        return(self.causal_convs[self.num_layers](concat))

model = MyModel()



# TODO: why does logit[:,0,:] vary across transcripts :( 
# "causal" Conv1D conditions on :t not :(t-1) so need to offset indexing. 

#%%
one_hot_tf = tf.constant(one_hot, dtype=np.float32)
is_exon_new = np.zeros_like(is_exon)
for i in range(is_exon.shape[1]):
    print(i)
    logits = model.call( (tf.constant(is_exon_new, dtype=np.float32),one_hot_tf) )
    p = 1. / (1. + np.exp(-logits[:,i,:]))
    is_exon_new[:,i,:]=(np.random.rand(*p.shape) < p).astype(np.float32)

#%%
a = is_exon_new.squeeze()
a.mean(2)
a[:,0:-1] - a[:,1:]

#%%
x = keras.layers.Input((5, 1))
causal_net = keras.layers.Conv1D(1, 3, activation="elu", padding="causal")(x)
model = keras.Model(x, causal_net)
x_np = np.random.rand(1,5,1)
print(model.call(tf.constant(x_np,dtype=np.float32)))
x_np[0,2,0]=0
print(model.call(tf.constant(x_np,dtype=np.float32)))

#%%
i=0
for ((is_exon, one_hot), is_exon) in get_gene(): 
    break


#%%
y = model.call((tf.constant(is_exon,dtype=np.float32), tf.constant(one_hot[0,:,:][np.newaxis,:,:],dtype=np.float32)))

#%%
context_length = 10
a = np.random.rand(3, context_length, 4).astype(np.float32)
b = np.random.rand(3, context_length, 4).astype(np.float32)

model(a,b)

#%%


#%%
context_length = None # 200

is_exon_input = keras.layers.Input((context_length, 1))
causal_net = is_exon_input
seq_input = keras.layers.Input((context_length, 4)) #TODO: use batchsize=1 and then broadcast
noncausal_net = seq_input

for i in range(7):
    concat = keras.layers.Concatenate()([causal_net, noncausal_net])
    causal_net = keras.layers.Conv1D(32, 7, activation="elu", padding="causal", dilation_rate=2)(concat)
    noncausal_net = keras.layers.Conv1D(32, 7, activation="elu", padding="same", dilation_rate=2)(noncausal_net)

concat = keras.layers.Concatenate()([causal_net, noncausal_net])
causal_net = keras.layers.Conv1D(1, 7, activation=None, padding="causal", dilation_rate=2)(concat)    

model = keras.Model(inputs = (is_exon_input, seq_input), outputs = (causal_net,))

#%%
context_length = 10 # None # 200

input_a = keras.layers.Input((context_length, 4))
input_b = keras.layers.Input((context_length, 4))

net_a = keras.layers.Dense(5)(input_a)
net_b = keras.layers.Dense(5)(input_b)

output = keras.layers.add([net_a, net_b])

model = keras.Model(inputs = (input_a, input_b), outputs = output)

a = np.random.rand(3, 4).astype(np.float32)
b = np.random.rand(3, 4).astype(np.float32)

model([a,b])

#%%
input_a = keras.layers.Input((4,))
input_b = keras.layers.Input((4,))
output = keras.layers.Dense(5)(input_a)
output2 = keras.layers.Dense(5)(input_b)
model = keras.Model(inputs = [input_a,input_b], outputs = [output,output2])
a = np.random.rand(3, 4).astype(np.float32)
b = np.random.rand(3, 4).astype(np.float32)
model((a,b))
#%%


conv = CausalConv1D(3, 5, dilation_rate = 5)
conv(one_hot).shape

#%%

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

main_data = np.random.randint(10000,size=(10,100)).astype(np.int32)
aux_data = np.random.rand(10,5).astype(np.float32)

model.predict_on_batch( (main_data, aux_data))

model(main_data, aux_data)

#%%

