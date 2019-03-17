import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from collections import OrderedDict

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        #initilization of variables
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        self.hidden_outputs = []

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(1-dp_keep_prob)
        self.sequenses = nn.ModuleList()

        # Input layer
        self.sequence_layers = nn.ModuleList()
        self.sequence_layers.append(nn.ModuleList([
            nn.Linear(emb_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(1-dp_keep_prob)]
        ))

        # Hidden layer(s)
        self.sequence_layers.extend(clones(nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(1-dp_keep_prob)]
        ), num_layers-1))
        
        # Output layer
        self.sequence_ouput = nn.ModuleList([
            nn.Linear(hidden_size,self.vocab_size)
            ]
        )

        # Creates hidden layers
        #self.hiddens = [Variable(torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32), requires_grad=True) for _ in range(self.seq_len)]

        self.init_weights_uniform()
        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers. 
        # Your implementation should support any number of stacked hidden layers 
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules 
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
        # provided clones function (as opposed to a regular python list), in order 
        # for Pytorch to recognize these parameters as belonging to this nn.Module 
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
   
    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        # in the range [-k, k] where k is the square root of 1/hidden_size

        torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for i in range(self.num_layers):
            self.initialize_sequence_weights(math.sqrt(1/self.hidden_size), i)

        torch.nn.init.uniform_(self.sequence_ouput[0].weight, -0.1, 0.1)
        torch.nn.init.constant_(self.sequence_ouput[0].bias, 0)

    def initialize_sequence_weights(self, b, layer):
        for i in range(2):
            torch.nn.init.uniform_(self.sequence_layers[layer][i].weight, -b, b)
            if self.sequence_layers[layer][i].bias is not None:
                torch.nn.init.uniform_(self.sequence_layers[layer][i].bias, -b, b)

    def init_hidden(self):
        # TODO =======================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32)
   
    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the 
        # inner for loop should iterate over hidden layers of the stack. 
        # 
        # Within these for loops, use the parameter tensors and/or nn.modules you 
        # created in __init__ to compute the recurrent updates according to the 
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).
   
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that 
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        
        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the 
                  mini-batches in an epoch, except for the first, where the return 
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details, 
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        #self.hidden_outputs.append(hidden)

        logits = Variable(torch.zeros([self.seq_len, self.batch_size, self.vocab_size]), requires_grad=True)
        logits = logits.to(torch.device("cuda"))

        #prev_hidden = None
        prev_hidden = hidden
        for i in range(self.seq_len):

            # Initialize hidden layer for next iteration
            hidden_out = self.init_hidden().to(torch.device('cuda'))
            data = self.embedding(inputs[i])
            data = self.dropout(data)
            for j in range(self.num_layers):
                #data = self.sequence_layers[j][0](data) + self.sequence_layers[j][1](self.hidden_outputs[i][j])
                data = self.sequence_layers[j][0](data) + self.sequence_layers[j][1](prev_hidden[j])
                data = self.sequence_layers[j][2](data)
                hidden_out[j] = data
                data = self.sequence_layers[j][3](data)
            #print(data)
            #print('-----------------------------------------------------')

            data = self.sequence_ouput[0](data)
            logits[i] = data

            # Pushes temp hidden out to list of hidden outputs
            #self.hidden_outputs.append(Variable(hidden_out, requires_grad=True))
            #prev_hidden = Variable(hidden_out, requires_grad=True)
            prev_hidden = hidden_out


        #return logits.view(self.seq_len, self.batch_size, self.vocab_size), self.hidden_outputs[-1]
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), prev_hidden
   
    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        # 
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution, 
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation 
        # function here in order to compute the parameters of the categorical 
        # distributions to be sampled from at each time-step.
   
        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
       
        return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for 
    GRU, not Vanilla RNN.

    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        self.hidden_outputs = []

        m = nn.Sigmoid()
        output_size=10
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(1-dp_keep_prob)
        self.sequenses = nn.ModuleList()

        # Input layer
        self.sequence_layers = nn.ModuleList()
        self.sequence_layers.append(nn.ModuleList([
            nn.Linear(emb_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(emb_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(emb_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(1-dp_keep_prob),
            nn.Sigmoid(),
            nn.Sigmoid(),
            nn.Tanh(),
            ]
        ))

        # Hidden layer(s)
        self.sequence_layers.extend(clones(nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(1-dp_keep_prob),
                nn.Sigmoid(),
                nn.Sigmoid(),
                nn.Tanh(),]
        ), num_layers-1))
        
        # Output layer
        self.sequence_ouput = nn.ModuleList([
            nn.Linear(hidden_size,self.vocab_size)
            ]
        )

        # Creates hidden layers
        #self.hiddens = [Variable(torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32), requires_grad=True) for _ in range(self.seq_len)]

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        # in the range [-k, k] where k is the square root of 1/hidden_size

        torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for i in range(self.num_layers):
            self.initialize_sequence_weights(math.sqrt(1/self.hidden_size), i)

        torch.nn.init.uniform_(self.sequence_ouput[0].weight, -0.1, 0.1)
        torch.nn.init.constant_(self.sequence_ouput[0].bias, 0)

    def initialize_sequence_weights(self, b, layer):
        for i in range(6):
            torch.nn.init.uniform_(self.sequence_layers[layer][i].weight, -b, b)
            if self.sequence_layers[layer][i].bias is not None:
                torch.nn.init.uniform_(self.sequence_layers[layer][i].bias, -b, b)

    def init_hidden(self):
        # TODO ========================
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size], dtype=torch.float32)

    def forward(self, inputs, hidden):
        logits = Variable(torch.zeros([self.seq_len, self.batch_size, self.vocab_size]), requires_grad=True)
        logits = logits.to(torch.device("cuda"))

        #prev_hidden = None
        prev_hidden = hidden
        #sigmoid = nn.Sigmoid()
        #tanh = nn.Tanh()
        inputs = self.embedding(inputs)

        for i in range(self.seq_len):

            data = self.dropout(inputs[i])

            # Initialize hidden layer for next iteration
            hidden_out = self.init_hidden().to(torch.device('cuda'))

            for j in range(self.num_layers):
                #data = self.sequence_layers[j][0](data) + self.sequence_layers[j][1](self.hidden_outputs[i][j])
                reset = self.sequence_layers[j][7](self.sequence_layers[j][0](data) + self.sequence_layers[j][1](prev_hidden[j]))
                forget =self.sequence_layers[j][8](self.sequence_layers[j][2](data) + self.sequence_layers[j][3](prev_hidden[j]))
                tild_hidden = self.sequence_layers[j][9](self.sequence_layers[j][4](data) + self.sequence_layers[j][5](torch.mul(reset,prev_hidden[j])))
                data = torch.mul((1 - forget),prev_hidden[j]) + torch.mul(forget,tild_hidden)
                hidden_out[j] = data
                data = self.sequence_layers[j][6](data)
            #print(data)
            #print('-----------------------------------------------------')

            data = self.sequence_ouput[0](data)
            logits[i] = data

            # Pushes temp hidden out to list of hidden outputs
            #self.hidden_outputs.append(Variable(hidden_out, requires_grad=True))
            #prev_hidden = Variable(hidden_out, requires_grad=True)
            prev_hidden = hidden_out
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), prev_hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units
        self.n_heads = n_heads 

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        self.sequence_layers = nn.ModuleList()
        self.sequence_layers.extend(clones(nn.ModuleList([
                nn.Linear(n_units, n_units),
                nn.Linear(n_units, n_units),
                nn.Linear(n_units, n_units),
                nn.Dropout(dropout)]
        ), n_heads))

        self.output = nn.Linear(n_heads*n_units, n_units)

        self.initialize_parameters_uniform()
    
    def initialize_parameters_uniform(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -math.sqrt(1/self.n_units), math.sqrt(1/self.n_units))

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        attention_i = []
        #mask = mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
        for head in range(self.n_heads):
            query_i = self.sequence_layers[head][0](query)
            key_i = self.sequence_layers[head][1](key)
            value_i = self.sequence_layers[head][2](value)
            x = torch.matmul(query_i, torch.transpose(key_i, 1,2))/math.sqrt(self.d_k)
            #x_tild = torch.mul(x,mask) - ((10**9)*(1-mask))
            x_tild = x
            x_tild[~mask] = -(10)**9
            #x_tild[mask] = x[mask]
            #x_tild[~mask] = -(10)**9
            score_i = F.softmax(x_tild, dim=0)
            score_i = self.sequence_layers[head][3](score_i)
            h_i = torch.matmul(score_i, value_i)
            attention_i.append(h_i)
        data = torch.cat(attention_i, dim=2)
        attention = self.output(data)

        return attention






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

