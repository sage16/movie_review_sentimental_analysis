import numpy as np
import streamlit as st
from PIL import Image
from collections import Counter
from string import punctuation
import torch
import torch.nn as nn

# load saved model
with open('C:/da/Pytorch Neural Network/rnn_04_epoch.pth', 'rb') as f:
    checkpoint = torch.load(f)

vocab_to_int = checkpoint['vocab_to_int_']


def tokenize_review(test_review):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words])

    return test_ints


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


# Model Architecture
class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


def predict(net, test_review, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    list_review = ["Positive review detected!", "Negative review detected."]

    return list_review[0] if (pred.item() == 1) else list_review[1]

    # print custom response
    # if (pred.item() == 1):
    # print("Positive review detected!")
    # else:
    # print("Negative review detected.")


# load model architecture
loaded = SentimentRNN(checkpoint['vocab_size'], checkpoint['output_size'], checkpoint['embedding_dim'],
                      hidden_dim=checkpoint['hidden_dim'], n_layers=checkpoint['n_layers'])

loaded.load_state_dict(checkpoint['state_dict'])

# streamlit webapp creation
st.title('Sentimental Analysis App')
st.write('The data for the following example was gotten Leo Tolstoy\'s novel Anna Karenina')
image = Image.open('C:/da/Streamlit Folder/sentimental.jpg')
st.image(image, use_column_width=True)
st.write(
    'Please fill in the details of the person under consideration in the left siderbar and click on the button below')

sentiment = st.text_input('Enter Your Start Word', 'Type Here')

button = st.button('Submit')
output =  predict(loaded, sentiment)
if button:
   output

st.write()
