import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model

# read files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#check train set for nulls
train.isnull().any()

# define toxicity categories
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# isolate values of classes for train set
y = train[list_classes].values

# isolate comments for train and test set
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

# define the number of unique words in our dataset
MAXFEATURES = 20000

# Initialize Tokenizer: splitting the comments into (unique) words
tokenizer = Tokenizer(num_words=MAXFEATURES)

# fit Tokenizer in our comments
tokenizer.fit_on_texts(list(list_sentences_train))

# map words into their unique id for train and test set
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# padding comments into sentences in order to get fixed length comments into batch
CommentLengths = [len(comment) for comment in list_tokenized_train]
MAXLENGTH = int(pd.DataFrame(CommentLengths ).quantile(0.90)[0]) ### gets the value 148

# create batches of length MAXLENGTH
XTR = pad_sequences(list_tokenized_train, maxlen=MAXLENGTH)
XTS = pad_sequences(list_tokenized_test, maxlen=MAXLENGTH)

# automatically set the number of input
inp = Input(shape=(MAXLENGTH, ))

# define the vector space of the embendings
EMBEDSIZE = 128
x = Embedding(MAXFEATURES, EMBEDSIZE)(inp)

# feed Tensor into the LSTM layer
# LSTM takes in a tensor of [Batch Size, Time Steps, Number of Inputs]
BATCHSIZE = 60 # number of samples in a batch
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

# reshape the 3D tensor into 2D
# use Global Max Pooling to reduce the dimensionality by taking the maximum values of each batch
x = GlobalMaxPool1D()(x)

# set DropOut rate
DROPOUT = 0.15
x = Dropout(DROPOUT)(x)

# Dense layer to produce an output dimension of 50.
DENSE = 60
x = Dense(DENSE, activation="relu")(x)

x = Dropout(0.1)(x)

# use sigmoid for binary classification of six labels
x = Dense(6, activation="sigmoid")(x)


# define the model
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

VALIDATIONSPLIT = 0.2

EPOCHS = 2
model.fit(XTR,y, epochs=EPOCHS, batch_size=32, validation_split=VALIDATIONSPLIT )

y_pred = model.predict(XTS, batch_size=1024)

outputcsv = pd.DataFrame.from_dict({'id': test['id']})
for idx, col in enumerate(list_classes):
    outputcsv[col] = y_pred[:,idx]
outputcsv.to_csv('output.csv', index=False)