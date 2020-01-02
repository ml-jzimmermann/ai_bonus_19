import pandas as pd
import numpy as np


input = pd.read_csv('../data/complete.csv')
texts = input['sentence'].values
labels = input['label'].values

# preprocessing text
stop_list = 'the to and i a of that is in you for it have he my with was are on but be this so not'.split()

# tokenization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_words = 10000
max_length = 75
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre', value=0.0)
sequences = np.array(sequences)
print(len(tokenizer.word_index))
print(sequences.shape)

# preprocessing labels
from keras.utils import to_categorical
labels = to_categorical(labels)
labels = np.array(labels)
print(labels.shape)
classes = labels.shape[1]
print('classes: ' + str(classes))

# split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.15, shuffle=True, random_state=42)
print(x_train.shape)
print(y_train.shape)

# generate balanced weights for training
from sklearn.utils import class_weight
def generate_balanced_weights(y_train):
    y_labels = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
    weight_dict = {}
    for key in range(len(class_weights)):
        weight_dict[key] = class_weights[key]
    return weight_dict

class_weight_dict = generate_balanced_weights(y_train)
print(class_weight_dict)

# model
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Flatten
from keras.models import Model

features = 200
encode_input = Input(shape=(max_length,))
embed_1 = Embedding(input_dim=(max_words - 1), output_dim=features, input_length=max_length)(encode_input)
lstm_encode_1 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(embed_1)
lstm_encode_2 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(lstm_encode_1)
lstm_encode_3 = Bidirectional(LSTM(units=16, activation='tanh', dropout=0.2, return_sequences=True))(lstm_encode_2)

lstm_decode_1 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=False))(lstm_encode_3)
aux_output = Dense(units=max_length, activation='relu', name='aux_output')(lstm_decode_1)

flat = Flatten()(lstm_encode_3)
dense = Dense(units=32, activation='relu')(flat)
output = Dense(units=classes, activation='softmax', name='class_output')(dense)

model = Model(inputs=encode_input, outputs=[aux_output, output])
lossWeights = {"aux_output": 2.0, "class_output": 1.0}
loss={'aux_output':'mae', 'class_output':'categorical_crossentropy'}
model.compile(loss=loss, loss_weights=lossWeights, optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

# show model
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='../images/model_plot_lstm_encoder.png', show_shapes=True, show_layer_names=True)

# training
model.fit(x=x_train, y=[x_train, y_train], validation_data=(x_val, [x_val, y_val]), batch_size=16, epochs=50)

