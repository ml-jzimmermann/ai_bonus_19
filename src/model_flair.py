import pandas as pd
import numpy as np


input = pd.read_csv('../data/complete.csv')
texts = input['sentence'].values
labels = input['label'].values

# preprocessing text
# stoplist

# process text
from keras.preprocessing.text import Tokenizer
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, Sentence, WordEmbeddings
embeddings_flair = StackedEmbeddings([FlairEmbeddings('mix-forward'), FlairEmbeddings('mix-backward')])
embeddings_glove = WordEmbeddings('glove')
max_words = 10000
max_length = 75
embedding_features = 4096

def embed_flair(texts, max_length=100, max_words=1000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    texts = tokenizer.sequences_to_texts(sequences)

    sentence_embeddings = []
    padding = np.zeros(embedding_features)
    count = 0
    step = 3
    max = len(texts)
    for text in texts:
        sentence_embedding = []
        paddings = []
        sentence = Sentence(text)
        embeddings_flair.embed(sentence)
        for token in sentence:
            sentence_embedding.append(token.embedding.cpu().numpy())
        for i in range(max_length - len(sentence_embedding)):
            paddings.append(padding)
        if len(paddings) > 0:
            sentence_embedding = np.concatenate([paddings, sentence_embedding], axis=0)
        else:
            sentence_embedding = np.array(sentence_embedding[:max_length])
        count += 1
        if (100 * count / max > step):
            print(str(step) + '%')
            step += 3
        sentence_embeddings.append(sentence_embedding)

    return np.array(sentence_embeddings)

sequences = embed_flair(texts, max_words=max_words, max_length=max_length)

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
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model

input_1 = Input(shape=(max_length, embedding_features))
bi_lstm_1 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(input_1)
bi_lstm_2 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(bi_lstm_1)
bi_lstm_3 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=False))(bi_lstm_2)
softmax_1 = Dense(units=classes, activation='softmax')(bi_lstm_3)

model = Model(inputs=input_1, outputs=softmax_1)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# show model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='../images/model_plot_flair.png', show_shapes=True, show_layer_names=True)

# training
model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=16, epochs=20, class_weight=class_weight_dict)

# explain predictions
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

# calculate confusion matrix
y = [np.argmax(v) for v in y_val]
x = [np.argmax(x) for x in model.predict(x_val)]
confusion = confusion_matrix(y, x)
classification = classification_report(y, x)
print(confusion)
print(classification)

# print confusion matrix
# import matplotlib.pyplot as plt
# labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']
# cm_df = pd.DataFrame(confusion, labels, labels)
# sn.set(font_scale=1.1, font='Arial')
# ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
# ax.set_xlabel("Actual")
# ax.set_ylabel("Predicted")
# ax.set_title("Confusion Matrix")
# plt.show()
