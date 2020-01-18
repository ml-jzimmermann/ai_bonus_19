import pandas as pd
import numpy as np


for index_run in range(10):
    input = pd.read_csv('../data/complete.csv')
    texts = input['sentence']
    labels = input['label'].values

    # preprocessing text
    stop_list = 'the to and i a of that is in you for it have he my with was are on but be this so not'.split()

    def remove_stopwords(text):
        text = text.split(' ')
        for word in text:
            if word in stop_list:
                text.remove(word)
        return (' ').join(text)

    texts = texts.values
    # tokenization
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    max_words = 10000
    max_length = 75
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    texts = tokenizer.sequences_to_texts(sequences)
    texts = list(map(remove_stopwords, texts))
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
    x_train, x_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, shuffle=True, random_state=42)
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

    features = 200
    input_1 = Input(shape=(max_length,))
    embed_1 = Embedding(input_dim=(max_words - 1), output_dim=features, input_length=max_length)(input_1)
    bi_lstm_1 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(embed_1)
    bi_lstm_2 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(bi_lstm_1)
    bi_lstm_3 = Bidirectional(LSTM(units=16, activation='tanh', dropout=0.2, return_sequences=False))(bi_lstm_2)
    softmax_1 = Dense(units=classes, activation='softmax')(bi_lstm_3)

    model = Model(inputs=input_1, outputs=softmax_1)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print(model.summary())

    # show model
    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file='../images/model_plot_lstm.png', show_shapes=True, show_layer_names=True)

    # training
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=16, epochs=20, class_weight=class_weight_dict)
    print(history.history)
    with open('../data/history/history_' + str(index_run) + '.txt', 'w') as file:
        file.write(str(history.history))

    # explain predictions
    import seaborn as sn
    from sklearn.metrics import classification_report, confusion_matrix

    # calculate confusion matrix
    y = [np.argmax(v) for v in y_val]
    x = [np.argmax(x) for x in model.predict(x_val)]
    confusion = confusion_matrix(y_true=y, y_pred=x, labels=None, sample_weight=None, normalize='true')
    classification = classification_report(y, x)
    print(confusion)
    print(classification)

    # print confusion matrix
    import matplotlib.pyplot as plt
    plt.clf()
    labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']
    cm_df = pd.DataFrame(confusion, labels, labels)
    sn.set(font_scale=1.1, font='Arial')
    ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('../images/confusion/confusion_matrix_' + str(index_run) + '.png')
