import pandas as pd


input = pd.read_csv('../data/test_set.csv')
sentences = input['sentence'].values
labels = input['label'].values
from keras.utils import to_categorical

with open('../data/test_set_ktrain.csv', 'w') as file:
    file.write('sentence,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,neutral')
    file.write('\n')

    labels = to_categorical(labels)
    for i in range(len(input)):
        sentence = sentences[i]
        label = labels[i]

        file.write(sentence)
        for c in label:
            file.write(',')
            file.write(str(int(c)))
        file.write('\n')

