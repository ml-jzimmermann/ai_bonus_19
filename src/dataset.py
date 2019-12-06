import pandas as pd
import os
import matplotlib.pyplot as plt


root = '../data/'
complete_df = pd.DataFrame()
for file in os.listdir('../data'):
    if '.csv' in file and 'sentences_' in file:
        csv = pd.read_csv(root + file)[['sentence', 'label']]
        csv = csv.dropna(axis=0)
        if len(complete_df) == 0:
            complete_df = csv
        else:
            complete_df = pd.concat([complete_df, csv], axis=0)

print(complete_df.reset_index().isna().sum())
complete_df['label'] = complete_df['label'].apply(lambda x: int(x))

complete_df.to_csv('../data/complete.csv', index=False)

frequency = {'joy':0, 'trust':0, 'fear':0, 'surprise':0, 'sadness':0, 'disgust':0, 'anger':0, 'anticipation':0, 'neutral':0}
emotions = {0:'joy', 1:'trust', 2:'fear', 3:'surprise', 4:'sadness', 5:'disgust', 6:'anger', 7:'anticipation', 8:'neutral'}
labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']

for index, row in complete_df.iterrows():
    label = int(row['label'])
    emotion = emotions[label]
    if emotion in frequency:
        frequency[emotion] = frequency[emotion] + 1
    else:
        frequency[emotion] = 1
print(frequency)

exit()

plt.bar(range(len(frequency)), frequency.values(), align='center')
plt.xticks(range(len(frequency)), list(frequency.keys()))
plt.title('classes')
plt.show()

texts = complete_df['sentence'].values
from keras.preprocessing.text import Tokenizer
import numpy as np
t = Tokenizer(num_words=10000)
t.fit_on_texts(texts)
sequences = t.texts_to_sequences(texts)

lengths = [len(text) for text in sequences]
print(np.max(lengths))
print(np.min(lengths))
print(np.mean(lengths))
plt.hist(lengths)
plt.title('lengths')
plt.show()
