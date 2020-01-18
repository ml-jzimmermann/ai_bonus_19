import ktrain
import pandas as pd
import numpy as np


test_set = pd.read_csv('../data/test_set.csv')
sentences = test_set['sentence'].values
y_test = test_set['label'].values

predictor = ktrain.load_predictor('models/bert.predictor.8.save')
prediction = predictor.predict(sentences[3], return_proba=True)
print(prediction)

# explain predictions
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
# calculate confusion matrix
y = y_test
x = [np.argmax(x) for x in [predictor.predict(sentence, return_proba=True) for sentence in sentences]]
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
plt.savefig('../images/confusion/confusion_matrix_bert_8.png')

