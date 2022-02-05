import ktrain
import pandas as pd


test_set = pd.read_csv('../data/test_set.csv')
sentences = test_set['sentence'].values
y_test = test_set['label'].values

predictor = ktrain.load_predictor('models/bert.predictor.8.save')
prediction = predictor.predict(sentences[3], return_proba=True)
print(prediction)


# explain predictions
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# calculate confusion matrix
y = y_test
x = [np.argmax(x) for x in predictor.predict(x_test)]
confusion = confusion_matrix(y, x)
classification = classification_report(y, x)
print(confusion)
print(classification)
#
# print confusion matrix
import matplotlib.pyplot as plt
labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']
cm_df = pd.DataFrame(confusion, labels, labels)
sn.set(font_scale=1.1, font='Arial')
ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix")
plt.show()
#
# learner.view_top_losses(n=5, preproc=preprocessing)
