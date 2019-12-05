import ktrain


predictor = ktrain.load_predictor('models/bert.predictor.10.save')
prediction = predictor.predict('i am feeling very good today i am very happy about checkra1n on my ipad')
print(prediction)


# # show model
# from keras.utils.vis_utils import plot_model
# plot_model(learner.model, to_file='../images/model_plot_bert.png', show_shapes=True, show_layer_names=True)
#
# # explain predictions
# import seaborn as sn
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
#
# # calculate confusion matrix
# y = [np.argmax(v) for v in y_test]
# x = [np.argmax(x) for x in model.predict(x_test)]
# confusion = confusion_matrix(y, x)
# classification = classification_report(y, x)
# print(confusion)
# print(classification)
#
# # print confusion matrix
# # import matplotlib.pyplot as plt
# # labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']
# # cm_df = pd.DataFrame(confusion, labels, labels)
# # sn.set(font_scale=1.1, font='Arial')
# # ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
# # ax.set_xlabel("Actual")
# # ax.set_ylabel("Predicted")
# # ax.set_title("Confusion Matrix")
# # plt.show()
#
# learner.view_top_losses(n=5, preproc=preprocessing)
