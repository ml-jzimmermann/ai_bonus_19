# AI Bonus Project WS 2019

Train and compare different classifiers on a self-labeled emotion classification dataset. This project extends https://github.com/ml-jzimmermann/emotion-mining and adds the BERT transformer model.

![explanation_image](https://github.com/ml-jzimmermann/ai_bonus_19/blob/master/images/lstm_explanation.png)

This picture shows an effort to make the reasoning of the classifier visible. Green words positively influenced the prediction while red ones had negative impact. In that example, the model is conncting the words "why", "liver" and "damage" with the emotional category "fear" which makes some sense. Since the sentence i phrased as a question, the emotion of the asking person was correctly labeled as "surprise".

![explanation_image_2](https://github.com/ml-jzimmermann/ai_bonus_19/blob/master/images/lstm_explanation_2.png)

The second picture shows a positively annotated sentence which the model also perceives as positive while not matching the exact same category. The conducted experiments showed that learning human emotions from written text is a very difficult task, especially with limited amounts of training data.
Nevertheless, the transformer model significantly outperformed the recurrent models.
