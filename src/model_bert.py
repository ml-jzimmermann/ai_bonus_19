import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ktrain.text as text
import ktrain
import pandas as pd

preprocessing = None
learner = None
model = None
for index_run in range(1,10):
    input = pd.read_csv('../data/complete_ktrain.csv')

    (x_train, y_train), (x_val, y_val), preprocessing = text.texts_from_df(train_df=input, text_column='sentence',
                                                                           label_columns=['joy', 'trust', 'fear',
                                                                                          'surprise', 'sadness',
                                                                                          'disgust', 'anger',
                                                                                          'anticipation', 'neutral'],
                                                                           preprocess_mode='bert',
                                                                           val_pct=0.2, max_features=1000, maxlen=75)

    model = text.text_classifier(name='bert', train_data=(x_train, y_train), preproc=preprocessing)

    learner = ktrain.get_learner(model=model, train_data=(x_train, y_train), val_data=(x_val, y_val), batch_size=16)

    history = learner.fit_onecycle(lr=3e-5, epochs=8)
    print(history.history)
    with open('../data/history/history_bert_' + str(index_run) + '.txt', 'w') as file:
        file.write(str(history.history))

learner.save_model('models/bert.learner.8.save')
predictor = ktrain.get_predictor(model=learner.model, preproc=preprocessing)
predictor.save('models/bert.predictor.8.save')
