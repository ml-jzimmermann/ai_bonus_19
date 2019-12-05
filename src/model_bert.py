import ktrain.text as text
import ktrain
import pandas as pd


input = pd.read_csv('../data/complete.csv')

(x_train, y_train), (x_test, y_test), preprocessing = text.texts_from_df(train_df=input,
                                            text_column='sentence', label_columns='label', preprocess_mode='bert',
                                            val_pct=0.15, max_features=10000, maxlen=75)

model = text.text_classifier(name='bert', train_data=(x_train, y_train), preproc=preprocessing)

learner = ktrain.get_learner(model=model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=16)

learner.fit_onecycle(lr=2e-5, epochs=10)

learner.save_model('models/bert.learner.10.save')
predictor = ktrain.get_predictor(model=learner.model, preproc=preprocessing)
predictor.save('models/bert.predictor.10.save')
