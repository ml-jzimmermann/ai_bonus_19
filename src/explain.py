
# explain predictions
import ktrain
from ktrain.text.preprocessor import TextPreprocessor
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_val, y_val))

class LSTM_Preprocessor(TextPreprocessor):
    def __init__(self, tokenizer, max_length, num_words):
        super().__init__(max_length, [], lang='en')
        self.tokenizer = tokenizer
        self.tokenizer_dict = {}
        self.max_length = max_length

    def get_preprocessor(self):
        return (self.tokenizer, self.tokenizer_dict)

    def preprocess(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        seq = pad_sequences(seq, maxlen=self.max_length)
        return (seq, None)

    def undo(self, doc):
        dct = self.tokenizer.index_word
        return " ".join([dct[wid] for wid in doc if wid != 0 and wid in dct])

preprocessor = LSTM_Preprocessor(tokenizer=tokenizer, max_length=max_length, num_words=max_words)

learner.view_top_losses(n=5, preproc=preprocessor)

# predictor = ktrain.get_predictor(learner.model, preproc=preprocessor)
# predictor.predict(['this is a sentence'])

