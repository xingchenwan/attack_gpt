"""
define customised classifier model.

Args:
    path: the saved pre_trained file path of customised classifier model.

Return:
    A classifier model.

"""

import nltk
import pickle
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class EnClassifierModel():
        def __init__(self):
            try:
                self.model = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
                self.model = SentimentIntensityAnalyzer()
        
        # get_pred return the label and conf score
        def get_pred(self, input_):
            return self.get_prob(input_).argmax(axis=1)

        # get_prob return the positive and negative float
        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 1e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)

class PytorchModel():
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path,'rb'))

    # get_pred return the label and conf score
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # get_prob return the positive and negative float
    def get_prob(self, input_):

        return self.model.predict_proba(input_)