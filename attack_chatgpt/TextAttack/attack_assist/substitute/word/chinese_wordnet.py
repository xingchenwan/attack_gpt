import os
from .base import WordSubstitute
from ....exceptions import WordNotInDictionaryException
from ....tags import *
import nltk
from nltk.corpus import wordnet as wn

class ChineseWordNetSubstitute(WordSubstitute):

    TAGS = { TAG_Chinese }

    def __init__(self, model_dir, k = None):
        """
        Chinese word substitute based on wordnet.

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
        
        :Language: chinese
        """
        self.k = k
        nltk.download(info_or_id="wordnet", download_dir=os.path.join(model_dir, "nltk_data"))
        nltk.download(info_or_id="omw", download_dir=os.path.join(model_dir, "nltk_data"))
    
    def substitute(self, word: str, pos: str):
        if pos == "other":
            raise WordNotInDictionaryException()
        pos_in_wordnet = {
            "adv": "r",
            "adj": "a",
            "verb": "v",
            "noun": "n"
        }[pos]

        synonyms = []
        for synset in wn.synsets(word, pos=pos_in_wordnet, lang='cmn'):
            for lemma in synset.lemma_names('cmn'):
                if lemma == word:
                    continue
                synonyms.append((lemma, 1))
        
        if self.k is not None:
            return synonyms[:self.k]

        return synonyms
