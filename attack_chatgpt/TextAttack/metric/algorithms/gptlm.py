import os
import math
import transformers
from ...tags import *
from .base import AttackMetric

class GPT2LM(AttackMetric):

    NAME = "fluency_ppl"
    TAGS = { TAG_English }

    def __init__(self, model_dir):
        """
        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__

        :Language: english
        
        """

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=os.path.join(model_dir, "huggingface")) # text_classification/
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=os.path.join(model_dir, "huggingface"))
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            ipt = self.tokenizer(adversarial_sample, return_tensors="pt", verbose=False)
            return math.exp( self.lm(**ipt, labels=ipt.input_ids)[0] )
        return None
    

class GPT2LMChinese(AttackMetric):
    
    NAME = "Fluency (ppl)"
    TAGS = { TAG_Chinese }

    def __init__(self, model_dir):
        """
        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__

        :Package Requirements:
            * tensorflow>=2
        :Language: chinese

        """
        ## TODO train a pytorch chinese gpt-2 model
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained("mymusise/EasternFantasyNoval", cache_dir=os.path.join(model_dir, "huggingface"))
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("mymusise/EasternFantasyNoval", from_tf=True, cache_dir=os.path.join(model_dir, "huggingface"))

    ## FIXME after_attack