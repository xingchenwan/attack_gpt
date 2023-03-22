"""
:type: TextAttack.utils.RobertaClassifier
:Size: 1.18GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on IMDB dataset. See :py:data:`Dataset.IMDB` for detail.
"""

from TextAttack.utils import make_zip_downloader

NAME = "Victim.ROBERTA.IMDB"

URL = "/TAADToolbox/victim/roberta_imdb.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    
    from TextAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)