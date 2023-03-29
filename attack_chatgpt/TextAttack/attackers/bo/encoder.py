from typing import List, Any, Union
from sentence_transformers import SentenceTransformer


class BaseEncoder:
  """
  Obtain embeddings of input strings / sentences from some pretrained
  lanugage model.
  """

  def __init__(
      self,
      model_spec: str,
  ):
    self.model_spec = model_spec

  def _get_embedding(self, text: List[str]):
    raise NotImplementedError()

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    return self._get_embedding(*args, **kwds)


class SentenceEncoder(BaseEncoder):

  def __init__(self,
               model_spec: str = "all-mpnet-base-v2",
               device: str = "cuda:0"):
    """Obtain a sentence-level embedding with Sentence-BERT model.
    """
    BaseEncoder.__init__(self, model_spec=model_spec)
    self.encoder = SentenceTransformer(model_spec, device=device)

  def _get_embedding(self, input: List[str]):
    embedding = self.encoder.encode(input)
    return embedding