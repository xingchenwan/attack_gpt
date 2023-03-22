from .base import MetricSelector
class SemanticSimilarity(MetricSelector):
    """
    :English: :py:class:`.UniversalSentenceEncoder`
    :Chinese: :py:class:`.UniversalSentenceEncoder`
    """

    def _select(self, lang):
        if lang.name == "english":
            from ..algorithms.usencoder import UniversalSentenceEncoder
            return UniversalSentenceEncoder()
        if lang.name == "chinese":
            from ..algorithms.usencoder import UniversalSentenceEncoder
            return UniversalSentenceEncoder()

        