from .base import MetricSelector
class GrammaticalErrors(MetricSelector):
    """
    :English: :py:class:`.LanguageTool`
    :Chinese: :py:class:`.LanguageToolChinese`
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def _select(self, lang):
        if lang.name == "english":
            from ..algorithms.language_tool import LanguageTool
            return LanguageTool(self.model_dir)
        if lang.name == "chinese":
            from ..algorithms.language_tool import LanguageToolChinese
            return LanguageToolChinese(self.model_dir)
    