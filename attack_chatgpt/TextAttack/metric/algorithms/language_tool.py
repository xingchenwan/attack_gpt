from .base import AttackMetric
from ...tags import *
import os
import sys

class LanguageTool(AttackMetric):
    
    
    NAME = "grammatical_errors"
    TAGS = { TAG_English }

    def __init__(self, model_dir) -> None:
        """
        Use language_tool_python to check grammer.

        :Package Requirements:
            * language_tool_python
        :Language: english

        """
        import language_tool_python
        # add download path cache
        self.language_tool = language_tool_python.LanguageTool(language='en-US', cache_path=model_dir)
        # self.language_tool = language_tool_python.LanguageToolPublicAPI('en-us')
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return len(self.language_tool.check(adversarial_sample))

class LanguageToolChinese(AttackMetric):
    
    NAME = "Grammatical Errors"
    TAGS = { TAG_Chinese }

    def __init__(self, model_dir) -> None:
        """
        Use language_tool_python to check grammer.

        :Package Requirements:
            * language_tool_python
        :Language: chinese

        """
        import language_tool_python
        # add download path cache
        self.language_tool = language_tool_python.LanguageTool(language='zh-CN', cache_path=model_dir)
        # self.language_tool = language_tool_python.LanguageToolPublicAPI('zh-cn')
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return len(self.language_tool.check(adversarial_sample))
    