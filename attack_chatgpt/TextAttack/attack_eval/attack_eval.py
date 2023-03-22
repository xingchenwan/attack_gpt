import sys
import json
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import logging
from tqdm import tqdm
from ..utils import levenshtein_visual, visualizer, result_visualizer, get_language, language_by_name
from .utils import worker_process, worker_init, attack_process
from ..tags import *
from ..text_process.tokenizer import Tokenizer, get_default_tokenizer
from ..victim.base import Victim
from ..attackers.base import Attacker
from ..metric import AttackMetric, MetricSelector
import numpy as np
import multiprocessing as mp

logger = logging.getLogger(__name__)

class AttackEval:
    def __init__(self,
        attacker : Attacker,
        victim : Victim,
        language : Optional[str] = None,
        tokenizer : Optional[Tokenizer] = None,
        invoke_limit : Optional[int] = None,
        metrics : List[Union[AttackMetric, MetricSelector]] = []
    ):
        """
        `AttackEval` is a class used to evaluate attack metrics in TextAttack.

        Args:
            attacker: An attacker, must be an instance of :py:class:`.Attacker` .
            victim: A victim model, must be an instance of :py:class:`.Vicitm` .
            language: The language used for the evaluation. If is `None` then `AttackEval` will intelligently select the language based on other parameters.
            tokenizer: A tokenizer used for visualization.
            invoke_limit: Limit on the number of model invokes.
            metrics: A list of metrics. Each element must be an instance of :py:class:`.AttackMetric` or :py:class:`.MetricSelector` .

        """

        if language is None:
            lst = [attacker]
            if tokenizer is not None:
                lst.append(tokenizer)
            if victim.supported_language is not None:
                lst.append(victim)
            for it in metrics:
                if isinstance(it, AttackMetric):
                    lst.append(it)

            lang_tag = get_language(lst)
        else:
            lang_tag = language_by_name(language)
            if lang_tag is None:
                raise ValueError("Unsupported language `%s` in attack eval" % language)

        self._tags = { lang_tag }

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang_tag)
        else:
            self.tokenizer = tokenizer

        self.attacker = attacker
        self.victim = victim
        self.metrics = []
        for it in metrics:
            if isinstance(it, MetricSelector):
                v = it.select(lang_tag)
                if v is None:
                    raise RuntimeError("`%s` does not support language %s" % (it.__class__.__name__, lang_tag.name))
                self.metrics.append( v )
            elif isinstance(it, AttackMetric):
                self.metrics.append( it )
            else:
                raise TypeError("`metrics` got %s, expect `MetricSelector` or `AttackMetric`" % it.__class__.__name__)
        self.invoke_limit = invoke_limit
        
    @property
    def TAGS(self):
        return self._tags
    
    def __measure(self, data, adversarial_sample):
        ret = {}
        for it in self.metrics:
            value = it.after_attack(data, adversarial_sample)
            if value is not None:
                ret[it.name] = value
        return ret


    def __iter_dataset(self, dataset):
        for data in dataset:
            v = data
            for it in self.metrics:
                ret = it.before_attack(v)
                if ret is not None:
                    v = ret
            yield v
    
    def __iter_metrics(self, iterable_result):
        for data, result in iterable_result:
            adversarial_sample, attack_time, invoke_times = result
            ret = {
                "data": data,
                "success": adversarial_sample is not None,
                "result": adversarial_sample,
                "metrics": {
                    "run_time": attack_time,
                    "query_exceeded": self.invoke_limit is not None and invoke_times > self.invoke_limit,
                    "victim_model_queries": invoke_times,
                    ** self.__measure(data, adversarial_sample)
                }
            }
            yield ret

    def ieval(self, dataset : Iterable[Dict[str, Any]], num_workers : int = 0, chunk_size : Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Iterable evaluation function of `AttackEval` returns an Iterator of result.

        Args:
            dataset: An iterable dataset.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Yields:
            A dict contains the result of each input samples.

        """

        if num_workers > 0:
            ctx = mp.get_context("spawn")
            if chunk_size is None:
                chunk_size = num_workers
            with ctx.Pool(num_workers, initializer=worker_init, initargs=(self.attacker, self.victim, self.invoke_limit)) as pool:
                for ret in self.__iter_metrics(zip(dataset, pool.imap(worker_process, self.__iter_dataset(dataset), chunksize=chunk_size))):
                    yield ret
                   
        else:
            def result_iter():
                for data in self.__iter_dataset(dataset):
                    yield attack_process(self.attacker, self.victim, data, self.invoke_limit)
            for ret in self.__iter_metrics(zip(dataset, result_iter())):
                yield ret

    def eval(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A dict of attack evaluation summaries.

        """


        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])

            if TAG_Classification in self.victim.TAGS:
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        y_adv = probs[1]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])
                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                info["Succeed"] = res["success"]
                if visualize:
                    if progress_bar:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.tokenizer)
                    else:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.tokenizer)
            # return
            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["total_sample_nums"] = total_inst
        summary["succ_sample_nums"] = success_inst
        summary["succ_sample_rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["succeed"]:
                continue
            if kw in ["query_exceeded"]:
                summary["Total " + kw] = total_result[kw]
            if kw in ["run_time"]:
                summary[kw] = total_result[kw]
            else:
                summary["avg_" + kw] = total_result[kw] / total_result_cnt[kw]
        
        if visualize:
            result_visualizer(summary, sys.stdout.write)
        return summary
    
    ## TODO generate return results
    def __token_changes_text(self, x_orig, x_adv, tokenizer):
        ret_orig = ''
        ret_adv = ''
        token_orig = tokenizer.tokenize(x_orig, pos_tagging = False)
        token_adv = tokenizer.tokenize(x_adv, pos_tagging = False)
        pairs = levenshtein_visual(token_orig, token_adv)
        
        for tokenA, tokenB in pairs:
            if tokenA.lower() == tokenB.lower():
                ret_orig += tokenA + ' '
                ret_adv += tokenB + ' '
            else:
                ret_orig += '[[' + tokenA + ']] '
                ret_adv += '[[' + tokenB + ']] '

        return ret_orig, ret_adv



    def get_eval(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, num_workers : int = 0, chunk_size : Optional[int] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A json of attack evaluation text.

        """

        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        result_iterator = self.ieval(dataset, num_workers, chunk_size)
        print(result_iterator)
        # return result_iterator
        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        # Begin for
        all_result = []
        for i, res in enumerate(result_iterator):
            dict_i = {}
            total_inst += 1
            success_inst += int(res["success"])

            if TAG_Classification in self.victim.TAGS:
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs_ori = self.victim.get_prob([x_orig])
                            probs_adv = self.victim.get_prob([x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs_ori
                        y_adv = probs_adv
                        y_orig = np.array(y_orig)
                        y_adv = np.array(y_adv)
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])

                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs
                        y_orig = np.array(y_orig)
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                # info["Succeed"] = res["success"]
            # compute the label and confidence
            if y_adv is None:
                if isinstance(y_orig, int):
                    label_orig, conf_orig, label_adv, conf_adv = y_orig, None, None, None
                else:
                    label_orig, conf_orig, label_adv, conf_adv = y_orig.argmax(), y_orig.max(), None, None
            else:
                if isinstance(y_orig, int):
                    label_orig, conf_orig, label_adv, conf_adv = y_orig, None, y_adv, None
                else:
                    
                    label_orig, conf_orig, label_adv, conf_adv = y_orig.argmax(), y_orig.max(), y_adv.argmax(), y_adv.max()
            # token the changed words
            if x_adv is None:
                ret_orig, ret_adv = x_orig, None
            else:
                ret_orig, ret_adv = self.__token_changes_text(x_orig, x_adv, self.tokenizer)

            # create json format data
            dict_i['id'] = int(i+1)
            dict_i['succeed'] = res['success']
            dict_i['orig'] = ret_orig
            dict_i['orig_label'] = label_orig
            dict_i['orig_conf'] = conf_orig
            dict_i['adv'] = ret_adv
            dict_i['adv_label'] = label_adv
            dict_i['adv_conf'] = conf_adv
            dict_i.update(info)


            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)

            all_result.append(dict_i)
        # End for

        json_result = {}
        json_result['data'] = all_result
        summary = {}
        summary["total_sample_nums"] = total_inst
        summary["succ_sample_nums"] = success_inst
        summary["succ_sample_rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["succeed"]:
                continue
            if kw in ["query_exceeded"]:
                summary["Total " + kw] = total_result[kw]
            if kw in ["run_time"]:
                summary[kw] = total_result[kw]
            else:
                summary["avg_" + kw] = total_result[kw] / total_result_cnt[kw]

        json_result['summary'] = summary

        return json_result

