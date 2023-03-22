'''
下载数据集
加载数据集
整理数据集
数据集处理
攻击方法
模型预测
结果判断

数据集包括：定义基础数据集
数据处理操作：1.不添加prompt；2.添加prompt
测试模型输入数据集：1.使用通用的token；2.直接使用原始sentence
攻击方法：黑盒查询攻击
评价指标支持：1.acc；2相似度；3.修改单词数量
'''
import os
import sys
import time
import copy
import random
import argparse
import logging as log
import _pickle as pkl
import ipdb as pdb
import torch
import tqdm
import datasets
import numpy as np
# from allennlp.data.iterators import BasicIterator, BucketIterator
# from util import device_mapping
from inference import Inference
from data_process.preprocess import build_tasks
import nltk
from data_process.numeric_field import NumericField
from config import LABEL_SET, PROMPT_SET, LABEL_TO_ID, DATA_PATH, MODEL_SET, MODEL_SET_TRANS
from data_process.tasks import CoLATask, MRPCTask, MultiNLITask, QQPTask, RTETask, \
                  QNLITask, QNLIv2Task, SNLITask, SSTTask, STSBTask, WNLITask
import pandas as pd
# nltk.download('punkt')
PATH_PREFIX = '/data/work/8g7/attack_chatgpt/glue_data/'
ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnliv2', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
NAME2INFO = {'sst2': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP'),
             'sts-b': (STSBTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'qnliv2': (QNLIv2Task, 'QNLIv2/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/')
            }
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

def sum_n(scen):
    n = ''
    for i in range(len(scen)):
        n += scen[i]
        n += " "
    return n 

def get_content_by_idx(idx, task, data_task):
    if task == 'sst2':
        content = data_task[0][idx]
    elif task == 'qqp':
        content = data_task[0][idx] + \
            ' ' + data_task[1][idx]
    elif task == 'mnli':
        content = data_task[0][idx] + \
            ' ' + data_task[1][idx]
    elif task == 'qnli':
        content = data_task[0][idx] + \
            ' ' + data_task[1][idx]
    elif task == 'rte':
        content = data_task[0][idx] + \
            ' ' + data_task[1][idx]
    elif task == 'mnli-mm':
        content = data_task[0][idx] + \
            ' ' + data_task[1][idx]
    label = data_task[2][idx]
    scent = sum_n(content)
    return scent, label

def merge_res(args):
    df = pd.DataFrame()
    dataset = args.dataset
    task = args.task
    if task.__contains__('translation'):
        model_list = MODEL_SET_TRANS[args.service]
    else:
        model_list = MODEL_SET[args.service]
    for model in model_list:
        res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '.csv')
        df['idx'] = res['idx']
        df['content'] = res['content']
        df['true_label'] = res['true_label']
        df['pred-'+model.replace('/', '_')] = res['pred_label']
    df.to_csv(
        f'result/merge_{dataset}_{task}_{args.service}.csv', index=False)


def compute_metric(pred_label, true_label, task):
    if task.__contains__('translation'):
        import jieba
        import nltk.translate.bleu_score as bleu
        import nltk.translate.gleu_score as gleu
        import nltk.translate.meteor_score as meteor
        # import nltk
        # nltk.download('wordnet')
        # jieba.enable_paddle()

        ref_list = [[list(jieba.cut(item.strip(), use_paddle=True, cut_all=False))]
                    for item in true_label]
        hyp_list = [list(jieba.cut(item.strip(), use_paddle=True,
                         cut_all=False)) for item in pred_label]
        bleu_score = []
        for r, h in zip(ref_list, hyp_list):
            s = bleu.sentence_bleu(r, h)
            bleu_score.append(s)
        bleu_score = np.mean(bleu_score)
        gleu_score = []
        for r, h in zip(ref_list, hyp_list):
            s = gleu.sentence_gleu(r, h)
            gleu_score.append(s)
        gleu_score = np.mean(gleu_score)
        meteor_score = []
        for r, h in zip(ref_list, hyp_list):
            s = meteor.meteor_score(r, h)
            meteor_score.append(s)
        meteor_score = np.mean(meteor_score)
        return {'bleu': bleu_score * 100.0, 'gleu': gleu_score * 100.0, 'meteor_score': meteor_score * 100.0}
    else:
        return {'num_examples': len(pred_label), 'acc': np.mean(pred_label == true_label) * 100.0, 'asr': 100.0 - np.mean(pred_label == true_label) * 100.0}


def stat(args):
    df = pd.read_csv(
        f'result/merge_{args.dataset}_{args.task}_{args.service}.csv')
    labels = {}
    labels['true_label'] = df['true_label'].to_numpy()
    if args.task.__contains__('translation'):
        model_list = MODEL_SET_TRANS[args.service]
    else:
        model_list = MODEL_SET[args.service]
    for model in model_list:
        labels['pred-'+model.replace('/', '_')] = df['pred-' +
                                                     model.replace('/', '_')].to_numpy()
    for key in labels.keys():
        if key != 'true_label':
            if args.service in ['gpt', 'chat'] and args.dataset != 'advglue-t':
                pred_label = []
                for label in labels[key]:
                    orig = label
                    label = label.strip()
                    if "not_entail" in label or "not_ent" in label:
                        label = "not_entailment"
                    elif "entails" in label or "is_entailment" in label \
                        or "two sentences are entailment" in label or "entailment." in label \
                            or "entailment relation holds" in label or "\"entailment\"" in label:
                        label = "entailment"
                    elif "the two sentences are neutral" in label or "neutral." in label or "these two sentences are neutral to each other" in label:
                        label = "neutral"
                    elif "the two sentences are a contradiction" in label or "contradiction." in label \
                        or "the first two sentences are a contradiction" in label or "the two sentences are contradictory" in label:
                        label = "contradiction"
                    elif "two questions are equivalent" in label or "therefore equivalent" in label:
                        label = "equivalent"
                    elif "two questions are not equivalent" in label or "not equivalent." in label or "they are not exactly equivalent" in label:
                        label = "not_equivalent"
                    elif "the sentence is negative" in label or "the sentence as negative" in label or "the answer is \"negative\"" in label:
                        label = "negative"
                    elif "the second sentence is not entailment" in label or "do not entail each other" in label \
                        or "the question and the sentence are not entailed" in label or "the given question and sentence are not related" in label:
                        label = "not_entailment"
                    elif "the classification would be \"positive\"." in label or "the answer would be \"positive\"." in label:
                        label = "positive"
                    if '_' in label:
                        label = label.split('_')[-1] if label not in LABEL_TO_ID[args.task] else label
                    if '.' in label:
                        label = label.strip('.')
                        
                    try:
                        pred_label.append(LABEL_TO_ID[args.task][label])
                    except:
                        print(orig)
                        pred_label.append(-1)
                pred_label = np.array(pred_label)

                if args.dataset == 'flipkart':
                    true_label = []
                    for label in labels['true_label']:
                        true_label.append(LABEL_TO_ID[args.task][label])
                    true_label = np.array(true_label)
                    # acc = np.mean(pred_label == true_label)
                    metric_dict = compute_metric(
                        pred_label, true_label, args.task)
                else:
                    # acc = np.mean(pred_label == labels['true_label'])
                    metric_dict = compute_metric(
                        pred_label, labels['true_label'], args.task)
            elif args.dataset == 'flipkart':
                true_label = []
                for label in labels['true_label']:
                    true_label.append(LABEL_TO_ID['sst2'][label])
                true_label = np.array(true_label)
                # acc = np.mean(labels[key] == true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            elif args.dataset == 'advglue':
                pred_label = []
                for label in labels[key]:
                    pred_label.append(LABEL_TO_ID[args.task][label])
                pred_label = np.array(pred_label)
                # acc = np.mean(labels[key] == true_label)
                metric_dict = compute_metric(
                    pred_label, labels['true_label'], args.task)
            elif args.dataset == 'anli':
                true_label = []
                map_dict = {'e': 'entailment',
                            'c': 'contradiction', 'n': 'neutral'}
                for label in labels['true_label']:
                    true_label.append(map_dict[label])
                true_label = np.array(true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            elif args.dataset == 'ddxplus':
                true_label = []
                for label in labels['true_label']:
                    true_label.append(label.lower())
                true_label = np.array(true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            else:
                # acc = np.mean(labels[key] == labels['true_label'])
                metric_dict = compute_metric(
                    labels[key], labels['true_label'], args.task)

            metric_string = ', '.join(
                ['{:s}:{:.2f}'.format(k, v) for k, v in metric_dict.items()])
            print("{:s} - {:s}".format(key, metric_string))

sys.path.insert(0, os.path.join(os.getcwd(), "ThirdParty_code/text_classification")) # text_classification/
import nltk
from nltk import data as nltk_data

import TextAttack
from built_in_models import EnClassifierModel

ATTACK_METHODS = {
    'genetic': 'GeneticAttacker',
    # 'scpn': 'SCPNAttacker',
    # 'fd': 'FDAttacker',
    'hotflip': 'HotFlipAttacker',
    'textfooler': 'TextFoolerAttacker',
    'pwws': 'PWWSAttacker',
    # 'uat': 'UATAttacker',
    # 'viper': 'VIPERAttacker',
    # 'deepwordbug': 'DeepWordBugAttacker',
    # 'gan': 'GANAttacker',
    'textbugger': 'TextBuggerAttacker',
    'pso': 'PSOAttacker',
    # 'bert_attack': 'BERTAttacker',
    # 'bae': 'BAEAttacker',
    # 'geometry': 'GEOAttacker'
}

# QUERY_ATTACK_METHODS = ['pwws', 'genetic', 'textfooler', 'textbugger', 'pso']
QUERY_ATTACK_METHODS = ['textbugger','textfooler']
# QUERY_ATTACK_METHODS = ['textfooler']

BLACK_ATTACK_METHODS = ['hotflip']

# ATTACK_TYPES = ['query', 'black', 'all']
ATTACK_TYPES = ['query']



DATASETS = {
    'sst' : 'sst',
    # 'ag' : 'ag',
    # 'imdb' : 'imdb',
    'amazon_zh' : 'amazon_reviews_multi'
}


VICTIM_MODELS = {
    'albert_ag': 'ALBERT.AG',
    'albert_sst': 'ALBERT.SST',
    'albert_imdb': 'ALBERT.IMDB',
    'roberta_ag': 'ROBERTA.AG',
    'roberta_sst': 'ROBERTA.SST',
    'roberta_imdb': 'ROBERTA.IMDB',
    'xlnet_ag': 'XLNET.AG',
    'xlnet_sst': 'XLNET.SST',
    'xlnet_imdb': 'XLNET.IMDB',
    'bert_zh': 'BERT.AMAZON_ZH',
    'bert_en': 'BERT.SST',
    'customised': 'customised'
}


def get_support_datasets():
    """
    Return:
        A list of support datasets.

    """
    return list(DATASETS.keys())


def get_support_victim_models():
    """
    Return:
        A list of support victim models.

    """
    return list(VICTIM_MODELS.keys())


def _transform_score_demo_(score):
    """
    compute the final score.

    Args:
        score: the score diff before adv or not.
    
    Return:
        A float of final score.

    """
    assert score >= -1.0 and score <= 1.0, 'score {} should in range[-1, 1]'.format(score)
    fscore = 0
    if score <=0:
        return 0
    if score <= 0.4:
        fscore = score / 0.4 * 0.7
    elif score <= 0.8:
        fscore = 0.7 + (score - 0.4) / 0.4 * 0.1
    else:
        fscore = score
    return fscore


def get_victim_model_scroe(json_result, label_gt):
    """
    compute the victim model score.

    Args:
        data: the model results in original and adverial samples.
    
    Return:
        A float of victim model score.

    """
    text_data = json_result['data']
    assert len(text_data) == len(label_gt), 'The number of groundtruth samples %d is not equal to the adv samples %d'%(len(label_gt), len(text_data))
    orig_count = 0
    adv_count = 0
    for i in range(len(label_gt)):
        # print(text_data[i]['orig'])
        if text_data[i]['succeed']:
            if int(label_gt[i]) == int(text_data[i]['orig_label']):
                orig_count += 1
            if int(label_gt[i]) == int(text_data[i]['adv_label']):
                adv_count += 1
        else:
            if int(label_gt[i]) == int(text_data[i]['orig_label']):
                orig_count += 1
                adv_count += 1
    
    score_t = (orig_count - adv_count) / len(label_gt)
    score = min((1-score_t), 1.0)

    victim_model_scroe = _transform_score_demo_(score)

    return victim_model_scroe


def get_attacker(attack_name, lang, model_dir):
    """
    get attacker method object.

    Args:
        attack_name: the name of attack methods, must in QUERY_ATTACK_METHODS or BLACK_ATTACK_METHODS.
        lang: the test dataset type, must in ['english', 'chinese']
    
    Return:
        A attacker class.

    """

    if attack_name == 'pwws':
        attacker = TextAttack.attackers.PWWSAttacker(lang=lang, model_dir=model_dir)
    elif attack_name == 'genetic':
        attacker = TextAttack.attackers.GeneticAttacker(lang=lang, model_dir=model_dir)
    elif attack_name == 'textfooler':
        attacker = TextAttack.attackers.TextFoolerAttacker(lang=lang, model_dir=model_dir)
    elif attack_name == 'textbugger':
        attacker = TextAttack.attackers.TextBuggerAttacker(lang=lang, model_dir=model_dir)
    elif attack_name == 'pso':
        attacker = TextAttack.attackers.PSOAttacker(lang=lang, model_dir=model_dir)
    elif attack_name == 'hotflip':
        attacker = TextAttack.attackers.HotFlipAttacker(lang=lang, model_dir=model_dir)
    else:
        raise ValueError("Unsupported attack name `%s`" % attack_name)
    
    return attacker


def get_victim_model(victim):
    """
    extend the "TextAttack.Classifier" to user victim model.

    Args:
        victim: the user model object.
    
    Return:
        A model extend the "TextAttack.Classifier".

    """
    class victimModel(TextAttack.Classifier):
        def __init__(self):
            self.model = victim

        def get_pred(self, input_):
            return self.model.get_pred(input_)

        def get_prob(self, input_):
            return self.model.get_prob(input_)

    return victimModel()

def get_victim_model_gpt(victim,name):
    """
    extend the "TextAttack.Classifier" to user victim model.

    Args:
        victim: the user model object.
    
    Return:
        A model extend the "TextAttack.Classifier".

    """
    class victimModel(TextAttack.Classifier):
        def __init__(self):
            self.model = victim
            self.name = name

        def get_pred(self, input_):
            if len(input_) == 1:
                pred_label = self.model.predict(input_, prompt=PROMPT_SET[self.name][-1])
                return pred_label["pred_class"]
            else:
                precls = []
                for i in range(len(input_)):
                    pred_label = self.model.predict(input_[i], prompt=PROMPT_SET[self.name][-1])
                    precls.append(pred_label["pred_class"])
                return precls

        def get_prob(self, input_):
            if len(input_) == 1:
                pred_label = self.model.predict(input_, prompt=PROMPT_SET[self.name][-1])
                return pred_label["pred_conf"]
            else:
                precls = []
                for i in range(len(input_)):
                    pred_label = self.model.predict(input_[i], prompt=PROMPT_SET[self.name][-1])
                    precls.append(pred_label["pred_conf"])
                return precls
           

    return victimModel()

def get_advTextResults(attack_type, data_scent, data_label, data_type, model_dir, victim_model=None, prompt_name=None, extra={}):
    """
    interface main function to server.

    Args:
        attack_type: the name of attack methods, must in ['query', 'black', 'all'].
        data: the test dataset, {'text':[], 'label':[]}
        data_type: the test dataset type, must in ['english', 'chinese']
        model_dir: the path of pre-trained model
        victim_model: the attack model, created by 'entry.py' class method
        extra: other parameter, such as: iterable times, modified rate etc.
    
    Return:
        A dict contains all text results.

    """
    # modify the load path of pretrained model and file
    nltk_data.path.insert(0, model_dir)
    TextAttack.set_path(path=os.path.join(model_dir, "data"))
    csv_data_label = []
    csv_data_text = []
    # init victim model
    if victim_model is None:
        ONLY_SCORE = True
        if data_type == 'english':
            victim_model = TextAttack.loadVictim("BERT.SST")
        elif data_type == 'chinese':
            victim_model = TextAttack.loadVictim("BERT.AMAZON_ZH")
        else:
            raise ValueError("Unsupported language `%s`" % data_type)
    else:
        ONLY_SCORE = False
        victim_model = get_victim_model_gpt(victim_model,prompt_name)

    # init dataset
    data_temp = {}
    csv_data_label.append(int(data_label))
    csv_data_text.append(data_scent)
    data_temp['x'] = csv_data_text
    data_temp['y'] = csv_data_label
    print(data_temp)
    data = datasets.Dataset.from_dict(data_temp)

    # choose metrics
    eval_metrics=[
        TextAttack.metric.Fluency(model_dir),
        TextAttack.metric.GrammaticalErrors(model_dir),
        TextAttack.metric.SemanticSimilarity(),
        TextAttack.metric.EditDistance(),
        TextAttack.metric.ModificationRate()
    ]

    # choose attack type
    if attack_type not in ATTACK_TYPES:
        raise ValueError("Unsupported attack type `%s`" % attack_type)
    elif attack_type == 'query':
        score_query = 0.0
        score_black = None
        final_result = {}
        for i in range(len(QUERY_ATTACK_METHODS)):
            attacker = get_attacker(QUERY_ATTACK_METHODS[i], data_type, model_dir)
            attack_eval = TextAttack.AttackEval(attacker, victim_model, metrics=eval_metrics)
            adv_result = attack_eval.get_eval(data)
            if not ONLY_SCORE:
                final_result[QUERY_ATTACK_METHODS[i]] = adv_result
            score = get_victim_model_scroe(adv_result, data['y'])
            score_query += score
        final_dict = {}
        if not ONLY_SCORE:
            final_dict['results'] = final_result
        final_dict['query_score'] = score_query / len(QUERY_ATTACK_METHODS)
        final_dict['black_score'] = score_black
        final_dict['total_score'] = final_dict['query_score']
    elif attack_type == 'black':
        score_query = None
        score_black = 0.0
        final_result = {}
        for i in range(len(BLACK_ATTACK_METHODS)):
            attacker = get_attacker(BLACK_ATTACK_METHODS[i], data_type, model_dir)
            attack_eval = TextAttack.AttackEval(attacker, victim_model, metrics=eval_metrics)
            adv_result = attack_eval.get_eval(data)
            if not ONLY_SCORE:
                final_result[BLACK_ATTACK_METHODS[i]] = adv_result
            score = get_victim_model_scroe(adv_result, data['y'])
            score_black += score
        final_dict = {}
        if not ONLY_SCORE:
            final_dict['results'] = final_result
        final_dict['query_score'] = score_query
        final_dict['black_score'] = score_black / len(BLACK_ATTACK_METHODS)
        final_dict['total_score'] = final_dict['black_score']
    else:
        ALL_METHODS = QUERY_ATTACK_METHODS + BLACK_ATTACK_METHODS
        score_query = 0.0
        score_black = 0.0
        final_result = {}
        for i in range(len(ALL_METHODS)):
            attacker = get_attacker(ALL_METHODS[i], data_type, model_dir)
            attack_eval = TextAttack.AttackEval(attacker, victim_model, metrics=eval_metrics)
            adv_result = attack_eval.get_eval(data)
            if not ONLY_SCORE:
                final_result[ALL_METHODS[i]] = adv_result
            score = get_victim_model_scroe(adv_result, data['y'])
            if ALL_METHODS[i] in QUERY_ATTACK_METHODS:
                score_query += score
            else:
                score_black += score
        final_dict = {}
        if not ONLY_SCORE:
            final_dict['results'] = final_result
        final_dict['query_score'] = score_query / len(QUERY_ATTACK_METHODS)
        final_dict['black_score'] = score_black / len(BLACK_ATTACK_METHODS)
        final_dict['total_score'] = (score_query + score_black) / len(ALL_METHODS)

    return final_dict


def main(arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='text-davinci-003')
    parser.add_argument('--service', type=str, default='gpt')
    parser.add_argument('--test_num', type=int, default=10, help='attack number of data ')
    parser.add_argument('--begin_num', type=int, default=100, help='attack  begin num ')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--attack', nargs='?', const=True, default=False, help='attack')
    # Logistics
    parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=19)

    # Paths and logging
    parser.add_argument('--log_file', help='file to log to', type=str, default='log.log')
    parser.add_argument('--exp_dir', help='directory containing shared preprocessing', type=str)
    parser.add_argument('--run_dir', help='directory for saving results, models, etc.', type=str)
    parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='')
    parser.add_argument('--preproc_file', help='file containing saved preprocessing stuff',
                        type=str, default='preproc.pkl')

    # Time saving flags
    parser.add_argument('--should_train', help='1 if should train model', type=int, default=1)
    parser.add_argument('--load_model', help='1 if load from checkpoint', type=int, default=1)
    parser.add_argument('--load_epoch', help='Force loading from a certain epoch', type=int,
                        default=-1)
    parser.add_argument('--load_tasks', help='1 if load tasks', type=int, default=1)
    parser.add_argument('--load_preproc', help='1 if load vocabulary', type=int, default=1)

    # Tasks and task-specific classifiers
    parser.add_argument('--train_tasks', help='comma separated list of tasks, or "all" or "none"',
                        type=str)
    parser.add_argument('--eval_tasks', help='list of additional tasks to train a classifier,' +
                        'then evaluate on', type=str, default='')
    parser.add_argument('--classifier', help='type of classifier to use', type=str,
                        default='log_reg', choices=['log_reg', 'mlp', 'fancy_mlp'])
    parser.add_argument('--classifier_hid_dim', help='hid dim of classifier', type=int, default=512)
    parser.add_argument('--classifier_dropout', help='classifier dropout', type=float, default=0.0)

    # Preprocessing options
    parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=40)
    parser.add_argument('--max_word_v_size', help='max word vocab size', type=int, default=30000)

    # Embedding options
    parser.add_argument('--dropout_embs', help='dropout rate for embeddings', type=float, default=.2)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--glove', help='1 if use glove, else from scratch', type=int, default=1)
    parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=0)
    parser.add_argument('--elmo', help='1 if use elmo', type=int, default=0)
    parser.add_argument('--deep_elmo', help='1 if use elmo post LSTM', type=int, default=0)
    parser.add_argument('--elmo_no_glove', help='1 if no glove, assuming elmo', type=int, default=0)
    parser.add_argument('--cove', help='1 if use cove', type=int, default=0)

    # Model options
    parser.add_argument('--pair_enc', help='type of pair encoder to use', type=str, default='simple',
                        choices=['simple', 'attn'])
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=4096)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=1)
    parser.add_argument('--n_layers_highway', help='num of highway layers', type=int, default=1)
    parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=.2)

    # Training options
    parser.add_argument('--no_tqdm', help='1 to turn off tqdm', type=int, default=0)
    parser.add_argument('--trainer_type', help='type of trainer', type=str,
                        choices=['sampling', 'mtl'], default='sampling')
    parser.add_argument('--shared_optimizer', help='1 to use same optimizer for all tasks',
                        type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--optimizer', help='optimizer to use', type=str, default='sgd')
    parser.add_argument('--n_epochs', help='n epochs to train for', type=int, default=10)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=1.0)
    parser.add_argument('--min_lr', help='minimum learning rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', help='max grad norm', type=float, default=5.)
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0)
    parser.add_argument('--task_patience', help='patience in decaying per task lr',
                        type=int, default=0)
    parser.add_argument('--scheduler_threshold', help='scheduler threshold',
                        type=float, default=0.0)
    parser.add_argument('--lr_decay_factor', help='lr decay factor when val score doesn\'t improve',
                        type=float, default=.5)

    # Multi-task training options
    parser.add_argument('--val_interval', help='Number of passes between validation checks',
                        type=int, default=10)
    parser.add_argument('--max_vals', help='Maximum number of validation checks', type=int,
                        default=100)
    parser.add_argument('--bpp_method', help='if using nonsampling trainer, ' +
                        'method for calculating number of batches per pass', type=str,
                        choices=['fixed', 'percent_tr', 'proportional_rank'], default='fixed')
    parser.add_argument('--bpp_base', help='If sampling or fixed bpp' +
                        'per pass, this is the bpp. If proportional, this ' +
                        'is the smallest number', type=int, default=10)
    parser.add_argument('--weighting_method', help='Weighting method for sampling', type=str,
                        choices=['uniform', 'proportional'], default='uniform')
    parser.add_argument('--scaling_method', help='method for scaling loss', type=str,
                        choices=['min', 'max', 'unit', 'none'], default='none')
    parser.add_argument('--patience', help='patience in early stopping', type=int, default=5)
    parser.add_argument('--task_ordering', help='Method for ordering tasks', type=str, default='given',
                        choices=['given', 'random', 'random_per_pass', 'small_to_large', 'large_to_small'])

    args = parser.parse_args(arguments)

    # Logistics #
    # log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    # log_file = os.path.join(args.run_dir, args.log_file)
    # file_handler = log.FileHandler(log_file)
    # log.getLogger().addHandler(file_handler)
    # log.info(args)
    # seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    # random.seed(seed)
    # torch.manual_seed(seed)
    # if args.cuda >= 0:
    #     log.info("Using GPU %d", args.cuda)
    #     torch.cuda.set_device(args.cuda)
    #     torch.cuda.manual_seed_all(seed)
    # log.info("Using random seed %d", seed)

    # Load tasks #
    log.info("Loading tasks...")
    start_time = time.time()
    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == 'all':
            tasks = ALL_TASKS
        elif task_list == 'none':
            tasks = []
        else:
            tasks = task_list.split(',')
        return tasks

    train_task_names = parse_tasks(args.train_tasks)
    eval_task_names = parse_tasks(args.eval_tasks)
    all_task_names = list(set(train_task_names + eval_task_names))
    print(all_task_names)
    for name in all_task_names:
        task = NAME2INFO[name][0](NAME2INFO[name][1], 40, name)
        # task = NAME2INFO[name][0]
        a = task.train_data_text
        # a = task.test_data_text
        # a = task.val_data_text
        infer = Inference(name, args.service, LABEL_SET, MODEL_SET, LABEL_TO_ID, args.model, args.gpu)
        data_len = len(a)
        args.save_file = 'result/' + "glue"+ '_' + name + \
            '_' + args.service + '_' + args.model.replace('/', '_') + '.csv'
        lst = []
        flag_num = 0
        for idx in range(len(a[0])):
            idx += args.begin_num
            if flag_num < args.test_num:
                res_dict = {}
                content, label = get_content_by_idx(idx, name, a)
                model_dir = '/data/work/8g7/attack_chatgpt/ThirdParty_code/text_classification'
                TextAttack.set_path(path=os.path.join(model_dir, "data"))
                # model = TextAttack.loadVictim("BERT.SST")
                if args.attack:
                    final_dict_result = get_advTextResults(
                        attack_type = 'query',
                        data_scent = content,
                        data_label = label,
                        data_type = 'english', # chinese, english
                        model_dir = model_dir,
                        victim_model = infer,
                        prompt_name = name,
                    )
                    lst.append(final_dict_result)
                flag_num += 1
            else:
                break
    print(lst)
            # for i in range(len(QUERY_ATTACK_METHODS)):

            #     args.save_file = 'result/' + "glue"+ '_' + name + '_' + QUERY_ATTACK_METHODS[i] + \
            #     '_' + args.service + '_' + args.model.replace('/', '_') + '.csv'
            #     content = final_dict_result['results'][QUERY_ATTACK_METHODS[i]]['adv']
            #     pred_label = infer.predict(
            #         content, prompt=PROMPT_SET[name][-1])
            #     res_dict['idx'] = idx
            #     res_dict['content'] = content
            #     res_dict['true_label'] = label
            #     res_dict['pred_label'] = pred_label
            #     lst.append(res_dict)
            #     pd.DataFrame(lst).to_csv(args.save_file, index=False)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
