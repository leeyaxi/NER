import csv
import json
import torch
import unicodedata
import numpy as np
from itertools import chain
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from senticnet.babelsenticnet import BabelSenticNet
SN = BabelSenticNet("cn")

from transformers import BertTokenizer

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_ap_labels(self):
        """Gets the list of aspect labels for this data set."""
        raise NotImplementedError()

    def get_ap_polarity(self):
        """Gets the list of aspect polarities for this data set."""
        raise NotImplementedError()

    def get_POS_tag(self):
        """Gets the list of POS tag for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            ap_classes = []
            ap_polarities = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"ap_classes":ap_classes, "ap_polarities": ap_polarities})
                        words = []
                        ap_classes = []
                        ap_polarities = []
                else:
                    splits = line.strip().split(" ")
                    words.append(splits[0])
                    if len(splits) == 3:
                        ap_classes.append(splits[1].replace("\n", ""))
                        ap_polarities.append(splits[2].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        ap_classes.append("O")
                        ap_polarities.append("-100")
            if words:
                lines.append({"words":words,"ap_classes":ap_classes, "ap_polarities": ap_polarities})
        return lines

    @classmethod
    def _read_json(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label',None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key,value in label_entities.items():
                        for sub_name,sub_index in value.items():
                            for start_index,end_index in sub_index:
                                assert  ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines



class EmotionEntityLib():
    """
    大连理工大学情感词汇本体库解析
    """
    def __init__(self):
        self.emotion_file = "./dependency/DLUT_emotionontology.csv"
        self.tail_size = None
        self.relation_size = None
        self.word_max_len = None
        self.emotion_entity_dic, self.emotion_idx_entity_dic = self.read_file()
        self.head_size = len(self.emotion_entity_dic)
    def read_file(self):
        emotion_data = pd.read_csv(self.emotion_file, sep=", ")

        #对tail实体以及relation标签映射
        for feat_name in ["情感分类", "强度"]:
            lbe = LabelEncoder()
            emotion_data[feat_name] = lbe.fit_transform(emotion_data[feat_name])
            emotion_data[feat_name] = emotion_data[feat_name].apply(lambda x: x+1) #把零值作为padding

        self.tail_size = int(emotion_data["情感分类"].max()) + 1
        self.relation_size = int(emotion_data["强度"].max()) + 1
        self.word_max_len = int(emotion_data["词语"].apply(lambda x: len(x)).max())
        head = emotion_data["词语"].values.tolist()
        tail = emotion_data["情感分类"].values.tolist()
        relation = emotion_data["强度"].values.tolist()
        emotion_entity_dic = dict(zip(head, zip(tail, relation)))
        emotion_idx_entity_idc = dict(zip([i for i in range(len(head))], head))

        return emotion_entity_dic, emotion_idx_entity_idc

    def get_entity_sample(self, text_a_cut, tokenizer, pad_token, max_sample_size=5):
        postive_sample = []
        negtive_sample = []
        for word in text_a_cut:
            if word in self.emotion_entity_dic:
                tail, relation = self.emotion_entity_dic[word]
                word_id = tokenizer(word, add_special_tokens=False).input_ids
                word_id += [pad_token] * (self.word_max_len - len(word_id))
                word_id += [tail]
                word_id += [relation]
                postive_sample.append(word_id)

        #对正样本进行截断
        if len(postive_sample) > max_sample_size:
            idx = [i for i in range(len(postive_sample))]
            np.random.shuffle(idx)
            idx = idx[:max_sample_size]
            postive_sample = [postive_sample[i] for i in idx]

        if len(postive_sample) > 0:
            head_or_tail = np.random.randint(low=0, high=2, size=len(postive_sample))
            negtive_sample = [None for i in range(len(postive_sample))]
            random_entities = np.random.randint(low=1, high=self.head_size, size=len(postive_sample))
            random_tail = np.random.randint(low=1, high=self.tail_size, size=len(postive_sample))

            for idx, sample in enumerate(postive_sample):
                if head_or_tail[idx] == 1:
                    break_head = self.emotion_idx_entity_dic[random_entities[idx]]
                    word_id = tokenizer(break_head, add_special_tokens=False).input_ids
                    word_id += [pad_token] * (self.word_max_len - len(word_id))
                    negtive_sample[idx] = word_id + [sample[-2]] + [sample[-1]]
                else:

                    negtive_sample[idx] = sample[:-2] + [random_tail[idx]] + [sample[-1]]

        #对样本进行补齐
        if len(postive_sample) < max_sample_size:
            postive_sample += [[pad_token] * self.word_max_len + [0, 0]] * (max_sample_size - len(postive_sample))
            negtive_sample += [[pad_token] * self.word_max_len + [0, 0]] * (max_sample_size - len(negtive_sample))
        if len(postive_sample[0]) != 17:
            print (postive_sample)
        return postive_sample+negtive_sample







def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def word_pos_alignment(cut_token_ls, origin_token_ls):
        cur_pos = 0
        origin_pos = 0
        last_len = 0
        cut_pos_align_dic = {}
        while cur_pos < len(cut_token_ls) and origin_pos < len(origin_token_ls):
            cur_cut_word = cut_token_ls[cur_pos]
            cur_origin_char = origin_token_ls[origin_pos]

            if len(cur_cut_word) > len(cur_origin_char) + last_len:
                last_len += len(cur_origin_char)
                if cur_pos not in cut_pos_align_dic:
                    cut_pos_align_dic[cur_pos] = [origin_pos]
                else:
                    cut_pos_align_dic[cur_pos].append(origin_pos)
                origin_pos += 1
            else:
                last_len = 0
                if cur_pos not in cut_pos_align_dic:
                    cut_pos_align_dic[cur_pos] = [origin_pos]
                else:
                    cut_pos_align_dic[cur_pos].append(origin_pos)
                cur_pos += 1
                origin_pos += 1
        return cut_pos_align_dic

class MultiFusionAdjacencyGraph():
    def __init__(self, cut_token_ls, origin_token_ls):
        self.word_path = []
        self.char_path = []
        self.cut_pos_align_dic = word_pos_alignment(cut_token_ls, origin_token_ls)
        self.invalid_POS = self.get_invalid_POS()

    def get_invalid_POS(self):
        return ["DT", "MSP"]

    def find_tree_path(self, dependency_graph, idx, path):
        # path.append((idx, dependency_graph.nodes[idx]["word"]))
        path.append(idx-1)
        if len(dependency_graph.nodes[idx]["deps"]) == 0:
            self.word_path.append([item for item in path[1:]])
            return
        next_idx_ls = []
        for child_idx in dependency_graph.nodes[idx]["deps"].values():
            next_idx_ls.extend(child_idx)
        for next_idx in next_idx_ls:
            self.find_tree_path(dependency_graph, next_idx, path)
            path.pop(-1)
        return

    def mapping_word_pos_to_char(self):
        if len(self.word_path) == []:
            return
        for word_path in self.word_path:
            char_path = []
            for path in word_path:
                char_path.extend(self.cut_pos_align_dic[path])
            self.char_path.append(char_path)
        return

    def get_word_path(self):
        return self.word_path

    def get_char_path(self):
        return self.char_path

    def get_word_polarity(self, text_a_cut, ap_class_ids, ap_class_map, position_mapping_dic, seq_len):
        text_a_polarity = np.zeros((seq_len, ))
        for word_idx, word in enumerate(text_a_cut):
            try:
                polarity_value = SN.polarity_value(word)
            except KeyError:
                polarity_value = 0.0
            for abs_char_idx in self.cut_pos_align_dic[word_idx]:
                for rel_char_idx in position_mapping_dic[abs_char_idx]:
                    text_a_polarity[rel_char_idx] = polarity_value

        # Wij的情感分数=Wi的情感分数+Wj的情感分数(Sij) + Cij
        polarity_score = np.tile(np.expand_dims(text_a_polarity, 1), (1, seq_len)) + np.tile(np.transpose(np.expand_dims(text_a_polarity, 1)), (seq_len, 1))
        ap_flag = [idx for idx in range(text_a_polarity.shape[0]) if ap_class_map["O"] != ap_class_ids[idx]] #方面词
        emotion_flag = [idx for idx in range(text_a_polarity.shape[0]) if text_a_polarity[idx] != 0.0] #情感词
        for idx in range(len(emotion_flag)):
            polarity_score[idx, ap_flag] += 1
        for idx in range(len(ap_flag)):
            polarity_score[idx, emotion_flag] += 1
        return polarity_score


    def get_visual_mask(self, text_a_cut, text_a_pos, ap_class_ids, ap_class_map, position_mapping_dic, seq_len, max_seq_length):
        visual_mask = np.zeros((seq_len, seq_len))
        for abs_path in self.word_path:
            rel_path = [position_mapping_dic[abs_pos] for abs_pos in abs_path]
            relavance_path = list(chain(*rel_path))
            for node in relavance_path:
                visual_mask[node, relavance_path] = 1

        #添加word的情感分数Aij = Rij * (Sij+Cij)
        text_a_polarity_matrix = self.get_word_polarity(text_a_cut, ap_class_ids, ap_class_map, position_mapping_dic, seq_len)
        visual_mask = visual_mask * text_a_polarity_matrix


        #剪枝Aij = Rij * (Sij+Cij) * Pij
        invalid_POS_position = self.pruning_with_POS(text_a_pos)
        if len(invalid_POS_position) > 0:
            visual_mask[invalid_POS_position, :] = 0
            visual_mask[:, invalid_POS_position] = 0

        #自循环约束-对角线元素非0
        for i in range(visual_mask.shape[0]):
            visual_mask[i][i] = 1

        if seq_len > max_seq_length:
            visual_mask = visual_mask[:max_seq_length, :max_seq_length]

        #add cls and sep
        visual_mask = np.concatenate([np.ones((1, visual_mask.shape[1])), visual_mask], axis=0)
        visual_mask = np.concatenate([visual_mask, np.ones((1, visual_mask.shape[1]))], axis=0)
        visual_mask = np.concatenate([np.ones((visual_mask.shape[0], 1)), visual_mask], axis=1)
        visual_mask = np.concatenate([visual_mask, np.ones((visual_mask.shape[0], 1))], axis=1)

        return visual_mask

    def pruning_with_POS(self, text_a_POS):
        invalid_POS_position = []
        if len(self.invalid_POS) == 0:
            return invalid_POS_position
        for word_idx, word_POS in enumerate(text_a_POS):
            if word_POS in self.invalid_POS:
                invalid_POS_position.extend(self.cut_pos_align_dic[word_idx])
        return invalid_POS_position

    def get_POS_input(self, text_a_POS, POS_tag_map, position_mapping_dic, seq_len):
        token_POS = [0 for i in range(seq_len)]
        for word_idx, word_POS in enumerate(text_a_POS):
            for abs_char_idx in self.cut_pos_align_dic[word_idx]:
                for rel_char_idx in position_mapping_dic[abs_char_idx]:
                    token_POS[rel_char_idx] = POS_tag_map[word_POS]
        return token_POS

def cal_pos_weight_from_ap(ap_class_ids, ap_class_map):
    n = len(ap_class_ids)
    ap_class_flag = [idx for idx in range(n) if ap_class_map["O"] != ap_class_ids[idx]]

    if len(ap_class_flag) == 0:
        return [1 for idx in range(n)]

    weight_lf = [0.0 for idx in range(n)]
    last = 0
    for flag_t in ap_class_flag:
        for i in range(last, flag_t):
            weight_lf[i] = 1 - (flag_t-i) / n
        last = flag_t + 1
    weight_rg = [0.0 for idx in range(n)]
    last = n
    for flag_t in ap_class_flag[::-1]:
        for i in range(flag_t+1, last):
            weight_rg[i] = 1 - (i - flag_t) / n
        last = flag_t
    return [sum(item) for item in zip(weight_lf, weight_rg)]
