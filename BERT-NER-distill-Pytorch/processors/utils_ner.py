import csv
import json
import torch
import unicodedata
import numpy as np
from itertools import chain
from transformers import BertTokenizer

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
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
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.strip().split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].replace("\n", "")
                        labels.append(label)
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
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

def get_conll_out_format(original_text, tokens, preds, id2label):
    line = []
    last_word, last_pred = "", ""
    pos = 0
    tag_flag = True
    unk_flag = False
    for token, pred in zip(tokens, preds):
        token = token.replace("##", "")
        last_word += token
        if unk_flag:
            if pos + 1 < len(original_text) and original_text[pos+1].startswith(last_word):
                unk_flag = False
                pos += 1
            elif not original_text[pos].endswith(last_word):
                continue
            else:
                last_word, last_pred = "", ""
                pos += 1
                unk_flag = False
                continue
        if tag_flag:
            last_pred = id2label[pred]
            last_pred = "O" if last_pred == "X" or last_pred == "[END]" or last_pred == "[START]" else last_pred
        if token == "[UNK]":
            if len(original_text[pos]) != 1:
                last_word = ""
                unk_flag = True
            else:
                last_word, last_pred = "", ""
                tag_flag = True
                pos += 1
            line.append([original_text[pos], last_pred])
            continue

        if original_text[pos].lower() != last_word and unicodedata.normalize('NFKD', original_text[pos].lower()).encode('ascii','ignore') != last_word.encode('ascii','ignore'):
            tag_flag = False
            continue
        else:
            tag_flag = True
            line.append([original_text[pos], last_pred])
            last_word, last_pred = "", ""
            pos += 1
    if len(line) != len(original_text):
        print (tokens, line, original_text)
    return line


# def get_conll_out_format(tokens, preds, id2label):
#     line = []
#     last_word, last_pred = "", ""
#     token_flag = False
#     for token, pred in zip(tokens, preds):
#         pred = id2label[pred]
#         pred = "O" if pred == "X" else pred
#         if token.startswith("##"):
#             token_flag = True
#             last_word += token.replace("##", "")
#             continue
#         if token_flag:
#             line[-1][0] = last_word
#             token_flag = False
#         last_word = token
#         last_pred = pred
#         line.append([last_word, last_pred])
#     return line


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return

class parserDenpendencyTree():
    def __init__(self):
        self.total_path = []
    def find_tree_path(self, dependency_graph, idx, path):
        # path.append((idx, dependency_graph.nodes[idx]["word"]))
        path.append(idx-1)
        if len(dependency_graph.nodes[idx]["deps"]) == 0:
            self.total_path.append([item for item in path[1:]])
            return
        next_idx_ls = []
        for child_idx in dependency_graph.nodes[idx]["deps"].values():
            next_idx_ls.extend(child_idx)
        for next_idx in next_idx_ls:
            self.find_tree_path(dependency_graph, next_idx, path)
            path.pop(-1)
        return

    def get_total_path(self):
        return self.total_path

    def get_visual_mask(self, position_mapping_dic, seq_len, max_seq_length):
        visual_mask = np.zeros((seq_len, seq_len))
        for abs_path in self.total_path:
            rel_path = [position_mapping_dic[abs_pos] for abs_pos in abs_path]
            # relavance_path = list(chain(*rel_path))
            # for node in relavance_path:
            #     visual_mask[node, relavance_path] = 1
            for idx, relnode_pos_ls in enumerate(rel_path):
                for relnode_pos in relnode_pos_ls:
                    visual_mask[relnode_pos, list(chain(*rel_path[idx:]))] = 1
        if seq_len > max_seq_length:
            visual_mask = visual_mask[:max_seq_length, :max_seq_length]

        #add cls and sep
        visual_mask = np.concatenate([np.ones((1, visual_mask.shape[1])), visual_mask], axis=0)
        visual_mask = np.concatenate([visual_mask, np.ones((1, visual_mask.shape[1]))], axis=0)
        visual_mask = np.concatenate([np.ones((visual_mask.shape[0], 1)), visual_mask], axis=1)
        visual_mask = np.concatenate([visual_mask, np.ones((visual_mask.shape[0], 1))], axis=1)

        return visual_mask
