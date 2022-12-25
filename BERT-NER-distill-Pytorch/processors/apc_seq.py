""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import time
import copy
import json
import jieba
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
import numpy as np
import os

from processors.utils_ner import MultiFusionAdjacencyGraph, DataProcessor

# java_path = "/opt/homebrew/opt/openjdk/bin/java" #write your java path
# os.environ['JAVAHOME'] = java_path

# stanford_parser_dir = '/Users/fran/Documents/github/NER/BERT-NER-distill-Pytorch/dependency/'
stanford_parser_dir = './dependency/'

chi_tagger = StanfordPOSTagger(path_to_jar=stanford_parser_dir+"stanford-postagger-full-2020-11-17/stanford-postagger.jar",
                               model_filename=stanford_parser_dir+"stanford-postagger-full-2020-11-17/models/chinese-distsim.tagger")


logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, text_a_cut, text_a_pos, ap_classes):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_a_cut = text_a_cut
        self.text_a_pos = text_a_pos
        self.ap_classes = ap_classes

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_POS_ids, input_len,
                 decode_mask, segment_ids,  ap_class_ids):
        self.input_ids = input_ids
        self.input_POS_ids = input_POS_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.ap_class_ids = ap_class_ids
        self.input_len = input_len
        self.decode_mask = decode_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, \
        all_lens, all_POS_ids, all_decode_mask, all_ap_class_ids, = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_POS_ids = all_POS_ids[:, :max_len]
    all_ap_class_ids = all_ap_class_ids[:,:max_len]
    all_deocde_mask = all_decode_mask[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids,  \
        all_POS_ids, all_ap_class_ids, all_deocde_mask

def convert_examples_to_features(examples, ap_class_list, POS_tag_list,
                                 max_seq_length,tokenizer,input_type="char", cls_token_at_end=False,
                                 cls_token="[CLS]",cls_token_segment_id=1, sep_token="[SEP]",
                                 pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    pad_token_label_id = -100
    ap_class_map = {label: i for i, label in enumerate(ap_class_list)}
    POS_tag_map = {label: i+1 for i, label in enumerate(POS_tag_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 50 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        mfag = MultiFusionAdjacencyGraph(example.text_a_cut, example.text_a)
        tokens, ap_class_ids = [], []
        position_mapping_dic = {}
        abs_pos, rel_pos = 0, 0
        for word, ap_class  in zip(example.text_a, example.ap_classes):
            tokenized_word = tokenizer.tokenize(word)
            rel_move = 1
            for token in tokenized_word:
                tokens.append(token)
            ap_class_ids.append(ap_class_map[ap_class])

            for i in range(1, len(tokenized_word)):
                rel_move += 1
                ap_class_ids.append(ap_class_map["X"])
            position_mapping_dic[abs_pos] = [idx for idx in range(rel_pos, rel_pos+rel_move)]
            abs_pos += 1
            rel_pos += rel_move

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2

        input_POS_ids = mfag.get_POS_input(example.text_a_pos, POS_tag_map, position_mapping_dic, len(tokens))
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            ap_class_ids = ap_class_ids[: (max_seq_length - special_tokens_count)]
            input_POS_ids = input_POS_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        ap_class_ids += [pad_token_label_id]

        input_POS_ids += [pad_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            ap_class_ids += [pad_token_label_id]
            input_POS_ids += [pad_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            ap_class_ids = [pad_token_label_id] + ap_class_ids
            input_POS_ids = [pad_token] + input_POS_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # original mask
        input_len = len(ap_class_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length  # original mask
        input_POS_ids += [pad_token] * padding_length

        segment_ids += [pad_token_segment_id] * padding_length
        ap_class_ids += [pad_token_label_id] * padding_length

        decode_mask = [(x != pad_token_label_id and x!= ap_class_map["X"]) for x in ap_class_ids]
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_POS_ids: %s", " ".join([str(x) for x in input_POS_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("decoder_mask: %s", " ".join([str(x) for x in decode_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_POS_ids=input_POS_ids,
                                      input_len = input_len, decode_mask=decode_mask, segment_ids=segment_ids, ap_class_ids=ap_class_ids))
    return features

class MoocProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "mooc.train.txt.atepc")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "mooc.dev.txt.atepc")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "mooc.test.txt.atepc")), "test")

    def get_ap_labels(self):
        """See base class."""
        return ["X", "B-ASP", "I-ASP", "O","[START]", "[END]"]

    def get_ap_polarity(self):
        """See base class."""
        return ["Positive", "Negative"]

    def get_POS_tag(self):
        """词性标签"""
        return ["AD", "AS", "BA", "CC", "CD", "CS", "DEC", "DEG",
                "DER", "DEV", "DT", "ETC", "FW", "IJ", "JJ", "LB",
                "LC", "M", "MSP", "NN", "NR", "NT", "OD", "ON",
                "P", "PN", "PU", "SB", "SP", "VA", "VC", "VE", "VV"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            text_a = line['words']
            text_a_cut = list(jieba.cut("".join(text_a)))  # 分词结果
            text_a_pos = [item[-1].split("#")[-1] for item in chi_tagger.tag(text_a_cut)]  # 词性结果

            # BIOS
            ap_classes = []
            guid = "%s-%s" % (set_type, i)
            for class_idx, (x, y) in enumerate(zip(line['ap_classes'], line['ap_polarities'])):
                ap_classes.append(x)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_cut=text_a_cut, text_a_pos=text_a_pos,
                             ap_classes=ap_classes))
        return examples

class LaptopProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.iob")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.iob")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.iob")), "test")

    def get_ap_labels(self):
        """See base class."""
        return ["X", "B-A", "I-A", "O","[START]", "[END]"]

    def get_ap_polarity(self):
        """See base class."""
        return ["Positive", "Negative"]

    def get_POS_tag(self):
        """词性标签"""
        return ["PRP", "VBP", "IN", "NN", "CC", "VBG", "DT",
                "JJ", ".", "VBD", "NNP", ":", "CD", "VBN", "RB",
                "NNS", "VBZ", "VB", "TO", "PRP$", ",", "WDT", "RP",
                "FW", "JJR", "MD", "(", ")", "UH", "EX", "JJS", "WRB",
                "WP", "PDT", "RBS", "RBR", '"', "POS", "$", "#", "O",
                "LS", "SYM"]

    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            ap_classes = []
            ap_POS = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"ap_POS":ap_POS, "ap_classes": ap_classes})
                        words = []
                        ap_classes = []
                        ap_polarities = []
                else:
                    splits = line.strip().split("\t")
                    words.append(splits[0])
                    ap_POS.append(splits[1].replace("\n", ""))
                    ap_classes.append(splits[2].replace("\n", ""))
            if words:
                lines.append({"words":words,"ap_POS":ap_POS, "ap_classes": ap_classes})
        return lines
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            text_a = line['words']
            # BIOS
            ap_classes = []
            text_a_pos = []
            guid = "%s-%s" % (set_type, i)
            for class_idx, (x, y) in enumerate(zip(line['ap_POS'], line['ap_classes'])):
                text_a_pos.append(x)
                ap_classes.append(y)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_cut=text_a, text_a_pos=text_a_pos,
                             ap_classes=ap_classes))
        return examples


ner_processors = {
    'mooc':MoocProcessor,
    'laptop': LaptopProcessor
}
