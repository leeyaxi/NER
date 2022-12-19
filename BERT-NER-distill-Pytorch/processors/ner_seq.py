""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import time
import copy
import json
from .utils_ner import DataProcessor, parserDenpendencyTree
from nltk.parse.stanford import StanfordDependencyParser
import numpy as np
import os
# java_path = "/opt/homebrew/opt/openjdk/bin/java" #write your java path
# os.environ['JAVAHOME'] = java_path

# stanford_parser_dir = '/Users/fran/Documents/github/NER/BERT-NER-distill-Pytorch/dependency/'
stanford_parser_dir = './dependency/'
eng_model_path = stanford_parser_dir + "stanford-corenlp-4.2.0-models-english.jar"
my_path_to_models_jar = stanford_parser_dir + "stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar"
my_path_to_jar = stanford_parser_dir + "stanford-parser-full-2020-11-17/stanford-parser.jar"

dependency_parser = StanfordDependencyParser(path_to_models_jar=my_path_to_models_jar,
                            path_to_jar=my_path_to_jar)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

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
    def __init__(self, input_ids, input_mask, input_dependency_mask, input_len, decode_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_dependency_mask = input_dependency_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
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
    all_input_ids, all_attention_mask, all_token_type_ids, all_input_denpency_mask, all_lens, all_decode_mask, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_dependency_mask = all_input_denpency_mask[:, :max_len, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    all_deocde_mask = all_decode_mask[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_dependency_mask, all_labels, all_lens, all_deocde_mask

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,input_type="char",
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 50 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        #get dependency path for example
        start_time = time.time()

        dependency_graph = list(dependency_parser.parse(example.text_a))[0]
        pdt = parserDenpendencyTree()
        pdt.find_tree_path(dependency_graph, 0, [])
        end_time = time.time()
        # print ("get and calculate dependency path: {:.2f}second".format(end_time - start_time))

        tokens, label_ids = [], []
        position_mapping_dic = {}
        abs_pos, rel_pos = 0, 0
        for word, label  in zip(example.text_a, example.labels):
            tokenized_word = tokenizer.tokenize(word)
            rel_move = 1
            for token in tokenized_word:
                tokens.append(token)
            label_ids.append(label_map[label])

            for i in range(1, len(tokenized_word)):
                rel_move += 1
                label_ids.append(label_map['X'])
            position_mapping_dic[abs_pos] = [idx for idx in range(rel_pos, rel_pos+rel_move)]
            abs_pos += 1
            rel_pos += rel_move

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        input_dependency_mask = pdt.get_visual_mask(position_mapping_dic, len(tokens), max_seq_length - special_tokens_count)

        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # original mask
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length  # original mask
        input_dependency_mask = np.concatenate([input_dependency_mask, np.zeros((padding_length, input_dependency_mask.shape[1]))], axis=0)
        input_dependency_mask = np.concatenate([input_dependency_mask, np.zeros((input_dependency_mask.shape[0], padding_length))], axis=1)

        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token] * padding_length

        decode_mask = [(x != label_map["X"]) for x in label_ids]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("decoder_mask: %s", " ".join([str(x) for x in decode_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_dependency_mask=input_dependency_mask,
                                      input_len = input_len, decode_mask = decode_mask, segment_ids=segment_ids, label_ids=label_ids))
    return features

class DNRTIBIOProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-HackOrg", "I-HackOrg", "B-OffAct", "I-OffAct", "B-SamFile", "I-SamFile",
                "B-SecTeam", "I-SecTeam", "B-Time", "I-Time", "B-Way", "I-Way", "B-Tool", "I-Tool",
                "B-Idus", "I-Idus", "B-Org", "I-Org", "B-Area", "I-Area", "B-Purp", "I-Purp",
                "B-Exp", "I-Exp", "B-Features", "I-Features", 'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:

                labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class DNRTIBIEOSProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""

        return ["X", "B-HackOrg", "I-HackOrg", "E-HackOrg", "S-HackOrg",
                "B-OffAct", "I-OffAct", "E-OffAct", "S-OffAct",
                "B-SamFile", "I-SamFile", "E-SamFile", "S-SamFile",
                "B-SecTeam", "I-SecTeam", "E-SecTeam", "S-SecTeam",
                "B-Time", "I-Time", "E-Time", "S-Time",
                "B-Way", "I-Way", "E-Way", "S-Way",
                "B-Tool", "I-Tool", "E-Tool", "S-Tool",
                "B-Idus", "I-Idus", "E-Idus", "S-Idus",
                "B-Org", "I-Org", "E-Org", "S-Org",
                "B-Area", "I-Area", "E-Area", "S-Area",
                "B-Purp", "I-Purp", "E-Purp", "S-Purp",
                "B-Exp", "I-Exp", "E-Exp", "S-Exp",
                "B-Features", "I-Features", "E-Features", "S-Features",
                'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIEOS
            labels = []
            for idx, x in enumerate(line['labels']):
                if x.startswith("B-"):
                    if(idx == len(line['labels']) - 1) or (idx+1 < len(line['labels']) and line['labels'][idx+1] != "I-" + x.split("-")[-1]):
                        x = x.replace("B-", "S-")
                elif x.startswith("I-"):
                    if(idx == len(line['labels']) - 1) or (idx+1 < len(line['labels']) and line['labels'][idx+1] != x):
                        x = x.replace("I-", "E-")
                labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    'dnrti_bio':DNRTIBIOProcessor,
    'dnrti_bieos':DNRTIBIEOSProcessor,
}
