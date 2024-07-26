# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

#from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
import jieba
import io

from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# jieba.load_userdict('Datasets/vocab_lunyu.txt')


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with io.open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BertTokenizer.
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
    """

    def __init__(self, vocab_file, **kwargs):
        """Constructs a BertTokenizer.
        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
        """
        super(BertTokenizer, self).__init__(unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                                            mask_token="[MASK]", **kwargs)
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        # let jieba use user defined dictionary
        jieba.load_userdict(vocab_file)


    @property
    def vocab_size(self):
        return len(self.vocab)


    # def _tokenize(self, text):
    #     split_tokens = []
    #     print("_tokenize start")
        
    #     for token in self.tokenize(text):
    #         #print(token)
    #         if self._convert_token_to_id(token) == 4: # [UNK] == 4
    #             #print("token == [UNK]")
    #             #print(token)
    #             jieba.del_word(token)
    #             sub_token = jieba.cut(token)
    #             for sub in sub_token:
    #                 split_tokens.append(sub)
    #         else:
    #             split_tokens.append(token)

    #     print(split_tokens)
    #     print("_tokenize end")
    #     return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        vocab_file = vocab_path
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, '/vocab-file.txt')
        with io.open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """ Instantiate a BertTokenizer from pre-trained vocabulary files.
        """
        return super(BertTokenizer, cls)._from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)


    def tokenize(self, text):
        """ Basic Tokenization of a piece of text.
        """
        text = self._clean_text(text)

        text = self._tokenize_chinese_chars_jieba(text)
        print("_tokenize_chinese_chars_jieba():")
        print(text)
        orig_tokens = whitespace_tokenize(text)
        print("whitespace_tokenize():")
        print(orig_tokens)

        split_tokens = []
        for token in orig_tokens:
            #print(token)
            if self._convert_token_to_id(token) == 4: # [UNK] == 4
                #print("token == [UNK]")
                #print(token)
                jieba.del_word(token)
                sub_token = jieba.cut(token)
                for sub in sub_token:
                    split_tokens.append(sub)
            else:
                split_tokens.append(token)

        
        return split_tokens


    def _tokenize_chinese_chars_jieba(self, text):
        t = []
        for item in jieba.cut(text):
            t.append(item)

        return " ".join(t)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False


    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
