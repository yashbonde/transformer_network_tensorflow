"""
ops_utils.py

This file has utilities for operations.

14.09.2019 - @yashbonde
"""

import tensorflow as tf
import argparse
import logging
import json

REQUIRED_PARAM = ['num_layers', 'num_heads', 'embedding_dim', 'vocab_size', 'cntx_len',
    'model_save_path', 'use_inverse_embedding', 'opt', 'lr']

def get_opt(opt_name):
    opts = {
        'sgd': tf.train.GradientDescentOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'adam': tf.train.AdamOptimizer
    }
    return opts[opt_name]


class ModelConfig:
    """
    Custom config handler. It has following schema

    ModelConfig
    |---> description
    |---> config (all command line arguments)
    |---> extern (externally added key value pairs, path to checkpoint and models)

    """
    def __init__(self, description = '', path = None, loading = False):
        self.flag_json = {
            'description': description,
            'config': {},
            'extern': {}
        }
        self.description = description
        self.path = path

        if loading:
            self.load_from_json()
        else:
            self.ap = argparse.ArgumentParser(description = description)

    def add_arg(self, flag, type, default = '', help = '', **kwargs):
        """
        Add CLI argument

        :param flag: name/flag
        :param type: dtype
        :param default: default value if any
        :param help: help string
        :param kwargs: kwargs sent to `arparse.ArgumentParser().add_argument()` method
        """
        self.ap.add_argument(flag, default = default, type = type, help = help, **kwargs)
        self.flag_json['config'][flag[2:]] = None

    def add_value(self, flag, value):
        setattr(self, flag, value)
        self.flag_json['config'][flag] = value

    def add_value_extern(self, flag, value):
        """
        Add simple flag-value argument to configuration, added to `extern` sub object
        """
        self.flag_json['extern'][flag] = value
        setattr(self, flag, value)

    def parse_args(self):
        """
        parse command line args
        """
        self.ap = self.ap.parse_args()

        for flag in self.flag_json['config']:
            val = getattr(self.ap, flag)
            setattr(self, flag, val)
            self.flag_json['config'][flag] = val

        del self.ap # save memory

    def save_json(self):
        """
        Save config json to path
        """
        logging.warning("Saving model configuration at {}".format(self.path))
        with open(self.path, 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(self.flag_json))

    def load_from_json(self):
        """
        load config from json
        """
        logging.warning("Loading model configuration from {}".format(self.path))

        res = json.load(open(self.path))
        self.flag_json = res

        self.description = self.flag_json['description']
        for k, v in self.flag_json['config'].items():
            setattr(self, k, v)
        for k, v in self.flag_json['extern'].items():
            setattr(self, k, v)


