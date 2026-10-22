# -*- coding: utf-8 -*-
"""讀取 configs/*.yaml，所有可調參數都從這裡來，不要在程式裡寫死數字。"""
import yaml


class Config:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, key):
        return self._d[key]

    def get(self, *keys, default=None):
        """多層取值用，例如 cfg.get('model', 'lookback_minutes')"""
        node = self._d
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def feature_columns(self):
        return list(self._d['data']['feature_columns'])

    @property
    def dead_columns(self):
        return list(self._d['data'].get('dead_columns') or [])

    @property
    def datetime_col(self):
        return self._d['data'].get('datetime_col', 'DateTime')

    @property
    def fault_col(self):
        return self._d['data'].get('fault_col', 'GlobalFault')

    @property
    def input_csv(self):
        return self._d['data']['input_csv']

    @property
    def lookback(self):
        return int(self._d['model']['lookback_minutes'])

    @property
    def horizon(self):
        return int(self._d['model']['horizon_minutes'])

    @property
    def block_len(self):
        return self.lookback + self.horizon

    @property
    def hidden_dim(self):
        return int(self._d['model']['hidden_dim'])

    @property
    def num_layers(self):
        return int(self._d['model']['num_layers'])

    @property
    def test_fraction(self):
        return float(self._d['split']['test_fraction'])

    @property
    def random_seed(self):
        return int(self._d['split']['random_seed'])

    @property
    def epochs(self):
        return int(self._d['training']['epochs'])

    @property
    def batch_size(self):
        return int(self._d['training']['batch_size'])

    @property
    def learning_rate(self):
        return float(self._d['training']['learning_rate'])

    @property
    def normal_sample_cap(self):
        v = self._d['training'].get('normal_sample_cap')
        return int(v) if v is not None else None

    @property
    def output_dir(self):
        return self._d['output']['output_dir']

    @property
    def threshold_percentiles(self):
        return list(self._d.get('evaluation', {}).get('threshold_percentiles', [90, 95, 99]))


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    return Config(d)
