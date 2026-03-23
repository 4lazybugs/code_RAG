import os
import yaml
import argparse
from types import SimpleNamespace

def load_yaml(path):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config(path):
    default_cfg = load_yaml(path=path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--temperature", type=float, default=default_cfg.get('temperature', 0.0))
    parser.add_argument("--k_each", type=int, default=default_cfg.get('k_each', 10))
    parser.add_argument("--top_k", type=int, default=default_cfg.get('top_k', 5))
    parser.add_argument("--embedor_model_name", type=str, default=default_cfg.get('embedor_model_name'))

    args, _ = parser.parse_known_args()

    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)