import argparse
import yaml
import os

##################################
##### load config.yaml ###########
##################################
def load_yaml(path='src/config.yaml'):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config():
    default_cfg = load_yaml()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--sbert_model_name", type=str, default=default_cfg.get('sbert_model_name'))
    parser.add_argument("--bert_model_name", type=str, default=default_cfg.get('bert_model_name'))
    parser.add_argument("--embedor_model_name", type=str, default=default_cfg.get('embedor_model_name'))
    
    
    parser.add_argument("--input_file", type=str, default=default_cfg.get('input_file'))
    parser.add_argument("--output_file", type=str, default=default_cfg.get('output_file'))
    parser.add_argument("--task", type=str, default=default_cfg.get('task'))
    parser.add_argument("--ndocs", type=int, default=default_cfg.get('ndocs'))
    parser.add_argument("--max_new_tokens", type=int, default=default_cfg.get('max_new_tokens'))
    parser.add_argument("--threshold", type=float, default=default_cfg.get('threshold'))
    parser.add_argument("--w_rel", type=float, default=default_cfg.get('w_rel'))
    parser.add_argument("--w_sup", type=float, default=default_cfg.get('w_sup'))
    parser.add_argument("--w_use", type=float, default=default_cfg.get('w_use'))


    args = parser.parse_args()
    return args
