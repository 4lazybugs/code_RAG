from pathlib import Path
import json
import random
import os
import yaml
import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.router.train import (
    TextDataset,
    FrozenBertClassifier,
    train_one_epoch,
    evaluate,
    get_predictions,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
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
    parser.add_argument("--model_name", type=str, default=default_cfg["model"]["name"])
    args, _ = parser.parse_known_args()

    default_cfg["model"]["name"] = args.model_name

    return SimpleNamespace(**default_cfg)


# json에 retrieval_needed key를 추가하는 함수
def add_retrieval_needed(json_file, value=1):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            item["retrieval_needed"] = value
    elif isinstance(data, dict):
        data["retrieval_needed"] = value
    else:
        print(f"[WARN] 지원하지 않는 JSON 구조 skip: {json_file}")
        return

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_all_data_from_jsons(input_dir: Path):
    all_texts = []
    all_labels = []

    json_files = list(input_dir.rglob("*.json"))
    print(f"[INFO] 발견된 json 파일 수: {len(json_files)}")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            print(f"[WARN] 지원하지 않는 JSON 구조 skip: {json_file}")
            continue

        valid_count = 0
        skip_no_question = 0
        skip_no_label = 0
        skip_empty_question = 0
        skip_bad_label = 0

        for item in items:
            if "question" not in item:
                skip_no_question += 1
                continue

            if "retrieval_needed" not in item:
                skip_no_label += 1
                continue

            question = str(item["question"]).strip()

            try:
                label = int(item["retrieval_needed"])
            except (ValueError, TypeError):
                skip_bad_label += 1
                continue

            if question == "":
                skip_empty_question += 1
                continue

            if label not in (0, 1):
                skip_bad_label += 1
                continue

            all_texts.append(question)
            all_labels.append(label)
            valid_count += 1

        print(
            f"[INFO] {json_file} -> "
            f"used={valid_count}, "
            f"no_question={skip_no_question}, "
            f"no_label={skip_no_label}, "
            f"empty_question={skip_empty_question}, "
            f"bad_label={skip_bad_label}"
        )

    return all_texts, all_labels


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    CFG = get_config("configs/config_route.yaml")

    set_seed(seed=35)

    '''
    # 필요 시 retrieval_needed 일괄 추가
    input_dir = Path("db/qa_data/qa_in_use/without_md")
    for json_file in input_dir.rglob("*.json"):
        add_retrieval_needed(json_file, value=0)
    '''

    # ----------------------------------
    # 2) 전체 json에서 데이터 모으기
    # ----------------------------------
    input_dir = Path("db/qa_data/qa_in_use/")
    texts, labels = load_all_data_from_jsons(input_dir)

    num_label_0 = sum(1 for x in labels if x == 0)
    num_label_1 = sum(1 for x in labels if x == 1)

    print(f"[INFO] 전체 샘플 수: {len(texts)}")
    print(f"[INFO] label 0 개수: {num_label_0}")
    print(f"[INFO] label 1 개수: {num_label_1}")

    if len(texts) == 0:
        raise ValueError("학습할 데이터가 없습니다.")

    if len(set(labels)) < 2:
        raise ValueError("label이 한 종류뿐입니다. binary classification 학습이 불가능합니다.")

    # ----------------------------------
    # 3) train / valid split
    # ----------------------------------
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts,
        labels,
        test_size=CFG.data["test_size"],
        random_state=CFG.data["random_state"],
        stratify=labels,
    )

    print(f"[INFO] train 샘플 수: {len(train_texts)}")
    print(f"[INFO] valid 샘플 수: {len(valid_texts)}")

    # ----------------------------------
    # 4) tokenizer / dataset / dataloader
    # ----------------------------------
    tokenizer = AutoTokenizer.from_pretrained(CFG.model["name"])

    train_dataset = TextDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=CFG.training["max_length"],
    )
    valid_dataset = TextDataset(
        texts=valid_texts,
        labels=valid_labels,
        tokenizer=tokenizer,
        max_length=CFG.training["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.training["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.training["batch_size"],
        shuffle=False,
    )

    # ----------------------------------
    # 5) device / model / optimizer / scheduler
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = FrozenBertClassifier(
        model_name=CFG.model["name"],
        num_labels=CFG.model["num_labels"],
        hidden_dim=CFG.model["hidden_dim"],
        dropout=CFG.model["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CFG.training["lr"],
    )

    total_steps = len(train_loader) * CFG.training["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(CFG.training["warmup_ratio"] * total_steps)),
        num_training_steps=total_steps,
    )

    # ----------------------------------
    # 6) training loop
    # ----------------------------------
    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(CFG.training["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"[Epoch {epoch+1}/{CFG.training['epochs']}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    # ----------------------------------
    # 7) final result + confusion matrix
    # ----------------------------------
    print("\n[RESULT]")
    print(f"best_val_acc = {best_val_acc:.4f}")
    print(f"best_val_loss = {best_val_loss:.4f}")

    y_true, y_pred = get_predictions(model, valid_loader, device)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n[CONFUSION MATRIX]")
    print("          Pred 0    Pred 1")
    print(f"True 0    {cm[0, 0]:6d}    {cm[0, 1]:6d}")
    print(f"True 1    {cm[1, 0]:6d}    {cm[1, 1]:6d}")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, digits=4))
