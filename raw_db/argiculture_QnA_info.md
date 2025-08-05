---
license: apache-2.0
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answers
    dtype: string
  splits:
  - name: agirculture_QnA
    num_bytes: 4798384
    num_examples: 22615
  download_size: 1969746
  dataset_size: 4798384
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- question-answering
tags:
- Agriculture
- agriculture_qa
size_categories:
- 10K<n<100K
---

# Dataset Summary
이 데이터셋은 농업 분야에 관한 질문-답변 페어를 담고 있다. 농작물 재배, 가축 사육, 토양 관리, 농업 기술 등 다양한 주제를 포괄하며, 농업 관련 질문 답변 시스템 구축, 정보 검색, 자연어 이해 연구에 활용할 수 있다.

## Dataset Details

데이터 파일은 **Apache Parquet**(`.parquet`) 형식으로 저장되어 있으며, 각 파일에는 여러 개의 질문-답변 레코드가 들어 있다. 내부적으로 `question` 과 `answers` column을 가진 table 구조이다.

Pandas의 read_parquet메서드를 이용하여 dataframe으로 변경할 수 있음

- **question**: 질문 텍스트  
- **answers** : 답변 텍스트  

### Dataset Description

- **Curated by:** [Mohammed Ashraf](https://huggingface.co/mrSoul7766)  
- **Language(s) (NLP):** English  
- **License:** Apache 2.0  

### Dataset Sources

본 데이터셋은 농업 관련 포럼, 웹사이트, FAQ 등 온라인 자료에서 질문-답변을 수집·선별하여 제작하였다. 데이터 수집 과정에서 질문과 답변의 품질과 적합성을 수작업으로 검증하였다.
