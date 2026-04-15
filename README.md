
---

# Agricultural RAG Chatbot (Demo)

농업 도메인 전용 Retrieval-Augmented Generation(RAG) 기반 LLM 챗봇 데모 시스템이다.
로컬 LLM을 vLLM 서버로 서빙하여 대규모 추론을 수행한 뒤, 평가까지 진행한다.

---

# Scope

* 문서 자동 추출(ingest) 및 벡터 db화 파이프라인 구현
* RAG 기반 LLM 챗봇 구현
* vLLM 서버 기반 로컬 추론 구조
* Retrieval 및 Inference 모듈 분리 설계
* Retrieval에 lexical db, semantic db 기반 두가지 전략 구현 
* RAG 대규모 자동화 및 성능 평가 파이프라인 구현 완료

---

# Limitations

* RAG 성능평가를 위한 QA생성 코드 미구현 (추후 제공 예정)
* 파인튜닝 및 학습 모듈 미포함

---

# System Overview

시스템은 문서 전처리, RAG 기반 질의응답, 대규모 추론, 평가의 네 단계 파이프라인으로 구성된다.

- **전처리 단계**에서는 PDF 문서로부터 OCR을 통해 텍스트를 추출하고 Markdown 형식으로 변환한 뒤, 임베딩을 생성하여 **Vector DB (Chroma)** 에 저장한다.  
- **RAG 추론 단계**에서는 사용자 질의를 임베딩하고 Vector DB에서 관련 문서를 검색한 뒤, 검색된 문맥을 기반으로 **vLLM 서버의 LLM** 이 답변을 생성한다.  
- **대규모 추론 단계**에서는 QA 데이터셋(JSON)을 입력으로 받아 QA 유형별 Prompt/Input/Output 조합을 구성하고, **LLM / RAG / IterRAG 에이전트**를 선택적으로 사용하여 일괄 추론을 수행한 후 결과를 JSON으로 저장한다.  
- **평가 단계**에서는 정답 데이터와 추

## 처리 흐름

```
(1) 전처리
PDF
  → OCR
  → md 
  → Embedder
  → Vector DB(Chroma)


(2) RAG - single inference
Query
  → Embedder
  → Vector Retrieval
  → LLM (vLLM Server)
  → Answer


(3) RAG - multiple inference for evaluation
merged JSON
  → QA Type Build (Prompt / Input / Output 조합)
  → Agent Selection (LLM / RAG / IterRAG)
  → Retriever (BM25 or Vector Retrieval, optional)
  → LLM (vLLM Server)
  → Generated Answer
  → Save as JSON 


(4) Extensive evaluation
Reference QA JSON + Inference Result JSON
  → Load Answers / Retrieved Docs
  → Build Evaluation Batches
  → Retrieval Metrics (Recall, MRR)
  → Generation Metrics (EM, ROUGE, BERT, SBERT, BLEURT)
  → Score Aggregation
  → Save JSON / Excel Summary
```

---

# Project Structure

```
┌── db/                
│   ├── raw_db/     # 전처리 되지 않은 원시 데이터들
│   │   ├── db_in_use/  # 농촌진흥청에서 제공한 원시 PDF 문서들; 전처리 대상
│   │   └── test_db/    # 테스트용으로 db_in_use에서 몇 페이지를 sample로 뽑았음
│   └─── vector_db/     # 텅 비어있음; pipe_extract -> pipe_vectorize 하면 채워질 것        
│
│
├── scripts/      # 스크립트 패키지: src(소스코드)를 import하여 데이터 전처리, 모델 실행, 평가 등 핵심 기능을 호출하고 전체 실행
│   ├── run_chatbot.py  # 챗봇, 일회성 추론; src/infer/base.py의 answer_once가 쓰임
│   ├── pipe_eval.py  # !대규모 평가
│   ├── pipe_infer.py  # !대규모 추론; src/infer/base.py의 answer_all(answer_once를 반복호출한 함수)가 쓰임
│   ├── pipe_extract.py  # pdf -> markdown
│   └── pipe_vectorize.py  #  markdown -> database
│
│
├── src/                # 프로젝트 기능 구현을 위한 주요 소스 코드 패키지; scripts 패키지에서 import할 모듈들의 모음
│   ├── preprocess/     # 전처리 패키지: pipe_extract 및 pipe_vectorize에 필요한 모듈들이 들어있음        
│   │   ├── extract.py  # PaddleOCR, Docling의 전처러 라이브러리들을 통한 데이터 추출 함수 구현되어 있음
│   │   └── embedding.py # Embeddor 클래스, md->vector 만드는 함수 구현 되어있음 
│   │
│   ├── eval/          # 평가 패키지: 다양한 평가 metric 모듈들이 들어있음, Strategy pattern으로 구현
│   │                    
│   │
│   ├── infer/          # 추론 패키지: pipe_infer, run_chatbot에 필요한 모듈들이 들어있음
│   │   ├── agents/     # 에이전트 패키지: 반복 RAG, 단순 LLM 호출, 단순 RAG 등 다양한 agent를 구현한 모듈들이 들어있음   
│   │   └── qa_type/     # 질의유형 패키지: Builder 패턴을 활용해 다양한 질의 유형에 유연하게 대응하는 에이전트를 구성함.
│   │
│   │   
│   └── retrieval/      # Retriever: embedding 모델은 preprocess/emebdding.py에서 정의된 모델을 사용함
│        ├── retriever.py  # build_retriever함수를 통해 vector_db에 있는 db하나당 retriever 한개가 붙음
│        │                  이후 Multi_Retriever클래스가 이들을 통합해 top k개를 뽑음         
│        └── qa_type/     # 질의유형 패키지: Builder 패턴을 활용해 다양한 질의 유형에 유연하게 대응하는 에이전트를 구성함.
│
│
├── configs.py           # configs loading 모듈; 재사용 하는 경우가 많아 따로 뺐음 
│   
├── configs/ 
│   ├── config_emb.yaml  # pipe_extract, pipe_vectorize 및 preprocess 패키지에서 쓰일 모델 및 임베딩 설정
│   ├── config_eval.yaml  # pipe_eval에서 쓰일 모델 및 임베딩 설정
│   └── config_infer.yaml  # run_chatbot, pipe_infer에 쓰이는 모델 및 임베딩 설정
│            
├── pyairports/         # vLLM 실행 관련 파일
│ 
├── vllm_setup.sh       # 8080 포트에서 vLLM 서버를 실행하는 스크립트
├── pyairports/         # vLLM 실행 관련 파일
│
├── requirements_rag.txt # pipe_extract 제외 가상환경 
└── requirements_paddleenv.txt # pipe_extract 실행시 가상환경 
```

---


# 실행 환경

* Python 3.10 이상
* CUDA 12.x 권장
* GPU 환경 필수 (vLLM 사용 시)

---

# 실행 방법

## 1. 환경 설정

```
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements_rag.txt

conda create -n paddle_env python=3.10
conda activate paddle_env
pip install -r requirements_paddleenv.txt

```
* 참고로 가상환경을 전처리용과 챗봇용으로 두 개로 나누어 생성한 이유는, 전처리에 사용되는 라이브러리와 챗봇에 사용되는 라이브러리 간에 의존성 충돌이 발생할 수 있기 때문임.
* 추가로 누락된 라이브러리는 pip로 다운로드할 것

## 2. 데이터 추출 및 벡터 임베딩

```
python -m scripts.pipe_extract
python -m scripts.pipe_vectorize
```
input/ouput 데이터 저장 디렉터리에 유의하여 사용할 것 

### ⚠️ PaddleOCRVL GPU 사용 문제 해결 방법

전처리 할 때 PaddleOCRVL로 실행하면 **GPU 메모리는 잡히지만 실제 GPU 연산이 수행되지 않는 문제**가 발생할 수 있다.

##### (1) GPU 사용 여부 확인

아래 명령어로 GPU 사용 여부를 확인할 수 있다.

```bash
watch -n 1 "nvidia-smi | grep -E 'MiB|%' | head -1"
```

###### 문제가 있는 경우 (GPU 미사용)

```
|  0%   32C    P8    15W / 350W |      0MiB / 24564MiB |      0%      Default |
```

이처럼 **GPU 사용률이 0%로 표시되면 GPU 연산이 이루어지지 않는 상태**이다.

#### 원인
Attention 계산 중 일부 **Tensor가 CPU에 남아있어서 병목이 발생**한다.  
이 때문에 GPU는 메모리만 사용되고 실제 연산은 CPU에서 진행되어 **속도가 매우 느려지거나 프로그램이 다운될 수 있다.**

#### 해결 방법
CPU에 남아있는 **Tensor들을 모두 GPU로 이동시키도록 코드 수정**이 필요하다.

---

##### (2) 수정해야 할 파일 위치

다음 파일을 수정한다.

```
/home/[본인계정명]/anaconda3/envs/paddle_env/lib/python3.10/site-packages/paddlex/inference/models/doc_vlm/predictor.py
```

수정 대상 함수

```
_switch_inputs_to_device
```

---

##### (3) 코드 수정

다음과 같이 고친다.(_to_device함수를 원래 함수 내부에 삽입한다.)

```python
def _switch_inputs_to_device(self, input_dict):
    """Switch the input to the specified device"""
    import paddle

    if self.device is None:
        return input_dict

    def _to_device(obj):
        if isinstance(obj, paddle.Tensor):
            return paddle.to_tensor(obj, place=self.device)
        elif isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(v) for v in obj)
        else:
            return obj

    return _to_device(input_dict)
```

---



## 3. vLLM 서버 실행

vLLM은 LLM을 서버 프로세스로 실행한 뒤 API 방식으로 호출하는 추론 엔진이다.

```
bash vllm_setup.sh
```

## 3. 챗봇 실행

```
python -m scripts.run_chatbot
```

## 4. 대규모 추론코드 실행

```
python -m scripts.pipe_infer
```

## 5. 검색 성능 및 답변 품질 평가 실행

```
python -m scripts.pipe_eval
```

---