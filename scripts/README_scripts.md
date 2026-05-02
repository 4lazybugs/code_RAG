# Pipeline Module Overview

`src/` 디렉토리 내 여러 모듈을 조합하여 실행하는 스크립트 모음입니다.  
각 `pipe_*`는 단일 스크립트가 아닌, 연관 모듈들을 묶어 처리하는 **파이프라인 단위**입니다.

---

## 전체 파이프라인 구성

```
raw_db (PDF 등)
    │
    ▼
[전처리]
pipe_extract  →  pipe_pproc
    │
    ▼
[DB화]
pipe_vectorize
    │
    ▼
[질문 생성]
pipe_chunk  →  pipe_qagen
    │
    ▼
[평가]
pipe_infer  →  pipe_eval
```

---

## 파이프라인 상세

### 🔧 전처리

| 모듈 | 역할 |
|------|------|
| `pipe_extract` | raw_db(PDF 등)를 고화질 전처리 및 OCR로 텍스트 추출 |
| `pipe_pproc` | OCR 결과에서 노이즈·무의미 데이터 제거, 전문가 지정 필터링 적용 후 저장 |

> ⚠️ `pipe_chunk`의 decision 로직은 추후 `pipe_pproc`으로 이식 예정

---

### 🗄️ DB화

| 모듈 | 역할 |
|------|------|
| `pipe_vectorize` | 전처리된 텍스트를 벡터화하여 벡터 DB로 저장 |

---

### ❓ 질문 생성

| 모듈 | 역할 |
|------|------|
| `pipe_chunk` | 텍스트를 줄글로 변환 후 LumberChunking으로 자연어 단위 청크 분리 |
| `pipe_qagen` | 청크 기반 질문 생성 → 평가용 `(Q, A, D)` 셋 구성 (D는 QA로부터 생성) |

---

### 📊 평가

| 모듈 | 역할 |
|------|------|
| `pipe_infer` | 생성된 QA 기반 대규모 RAG 추론 실행 및 결과 저장 |
| `pipe_eval` | 저장된 추론 결과를 다양한 성능 지표로 평가 |

---

## 실행 시나리오

### 1. 전체 초기화 및 검증 (최초 구축 시)
> raw_db 투입부터 평가까지 전 단계 실행

```
pipe_extract → pipe_pproc → pipe_vectorize → pipe_chunk → pipe_qagen → pipe_infer → pipe_eval
```

---

### 2. 새로운 자료 추가 시 (최신화)
> 기존 QA·평가는 유지, DB만 갱신

```
pipe_extract → pipe_pproc → pipe_vectorize
```

---

### 3. 중간 검증만 필요할 때
> 기존 DB 기반으로 질문 생성 및 평가만 재실행

```
pipe_chunk → pipe_qagen → pipe_infer → pipe_eval
```

---

## 서버 실행 기준

- 모든 파이프라인은 **서버 사이드**에서 실행됨
- 각 `pipe_*`는 독립적으로 중단·재개 가능한 단위로 설계됨
- 파이프라인 경계마다 중간 결과를 DB에 저장하여 디버깅 및 재실행 지점으로 활용