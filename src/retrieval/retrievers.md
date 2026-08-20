## Chroma vs LexDB(BM25) 비교

### 저장 형태
- **Chroma**: 문장을 벡터(숫자 배열)로 변환해 SQLite + 파일 기반 DB에 저장
- **LexDB**: 토큰화된 단어 빈도 통계를 `.pkl` 파일로 저장

### 검색 방식
- **Chroma**: 쿼리를 벡터로 변환 후 코사인 유사도 계산
$$\text{score} = \cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}||\vec{d}|}$$

- **LexDB**: 쿼리 토큰의 출현 빈도 기반 BM25 점수 계산
$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1\left(1 - b + b\dfrac{|d|}{\text{avgdl}}\right)}$$

### 파라미터 설명
| 기호 | 설명 |
|---|---|
| $f(t, d)$ | 문서 $d$ 에서 단어 $t$ 의 출현 빈도 |
| $\|d\|$ | 문서 길이 |
| $\text{avgdl}$ | 전체 문서의 평균 길이 |
| $k_1$ | 단어 빈도 포화 파라미터 (보통 $1.5$) |
| $b$ | 문서 길이 정규화 파라미터 (보통 $0.75$) |
| $\text{IDF}(t)$ | $\log\dfrac{N - n(t) + 0.5}{n(t) + 0.5}$ |

### 강점 비교
| | Chroma | LexDB (BM25) |
|---|---|---|
| **강점** | 동의어, 유사표현 커버 | 전문용어, 숫자, 코드 정확 매칭 |
| **약점** | 정확한 키워드 희석 가능 | 동의어, 유사표현 못 잡음 |
| **속도** | 문서 많을수록 느림 | 상대적으로 빠름 |