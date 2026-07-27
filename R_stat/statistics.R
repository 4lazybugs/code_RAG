# 필요한 패키지 설치
# install.packages(c("dplyr", "tidyr", "rstatix", "readr"))

library(dplyr)
library(tidyr)
library(rstatix)
library(readr)
# 
# 데이터 불러오기
df <- read_csv("table7_scores_byqa.csv")

# 분석할 지표
metrics <- c(
  "bertscore",
  "sbert",
  "rouge1",
  "rougeL"
)

# test set만 선택하고 qid 기준으로 RAG와 LLM-only를 한 행에 배치
test_wide <- df %>%
  filter(setting %in% c("RAG", "LLM-only")) %>%
  select(setting, qid, all_of(metrics)) %>%
  pivot_wider(
    names_from = setting,
    values_from = all_of(metrics),
    names_sep = "_"
  ) %>%
  arrange(qid)

cat("대응된 질문 수:", nrow(test_wide), "\n")

# ── 1. Wilcoxon 검정 (유의성만 뽑아서 box plot에 얹을 용도) ──────────────
run_wilcoxon <- function(metric) {
  rag_col <- paste0(metric, "_RAG")
  llm_col <- paste0(metric, "_LLM-only")
  
  temp <- test_wide %>%
    select(rag = all_of(rag_col), llm = all_of(llm_col)) %>%
    drop_na()
  
  test_result <- wilcox.test(
    temp$rag, temp$llm,
    paired = TRUE, alternative = "two.sided",
    exact = FALSE, correct = TRUE
  )
  
  tibble(
    metric = metric,
    n = nrow(temp),
    p_value = test_result$p.value
  )
}

wilcoxon_results <- bind_rows(lapply(metrics, run_wilcoxon)) %>%
  mutate(
    p_holm = p.adjust(p_value, method = "holm"),
    significant_holm = p_holm < 0.05,
    sig_label = case_when(
      p_holm < 0.001 ~ "***",
      p_holm < 0.01  ~ "**",
      p_holm < 0.05  ~ "*",
      TRUE           ~ "n.s."
    )
  )

# ── 2. Box plot에 필요한 5-number summary + outlier 추출 ────────────────
get_box_stats <- function(metric, method_label, values) {
  values <- values[!is.na(values)]
  bs <- boxplot.stats(values)  # base R: whisker_low, Q1, median, Q3, whisker_high
  
  tibble(
    metric        = metric,
    method        = method_label,
    n             = length(values),
    whisker_low   = bs$stats[1],
    q1            = bs$stats[2],
    median        = bs$stats[3],
    q3            = bs$stats[4],
    whisker_high  = bs$stats[5],
    n_outliers    = length(bs$out),
    outliers      = paste(round(bs$out, 4), collapse = ";")
  )
}

boxplot_data <- bind_rows(
  lapply(metrics, function(m) {
    rag_col <- paste0(m, "_RAG")
    llm_col <- paste0(m, "_LLM-only")
    bind_rows(
      get_box_stats(m, "RAG",      test_wide[[rag_col]]),
      get_box_stats(m, "LLM-only", test_wide[[llm_col]])
    )
  })
)

# ── 3. 유의성 라벨을 box plot 데이터에 합치기 ────────────────────────────
boxplot_data <- boxplot_data %>%
  left_join(
    wilcoxon_results %>% select(metric, p_holm, sig_label, significant_holm),
    by = "metric"
  ) %>%
  mutate(across(c(whisker_low, q1, median, q3, whisker_high), ~ round(.x, 4)))

print(boxplot_data)

# CSV로 저장 (box plot 그리기용 최종 데이터)
write_csv(boxplot_data, "boxplot_stats.csv")