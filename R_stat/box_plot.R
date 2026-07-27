# 필요한 패키지
# install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "stringr"))

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)

# ── 1. 데이터 불러오기 ──────────────────────────────────────────────
stats <- read_csv("boxplot_stats.csv")
raw_df <- read_csv("table7_scores_byqa.csv")

metric_labels <- c(
  bertscore     = "BERTScore (F1)",
  sbert            = "SBERT",
  rouge1           = "ROUGE-1",
  rougeL           = "ROUGE-L"
)

metric_levels <- unname(metric_labels)

# Box plot용 요약 통계
stats <- stats %>%
  mutate(
    metric_label = factor(
      metric_labels[metric],
      levels = metric_levels
    ),
    method = factor(
      method,
      levels = c("LLM-only", "RAG")
    )
  )

# ── 2. 원본 데이터를 violin plot용 long format으로 변환 ────────────
violin_data <- raw_df %>%
  filter(setting %in% c("RAG", "LLM-only")) %>%
  select(
    setting,
    qid,
    bertscore,
    sbert,
    rouge1,
    rougeL
  ) %>%
  pivot_longer(
    cols = c(
      bertscore,
      sbert,
      rouge1,
      rougeL
    ),
    names_to = "metric",
    values_to = "score"
  ) %>%
  mutate(
    method = recode(
      setting,
      "llmonly_test" = "LLM-only",
      "rag_test" = "RAG"
    ),
    method = factor(
      method,
      levels = c("LLM-only", "RAG")
    ),
    metric_label = factor(
      metric_labels[metric],
      levels = metric_levels
    )
  ) %>%
  filter(!is.na(score))

# ── 3. Outlier 문자열을 long format으로 변환 ───────────────────────
outlier_points <- stats %>%
  filter(n_outliers > 0, !is.na(outliers), outliers != "") %>%
  select(metric, metric_label, method, outliers) %>%
  separate_rows(outliers, sep = ";") %>%
  mutate(outliers = as.numeric(outliers)) %>%
  filter(!is.na(outliers))

# ── 4. 유의성 라벨 위치 계산 ────────────────────────────────────────
sig_labels <- violin_data %>%
  group_by(metric, metric_label) %>%
  summarise(
    score_min = min(score, na.rm = TRUE),
    score_max = max(score, na.rm = TRUE),
    score_range = score_max - score_min,
    y_pos = score_max + if_else(
      score_range > 0,
      score_range * 0.10,
      0.05
    ),
    .groups = "drop"
  ) %>%
  left_join(
    stats %>%
      distinct(metric, sig_label),
    by = "metric"
  )

# ── 5. Violin + Box plot 그리기 ──────────────────────────────────────
p <- ggplot() +
  
  # 전체 분포
  geom_violin(
    data = violin_data,
    aes(
      x = method,
      y = score,
      fill = method
    ),
    width = 0.85,
    trim = TRUE,
    alpha = 0.45,
    linewidth = 0.4,
    color = "black"
  ) +
  
  # Box plot
  geom_boxplot(
    data = stats,
    aes(
      x = method,
      ymin = whisker_low,
      lower = q1,
      middle = median,
      upper = q3,
      ymax = whisker_high,
      fill = method
    ),
    stat = "identity",
    width = 0.20,
    linewidth = 0.5,
    alpha = 0.85,
    color = "black"
  ) +
  
  # 이상치
  geom_jitter(
    data = outlier_points,
    aes(
      x = method,
      y = outliers
    ),
    width = 0.05,
    size = 1.2,
    shape = 21,
    color = "black",
    fill = "white"
  ) +
  
  # 유의성 표시
  geom_text(
    data = sig_labels,
    aes(
      x = 1.5,
      y = y_pos,
      label = sig_label
    ),
    inherit.aes = FALSE,
    size = 6
  ) +
  
  facet_wrap(
    ~ metric_label,
    scales = "free_y",
    nrow = 1
  ) +
  
  scale_fill_manual(
    values = c(
      "LLM-only" = "#B4B2A9",
      "RAG" = "#1D9E75"
    )
  ) +
  
  labs(
    x = NULL,
    y = "Score",
    fill = NULL
  ) +

  coord_cartesian(ylim = c(0, 1)) +
    
  theme_minimal(base_size = 16) +

  
  theme(
    legend.position = "top",
    legend.text = element_text(size = 14),
    strip.text = element_text(
      face = "bold",
      size = 15
    ),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    axis.title.y = element_text(size = 15)
  )

print(p)

# ── 6. 저장 ──────────────────────────────────────────────────────────
ggsave(
  "violin_boxplot_rag_vs_llmonly.png",
  plot = p,
  width = 10,
  height = 4.2,
  dpi = 300
)

ggsave(
  "violin_boxplot_rag_vs_llmonly.pdf",
  plot = p,
  width = 10,
  height = 4.2
)