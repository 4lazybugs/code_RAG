import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "results/summary.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)
sheet_names = list(sheets.keys())

# 0번 시트: 평균, 1번 시트: 표준편차
mean_df = sheets[sheet_names[0]]
std_df  = sheets[sheet_names[1]]

# 🔥 원하는 순서대로 metrics 재정렬
metrics = ["rouge1", "rougeL", "bert", "sbert"]

# 필요한 컬럼만 사용
cols = ["mode"] + [m for m in metrics if m in mean_df.columns]
mean_df = mean_df[cols].copy()
std_df  = std_df[cols].copy()

# ---- mode 이름 바꾸기 ----
rename_map = {
    "partial_10": "RAG",
    "raw_llm": "LLM",
}
mean_df["mode"] = mean_df["mode"].replace(rename_map)
std_df["mode"]  = std_df["mode"].replace(rename_map)

# mode를 index로
mean_df.set_index("mode", inplace=True)
std_df.set_index("mode", inplace=True)

os.makedirs("results", exist_ok=True)

ax = mean_df[metrics].plot(
    kind="bar",
    yerr=std_df[metrics],   # 대칭 error bar
    capsize=4,
)

ax.set_title("RAG Performance")
ax.set_ylabel("score")
ax.set_xlabel("model")
plt.xticks(rotation=0)
plt.ylim(0, None)

plt.tight_layout()
plt.savefig("results/result_with_errorbar.png")
plt.close()

print("saved to results/result_with_errorbar.png")
