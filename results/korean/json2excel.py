import os
import glob
import json
import time

import pandas as pd
import numpy as np

def collect_stats(json_dir, prefixes):
    summary = []
    for prefix in prefixes:
        if prefix == 'raw_llm':
            pattern = os.path.join(json_dir, 'raw_llm_*.json')
        else:
            pattern = os.path.join(json_dir, f'{prefix}_*_*.json')

        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            base, _ = os.path.splitext(fname)
            parts = base.split('_', 2)

            # mode, metric 결정 로직은 이전과 동일...
            if prefix == 'raw_llm':
                if parts[0]=='raw' and parts[1]=='llm' and len(parts)==3:
                    mode, metric = 'raw_llm', parts[2]
                else:
                    continue
            elif prefix in ('selfask','self_ask'):
                if parts[0].startswith('self') and len(parts)==3:
                    mode, metric = f'selfask_{parts[1]}', parts[2]
                else:
                    continue
            else:
                if parts[0]==prefix and len(parts)==3:
                    mode, metric = f'{prefix}_{parts[1]}', parts[2]
                else:
                    continue

            # JSON 로드
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # metric 값만 뽑아서
            vals = [item.get(metric)
                    for item in data
                    if isinstance(item.get(metric), (int, float))]
            if not vals:
                print(f"[WARNING] {fname}: '{metric}' 값이 없습니다.")
                continue

            # ⬇️ 여기만 변경: round(..., 3) 추가
            mean_val = round(np.mean(vals), 3)
            std_val  = round(np.std(vals),  3)

            summary.append({
                'mode':   mode,
                'metric': metric,
                'mean':   mean_val,
                'std':    std_val,
            })

    return summary

def save_to_excel(summary, output_path):
    if not summary:
        print("[ERROR] summary가 비어있습니다!")
        return

    df = pd.DataFrame(summary)
    df_mean = df.pivot(index='mode', columns='metric', values='mean')
    df_std  = df.pivot(index='mode', columns='metric', values='std')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        df_mean.to_excel(writer, sheet_name='mean')
        df_std .to_excel(writer, sheet_name='std')

    print(f"✅ 결과 저장: {output_path}")

if __name__ == '__main__':
    start = time.time()
    json_dir = 'results/korean/'
    prefixes = ['partial', 'adaptive', 'selfask', 'raw_llm']
    stats    = collect_stats(json_dir, prefixes)
    print(f"[INFO] 수집된 레코드 수: {len(stats)}개")
    save_to_excel(stats, os.path.join(json_dir, 'all_modes_summary.xlsx'))
    print(f"⏱ 전체 처리 시간: {time.time()-start:.2f}s")
