import pandas as pd

# Đọc file CSV
df = pd.read_csv("/home/manh/Projects/temp/CS431/src_agent/eval_results_checkpoint_318.csv", encoding="utf-8")

# Ghi ra file Excel
df.to_excel("/home/manh/Projects/temp/CS431/data_318_1000sample.xlsx", index=False)
