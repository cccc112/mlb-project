import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

# 設定路徑
base_path_data = "C:/Users/richc/OneDrive/桌面/專題/mlbdata"
combined_path = os.path.join(base_path_data, "OPS", "mlb_player_stats_10years_combined.csv")
combined_w_path = os.path.join(base_path_data, "W", "mlb_player_stats_10years_combined-W.csv")
combined_era_path = os.path.join(base_path_data, "ERA", "mlb_player_stats_10years_combined-era.csv")

# 讀取資料
df_combined = pd.read_csv(combined_path)
df_w = pd.read_csv(combined_w_path)
df_era = pd.read_csv(combined_era_path)

# 處理 'XX--YY' 格式
def extract_wins(record_str):
    match = re.match(r'(\d+)--\d+', str(record_str))
    return int(match.group(1)) if match else np.nan

def extract_losses(record_str):
    match = re.match(r'\d+--(\d+)', str(record_str))
    return int(match.group(1)) if match else np.nan

# 提取 Home / Away 勝負場
if 'Home_Record' in df_w.columns:
    df_w['Home_Wins'] = df_w['Home_Record'].apply(extract_wins)
    df_w['Home_Losses'] = df_w['Home_Record'].apply(extract_losses)

if 'Away_Record' in df_w.columns:
    df_w['Away_Wins'] = df_w['Away_Record'].apply(extract_wins)
    df_w['Away_Losses'] = df_w['Away_Record'].apply(extract_losses)

# 加入勝差欄位
df_w['Home_WinDiff'] = df_w['Home_Wins'] - df_w['Home_Losses']
df_w['Away_WinDiff'] = df_w['Away_Wins'] - df_w['Away_Losses']
df_w['Total_WinDiff'] = df_w['Home_WinDiff'] + df_w['Away_WinDiff']

# 格式處理與欄位清理
for df in [df_combined, df_era]:
    df.columns = df.columns.str.strip()
    df['Team'] = df['Team'].str.strip().str.upper().replace({'WSH': 'WAS'})
    df['year'] = df['year'].astype(int)

df_w.columns = df_w.columns.str.strip()
df_w['Team'] = df_w['Team'].str.strip().str.upper().replace({'WSH': 'WAS'})
df_w['year'] = df_w['year'].astype(int)

try:
    df_w['Win_Pct'] = pd.to_numeric(df_w['Win_Pct'])
except:
    print("Win_Pct 欄位轉換失敗")

# 填補缺失值
for df, name in [(df_combined, "OPS"), (df_w, "勝率"), (df_era, "ERA")]:
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

# 平均每隊每年資料
df_combined_grouped = df_combined.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()
df_era_grouped = df_era.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()
columns_to_keep = [col for col in df_w.columns if col not in ['Home_Record', 'Away_Record']]
df_w_grouped = df_w[columns_to_keep].copy()

# 合併
merged = df_combined_grouped.merge(df_w_grouped, on=['Team', 'year'], how='left', suffixes=('', '_W'))
merged = merged.merge(df_era_grouped, on=['Team', 'year'], how='inner', suffixes=('', '_ERA'))

# 儲存未標準化版本
original_output_path = os.path.join(base_path_data, "merged_output_team_year_avg.csv")
merged.to_csv(original_output_path, index=False)

# 數據標準化
merged_standardized = merged.copy()
non_numeric_cols = ['Team', 'year', 'League'] if 'League' in merged.columns else ['Team', 'year']
numeric_cols = [col for col in merged.columns if col not in non_numeric_cols and merged[col].dtype != 'object']

scaler = StandardScaler()
merged_standardized[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

# 儲存標準化版本
standardized_output_path = os.path.join(base_path_data, "merged_output_team_year_avg_standardized.csv")
merged_standardized.to_csv(standardized_output_path, index=False)

# 關鍵特徵儲存
key_features = ['Team', 'year', 'OPS', 'ERA', 'Win_Pct', 'Total_WinDiff']
available_features = [col for col in key_features if col in merged_standardized.columns]

if len(available_features) >= 3:
    key_features_df = merged_standardized[available_features]
    key_features_path = os.path.join(base_path_data, "key_features_standardized.csv")
    key_features_df.to_csv(key_features_path, index=False)
    print(f"\n關鍵特徵儲存成功：{available_features}")
    print(f"儲存路徑：{key_features_path}")
else:
    print("\n無法儲存關鍵特徵，請確認欄位名稱是否正確")
