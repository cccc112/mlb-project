import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler

# 讀取三份 2025 年資料
path_ops = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/mlb-player-stats-2025.csv"
path_w = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/mlb_standings_2025_final.csv"
path_era = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/mlb-player-stats-P-2025.csv"

df_ops = pd.read_csv(path_ops)
df_w = pd.read_csv(path_w)
df_era = pd.read_csv(path_era)

# 處理 W 勝敗場欄位格式 'XX--YY'
def extract_wins(record_str):
    match = re.match(r'(\d+)--\d+', str(record_str))
    return int(match.group(1)) if match else np.nan

def extract_losses(record_str):
    match = re.match(r'\d+--(\d+)', str(record_str))
    return int(match.group(1)) if match else np.nan

if 'Home_Record' in df_w.columns:
    df_w['Home_Wins'] = df_w['Home_Record'].apply(extract_wins)
    df_w['Home_Losses'] = df_w['Home_Record'].apply(extract_losses)

if 'Away_Record' in df_w.columns:
    df_w['Away_Wins'] = df_w['Away_Record'].apply(extract_wins)
    df_w['Away_Losses'] = df_w['Away_Record'].apply(extract_losses)

# 計算勝差
df_w['Home_WinDiff'] = df_w['Home_Wins'] - df_w['Home_Losses']
df_w['Away_WinDiff'] = df_w['Away_Wins'] - df_w['Away_Losses']
df_w['Total_WinDiff'] = df_w['Home_WinDiff'] + df_w['Away_WinDiff']

# 基本欄位處理
for df in [df_ops, df_era, df_w]:
    df.columns = df.columns.str.strip()
    if 'Team' in df.columns:
        df['Team'] = df['Team'].str.strip().str.upper().replace({'WSH': 'WAS'})
    df['year'] = 2025

try:
    df_w['Win_Pct'] = pd.to_numeric(df_w['Win_Pct'])
except:
    print("Win_Pct 欄位轉換失敗")

# 補值缺失欄位
for df, name in [(df_ops, "OPS"), (df_w, "勝率"), (df_era, "ERA")]:
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

# 每隊平均資料（2025）
df_ops_grouped = df_ops.groupby('Team').mean(numeric_only=True).reset_index()
df_era_grouped = df_era.groupby('Team').mean(numeric_only=True).reset_index()
columns_to_keep = [col for col in df_w.columns if col not in ['Home_Record', 'Away_Record']]
df_w_grouped = df_w[columns_to_keep].copy()

# 合併三份資料
merged = df_ops_grouped.merge(df_w_grouped, on='Team', how='left', suffixes=('', '_W'))
merged = merged.merge(df_era_grouped, on='Team', how='inner', suffixes=('', '_ERA'))
merged['year'] = 2025

# 儲存原始合併資料
original_output_path = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/merged_output_2025.csv"
merged.to_csv(original_output_path, index=False)

# 標準化數據
merged_standardized = merged.copy()
non_numeric_cols = ['Team', 'year', 'League'] if 'League' in merged.columns else ['Team', 'year']
numeric_cols = [col for col in merged.columns if col not in non_numeric_cols and merged[col].dtype != 'object']

scaler = StandardScaler()
merged_standardized[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

# 儲存標準化後資料
standardized_output_path = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/merged_output_2025_standardized.csv"
merged_standardized.to_csv(standardized_output_path, index=False)

# 儲存關鍵特徵
key_features = ['Team', 'year', 'OPS', 'ERA', 'Win_Pct', 'Total_WinDiff']
available_features = [col for col in key_features if col in merged_standardized.columns]

if len(available_features) >= 3:
    key_features_df = merged_standardized[available_features]
    key_features_path = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/2025/key_features_2025_standardized.csv"
    key_features_df.to_csv(key_features_path, index=False)
    print(f"\n✅ 關鍵特徵儲存成功：{available_features}")
    print(f"📄 儲存路徑：{key_features_path}")
else:
    print("\n⚠️ 無法儲存關鍵特徵，請確認欄位名稱是否正確")
