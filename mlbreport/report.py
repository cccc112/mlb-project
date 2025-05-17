import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 讀取資料
merged = pd.read_csv("C:/Users/richc/OneDrive/桌面/專題/mlbdata/merged_output_team_year_avg.csv")

target_col = 'Win_Pct'

# 排除欄位
exclude_cols = ['Wins', 'Losses', 'W', 'L', 'OPS', 'ERA', target_col]
exclude_cols = [col for col in exclude_cols if col in merged.columns]

# 取得數值欄位，排除exclude_cols
num_cols = merged.select_dtypes(include='number').columns.tolist()
feature_candidates = [c for c in num_cols if c not in exclude_cols]

print(f"數值候選自變數: {feature_candidates}")

# 計算皮爾森相關係數
corr = merged[feature_candidates + [target_col]].corr()

# 與目標相關性 (絕對值)
win_corr = corr[target_col].drop(target_col).abs()

print("各候選變數與勝率絕對相關係數:")
print(win_corr)

threshold = 0.5
selected_features = win_corr[win_corr >= threshold].index.tolist()
print(f"選出自變數(相關係數≥{threshold}): {selected_features}")

# 如果沒選到變數就用 OPS 和 ERA 當作對照組(如果有的話)
if not selected_features:
    print("未選出自變數，使用 OPS 和 ERA 作為對照組")
    selected_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]

# 下面是一個簡單的函式來跑模型
def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    dt = DecisionTreeRegressor(random_state=42)

    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    dt_pred = dt.predict(X_test)

    results = {
        'Random Forest': {
            'MSE': mean_squared_error(y_test, rf_pred),
            'R2': r2_score(y_test, rf_pred)
        },
        'Decision Tree': {
            'MSE': mean_squared_error(y_test, dt_pred),
            'R2': r2_score(y_test, dt_pred)
        }
    }
    return results

# 一、單球隊單年資料 (假設 Team 和 year 欄位存在)
teams = merged['Team'].unique()
results_single = []
for team in teams:
    df_team = merged[merged['Team'] == team]
    if len(df_team) < 5:
        continue
    X = df_team[selected_features]
    y = df_team[target_col]
    res = run_model(X, y)
    for model in res:
        results_single.append({'Team': team, 'Model': model, 'MSE': res[model]['MSE'], 'R2': res[model]['R2']})

df_single = pd.DataFrame(results_single)

# 二、單球隊10年平均資料
df_10y = merged.groupby('Team')[selected_features + [target_col]].mean().reset_index()
X_10y = df_10y[selected_features]
y_10y = df_10y[target_col]
res_10y = run_model(X_10y, y_10y)
df_10y_results = pd.DataFrame([{'Team':'All','Model':k,'MSE':v['MSE'],'R2':v['R2']} for k,v in res_10y.items()])

# 三、全MLB 10年平均資料
df_all_10y = merged.groupby('year')[selected_features + [target_col]].mean().reset_index()
X_all_10y = df_all_10y[selected_features]
y_all_10y = df_all_10y[target_col]
res_all_10y = run_model(X_all_10y, y_all_10y)
df_all_10y_results = pd.DataFrame([{'Team':'MLB_years','Model':k,'MSE':v['MSE'],'R2':v['R2']} for k,v in res_all_10y.items()])

# 合併所有結果
df_all_results = pd.concat([df_single, df_10y_results, df_all_10y_results], ignore_index=True)

print(df_all_results)

# 繪圖看結果
plt.figure(figsize=(12,6))
sns.boxplot(x='Model', y='MSE', data=df_all_results)
plt.title("Model MSE Comparison")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='Model', y='R2', data=df_all_results)
plt.title("Model R2 Comparison")
plt.show()

target_col = 'Win_Pct'

exclude_cols = ['勝', '敗', 'W', 'L', 'Wins', 'Losses', 'OPS', 'ERA', target_col]
exclude_cols = [col for col in exclude_cols if col in merged.columns]

numeric_cols = merged.select_dtypes(include='number').columns.difference(exclude_cols)

corr_matrix = merged[numeric_cols.tolist() + [target_col]].corr()
corr_target = corr_matrix[target_col].drop(target_col)

top_n = 6
important_vars = corr_target.abs().sort_values(ascending=False).head(top_n).index.tolist()
important_corrs = corr_target.loc[important_vars]

if important_vars:
    plt.figure(figsize=(4, top_n * 0.6))
    sns.heatmap(
        important_corrs.to_frame(),
        annot=True,
        cmap='coolwarm',
        center=0,
        cbar=True,
        yticklabels=True,
        xticklabels=[target_col],
        linewidths=0.5,
        linecolor='gray',
        fmt=".2f"
    )
    plt.title(f"Top {top_n} Variables Correlation with {target_col}")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 沒有重要變數可畫圖")
