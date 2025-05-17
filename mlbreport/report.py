import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ======= 資料讀取與前處理 =======
merged = pd.read_csv("C:/Users/richc/OneDrive/桌面/專題/mlbdata/merged_output_team_year_avg.csv")
target_col = 'Win_Pct'
exclude_cols = ['Wins', 'Losses', 'W', 'L', 'OPS', 'ERA', target_col]
exclude_cols = [col for col in exclude_cols if col in merged.columns]
num_cols = merged.select_dtypes(include='number').columns.tolist()
feature_candidates = [c for c in num_cols if c not in exclude_cols]

# ======= 選擇與勝率相關變數 =======
corr = merged[feature_candidates + [target_col]].corr()
win_corr = corr[target_col].drop(target_col).abs()
threshold = 0.5
selected_features = win_corr[win_corr >= threshold].index.tolist()
if not selected_features:
    selected_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]

# ======= 建模函式 =======
def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    dt = DecisionTreeRegressor(random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    dt_pred = dt.predict(X_test)
    results = {
        'Random Forest': {'MSE': mean_squared_error(y_test, rf_pred), 'R2': r2_score(y_test, rf_pred)},
        'Decision Tree': {'MSE': mean_squared_error(y_test, dt_pred), 'R2': r2_score(y_test, dt_pred)}
    }
    return results, rf, dt

# ======= 單球隊逐年資料 =======
teams = merged['Team'].unique()
results_single = []
for team in teams:
    df_team = merged[merged['Team'] == team]
    if len(df_team) < 5:
        continue
    X = df_team[selected_features]
    y = df_team[target_col]
    res, _, _ = run_model(X, y)
    for model in res:
        results_single.append({'Team': team, 'Model': model, 'MSE': res[model]['MSE'], 'R2': res[model]['R2']})
df_single = pd.DataFrame(results_single)

# ======= 單球隊10年平均資料 =======
df_10y = merged.groupby('Team')[selected_features + [target_col]].mean().reset_index()
X_10y = df_10y[selected_features]
y_10y = df_10y[target_col]
res_10y, rf_10y, dt_10y = run_model(X_10y, y_10y)
df_10y_results = pd.DataFrame([{'Team': 'All', 'Model': k, 'MSE': v['MSE'], 'R2': v['R2']} for k, v in res_10y.items()])

# ======= 全MLB逐年平均資料 =======
df_all_10y = merged.groupby('year')[selected_features + [target_col]].mean().reset_index()
X_all_10y = df_all_10y[selected_features]
y_all_10y = df_all_10y[target_col]
res_all_10y, _, _ = run_model(X_all_10y, y_all_10y)
df_all_10y_results = pd.DataFrame([{'Team': 'MLB_years', 'Model': k, 'MSE': v['MSE'], 'R2': v['R2']} for k, v in res_all_10y.items()])

# ======= 合併結果與繪圖 =======
df_all_results = pd.concat([df_single, df_10y_results, df_all_10y_results], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', data=df_all_results)
plt.title("Model MSE Comparison")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='R2', data=df_all_results)
plt.title("Model R2 Comparison")
plt.show()

# ======= 畫出熱力圖：與勝率最相關變數 =======
exclude_cols = ['勝', '敗', 'W', 'L', 'Wins', 'Losses', 'OPS', 'ERA', target_col]
exclude_cols = [col for col in exclude_cols if col in merged.columns]
numeric_cols = merged.select_dtypes(include='number').columns.difference(exclude_cols)
corr_matrix = merged[numeric_cols.tolist() + [target_col]].corr()
corr_target = corr_matrix[target_col].drop(target_col)
top_n = 10
important_vars = corr_target.abs().sort_values(ascending=False).head(top_n).index.tolist()
important_corrs = corr_target.loc[important_vars]

if important_vars:
    plt.figure(figsize=(4, top_n * 0.6))
    sns.heatmap(
        important_corrs.to_frame(),
        annot=True, cmap='coolwarm', center=0, cbar=True,
        yticklabels=True, xticklabels=[target_col],
        linewidths=0.5, linecolor='gray', fmt=".2f"
    )
    plt.title(f"Top {top_n} Variables Correlation with {target_col}")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/heatmap_team_year_avg.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"熱力圖已儲存至 {save_path}")
else:
    print("沒有重要變數可畫圖")

# ======= 對照組模型（只用 OPS & ERA） =======
basic_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]
X_basic = merged[basic_features]
y_basic = merged[target_col]
res_basic, _, _ = run_model(X_basic, y_basic)

df_compare = pd.DataFrame(
    [{'Feature_Set': 'OPS+ERA', 'Model': model, 'MSE': res_basic[model]['MSE'], 'R2': res_basic[model]['R2']} for model in res_basic] +
    [{'Feature_Set': 'Selected_Features', 'Model': model, 'MSE': res_10y[model]['MSE'], 'R2': res_10y[model]['R2']} for model in res_10y]
)

plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='R2', hue='Feature_Set', data=df_compare)
plt.title("R² Comparison: OPS+ERA vs Selected Features")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='MSE', hue='Feature_Set', data=df_compare)
plt.title("MSE Comparison: OPS+ERA vs Selected Features")
plt.tight_layout()
plt.show()

# ======= 可視化決策樹 =======
# 1. Decision Tree (單一)
# 繪製決策樹並存成圖片
plt.figure(figsize=(20, 10))
plot_tree(dt_10y, feature_names=selected_features, filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/decision_tree_10y.png", dpi=300)
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(rf_10y.estimators_[0], feature_names=selected_features, filled=True, rounded=True, max_depth=3)
plt.title("Random Forest - One Tree (max_depth=3)")
plt.tight_layout()
plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/random_forest_tree_10y.png", dpi=300)
plt.close()

# ======= 儲存結果至 Excel 檔（多工作表） =======
excel_output_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/mlb_model_results.xlsx"

# 格式化結果數字
df_all_results_formatted = df_all_results.copy()
df_all_results_formatted['MSE'] = df_all_results_formatted['MSE'].round(9)
df_all_results_formatted['R2'] = df_all_results_formatted['R2'].round(9)

df_compare_formatted = df_compare.copy()
df_compare_formatted['MSE'] = df_compare_formatted['MSE'].round(9)
df_compare_formatted['R2'] = df_compare_formatted['R2'].round(9)

# 儲存成 Excel (多工作表)
with pd.ExcelWriter(excel_output_path) as writer:
    df_compare_formatted.to_excel(writer, sheet_name='Compare', index=False)
    df_all_results_formatted.to_excel(writer, sheet_name='AllTeams', index=False)

print(f"\n✅ 結果已成功儲存為 Excel 檔：{excel_output_path}")

