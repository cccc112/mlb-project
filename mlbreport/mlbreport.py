import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import os

# 設定路徑
base_path_data = "C:/Users/richc/OneDrive/桌面/專題/mlbdata"
base_path_report = "C:/Users/richc/OneDrive/桌面/專題/mlbreport"
# 資料路徑
combined_path = os.path.join(base_path_data, "OPS", "mlb_player_stats_10years_combined.csv")
combined_w_path = os.path.join(base_path_data, "W", "mlb_player_stats_10years_combined-W.csv")
combined_era_path = os.path.join(base_path_data, "ERA", "mlb_player_stats_10years_combined-era.csv")

# 讀取資料
df_combined = pd.read_csv(combined_path)
df_w = pd.read_csv(combined_w_path)
df_era = pd.read_csv(combined_era_path)

print("ERA columns:", df_era.columns)
print("OPS columns:", df_combined.columns)
print("勝率 columns:", df_w.columns)
# 統一格式
for df in [df_combined, df_w, df_era]:
    df.columns = df.columns.str.strip()
    df['Team'] = df['Team'].str.strip().str.upper()
    df['year'] = df['year'].astype(int)

# 平均每隊每年球員數據
df_combined_grouped = df_combined.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()
df_era_grouped = df_era.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()

# 勝率資料保留 Team 和 year 當 key
df_w_grouped = df_w.copy()
df_w_grouped['year'] = df_w_grouped['year'].astype(int)
df_w_grouped['Team'] = df_w_grouped['Team'].str.strip().str.upper()

# 合併資料
merged = df_combined_grouped.merge(df_w_grouped, on=['Team', 'year'], how='inner', suffixes=('', '_W'))
merged = merged.merge(df_era_grouped, on=['Team', 'year'], how='inner', suffixes=('', '_ERA'))

# 儲存合併結果
merged.to_csv(os.path.join(base_path_data, "merged_output_team_year_avg.csv"), index=False)
print(f"合併成功，共 {len(merged)} 筆資料")

# ---------- 熱力圖分析 ----------
winrate_col = 'Win_Pct'
exclude_cols = ['Wins', 'Losses', 'Win_Pct']
exclude_cols = [col for col in exclude_cols if col in merged.columns]

numeric_cols = merged.select_dtypes(include='number').columns.difference(exclude_cols)
correlation = merged[numeric_cols.tolist() + [winrate_col]].corr()
win_corr = correlation[[winrate_col]].drop(index=exclude_cols, errors='ignore')
win_corr = win_corr.sort_values(by=winrate_col, ascending=False)

# 畫熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(win_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation with Win Rate")
plt.tight_layout()
plt.savefig(os.path.join(base_path_report, "heatmap_team_year_avg.png"))
plt.show()

# ---------- 選出重要特徵 ----------
threshold = 0.5
selected_features = win_corr[win_corr[winrate_col].abs() >= threshold].index.tolist()
print("選出的重要特徵（相關係數 >= 0.5）：")
print(selected_features)

# ---------- 準備模型資料 ----------
label_encoder = LabelEncoder()
merged['Team'] = label_encoder.fit_transform(merged['Team'])

if 'League' in merged.columns:
    merged['League'] = merged['League'].map({'American League': 0, 'National League': 1})

X = merged[selected_features]
y = merged[winrate_col]

# ---------- 隨機森林模型 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

print("\nRandom Forest 預測結果：")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 特徵重要性圖
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
sorted_idx = importances.argsort()
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Selected Features)")
plt.tight_layout()
plt.savefig(os.path.join(base_path_report, "rf_feature_importance_team_year_avg.png"))
plt.show()

# ---------- 決策樹模型 ----------
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)

print("\nDecision Tree 預測結果：")
print("Mean Squared Error:", mean_squared_error(y_test, tree_predictions))
print("R2 Score:", r2_score(y_test, tree_predictions))

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree (Selected Features)")
plt.savefig(os.path.join(base_path_report, "decision_tree_team_year_avg.png"))
plt.show()

# ---------- 預測結果對比圖 ----------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, c='blue', label='Predicted', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('True Win Rate')
plt.ylabel('Predicted Win Rate')
plt.title('Random Forest: True vs Predicted Win Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path_report, "rf_vs_true_team_year_avg.png"))
plt.show()
