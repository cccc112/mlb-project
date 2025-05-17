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
base_path = "C:/Users/richc/OneDrive/桌面/專題/mlbdata/"

# 新的三個資料來源
combined_path = "OPS/mlb_player_stats_10years_combined.csv"
combined_w_path = "W/mlb_player_stats_10years_combined-W.csv"
combined_era_path = "ERA/mlb_player_stats_10years_combined-era.csv"

# 讀取資料
df_combined = pd.read_csv(combined_path)
df_w = pd.read_csv(combined_w_path)
df_era = pd.read_csv(combined_era_path)

# 清理欄位名稱與隊名格式
for df in [df_combined, df_w, df_era]:
    df.columns = df.columns.str.strip()
    df['Team'] = df['Team'].str.strip().str.upper()

# 合併三張表（使用 Team 為 key）
merged = df_combined.merge(df_w, on='Team', how='inner', suffixes=('', '_W'))
merged = merged.merge(df_era, on='Team', how='inner', suffixes=('', '_ERA'))

# 儲存合併後資料
merged.to_csv(os.path.join(base_path, "merged_output_3sources.csv"), index=False)
print(f"合併成功，共 {len(merged)} 筆資料")

# ---------- 熱力圖分析 ----------
# 假設 'Win Pct' 是最終我們要預測的目標
winrate_col = 'Win Pct'
exclude_cols = ['Wins', 'Losses', 'Win Pct']
exclude_cols = [col for col in exclude_cols if col in merged.columns]

# 計算相關係數
numeric_cols = merged.select_dtypes(include='number').columns.difference(exclude_cols)
correlation = merged[numeric_cols.tolist() + [winrate_col]].corr()
win_corr = correlation[[winrate_col]].drop(index=exclude_cols, errors='ignore')
win_corr = win_corr.sort_values(by=winrate_col, ascending=False)

# 畫出熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(win_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation with Win Rate (Excluding Wins/Losses)")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "heatmap_3sources.png"))
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

# 特徵重要性
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
sorted_idx = importances.argsort()
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Selected Features)")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "rf_feature_importance_3sources.png"))
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
plt.savefig(os.path.join(base_path, "decision_tree_3sources.png"))
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
plt.savefig(os.path.join(base_path, "rf_vs_true_3sources.png"))
plt.show()
