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
base_path = "C:/Users/richc/OneDrive/桌面/專題/mlbreport/"
standings_path = os.path.join(base_path, "MLB_2024_standings.csv")
batters_path = os.path.join(base_path, "mlb-player-stats-Batters.csv")

# 讀取資料
standings = pd.read_csv(standings_path)
batters = pd.read_csv(batters_path)

# 清理欄位名稱
standings.columns = standings.columns.str.strip()
batters.columns = batters.columns.str.strip()

# 統一隊名格式
standings['Team'] = standings['Team'].str.strip().str.upper()
batters['Team'] = batters['Team'].str.strip().str.upper()

# 檢查 Team 欄位
if 'Team' not in standings.columns or 'Team' not in batters.columns:
    raise ValueError("Team 欄位缺失，請確認兩張表是否都有 Team 欄位")

# 找出缺失的球隊
missing_teams = set(standings['Team']) - set(batters['Team'])
if missing_teams:
    print(f"無法在打者資料中找到這些球隊：{missing_teams}")

# 計算每隊平均數據
batters_avg = batters.groupby('Team').mean(numeric_only=True).reset_index()

# 合併資料
merged = pd.merge(standings, batters_avg, on='Team', how='inner')
print(f"合併成功，共 {len(merged)} 筆資料")
merged.to_csv(os.path.join(base_path, "merged_output.csv"), index=False)

# ---------- 熱力圖（排除勝場欄位） ----------
winrate_col = 'Win Pct'
exclude_cols = ['Wins', 'Losses', 'Win Pct', winrate_col]
exclude_cols = [col for col in exclude_cols if col in merged.columns]

numeric_cols = merged.select_dtypes(include='number').columns.difference(exclude_cols)
correlation = merged[numeric_cols.tolist() + [winrate_col]].corr()
win_corr = correlation[[winrate_col]].drop(index=exclude_cols, errors='ignore')
win_corr = win_corr.sort_values(by=winrate_col, ascending=False)

# 畫出熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(win_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation with Win Rate (Excluding Wins/Losses)")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "heatmap.png"))
plt.show()

# ---------- 選擇相關係數 >= 0.5 的變數 ----------
threshold = 0.5
selected_features = win_corr[win_corr[winrate_col].abs() >= threshold].index.tolist()
print("選出的重要特徵（相關係數 >= 0.5）：")
print(selected_features)

# ---------- 隨機森林模型（只用選出來的特徵） ----------
# Label encoding for Team
label_encoder = LabelEncoder()
merged['Team'] = label_encoder.fit_transform(merged['Team'])

# Encode League if exists
if 'League' in merged.columns:
    merged['League'] = merged['League'].map({'American League': 0, 'National League': 1})

# 建立訓練資料與目標變數
X = merged[selected_features]
y = merged[winrate_col]

# 切分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 建立並訓練隨機森林模型
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
# 訓練模型
rf_model.fit(X_train, y_train)

# 預測與評估
predictions = rf_model.predict(X_test)
print("\nRandom Forest 預測結果：")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# ---------- 特徵重要性圖 ----------
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
sorted_idx = importances.argsort()
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Selected Features)")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "rf_selected_feature_importance.png"))
plt.show()

# ---------- 決策樹模型（使用相同的選定特徵） ----------

# 建立決策樹回歸模型
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# 預測測試集
tree_predictions = tree.predict(X_test)

# 預測評估
print("\nDecision Tree 預測結果：")
print("Mean Squared Error:", mean_squared_error(y_test, tree_predictions))
print("R2 Score:", r2_score(y_test, tree_predictions))

# 畫出決策樹
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree (Selected Features)")
plt.savefig(os.path.join(base_path, "decision_tree_selected_features.png"))
plt.show()

# ---------- 再次畫出隨機森林預測 vs. 真實值 ----------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, c='blue', label='Predicted', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('True Win Rate')
plt.ylabel('Predicted Win Rate')
plt.title('Random Forest: True vs Predicted Win Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path, "rf_prediction_vs_actual.png"))
plt.show()
