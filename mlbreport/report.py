import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats # 引入 scipy.stats 用於斯皮爾曼和肯德爾相關係數
from sklearn.feature_selection import RFECV # 引入 RFECV

# ======= 資料讀取與前處理 =======
merged = pd.read_csv("C:/Users/richc/OneDrive/桌面/專題/mlbdata/merged_output_team_year_avg_standardized.csv")
target_col = 'Win_Pct'
# 排除目標欄位、Wins, Losses, W, L, OPS, ERA 等欄位，這些通常不作為獨立特徵或已有其他意義
# 這裡明確確保排除了 'Wins', 'Losses', 'W', 'L' 以及新提到的相關變數
exclude_cols_initial = [
    'Wins', 'Losses', 'W', 'L', target_col,
    'Total_WinDiff', 'Home_Losses', 'Home_Wins', 'Away_Losses', 'Home_WinDiff', 'Away_Wins', 'Away_WinDiff'
]

# 將 OPS 和 ERA 從初始排除列表移開，以便在特徵工程區塊中將其作為對照組處理，
# 並確保它們不會被其他特徵選擇方法錯誤地選中為候選特徵。
exclude_cols_for_candidates = [col for col in exclude_cols_initial if col in merged.columns]
# 專門排除 OPS 和 ERA，因為它們將作為對照組處理，不參與其他特徵選擇方法的候選
if 'OPS' in merged.columns:
    exclude_cols_for_candidates.append('OPS')
if 'ERA' in merged.columns:
    exclude_cols_for_candidates.append('ERA')

print(f"實際從數據中排除的初始欄位 (不含 OPS, ERA，因為它們將作為對照組處理): {exclude_cols_for_candidates}")

num_cols = merged.select_dtypes(include='number').columns.tolist()
# 從數字欄位中排除指定欄位，作為最終的特徵候選集
feature_candidates = [c for c in num_cols if c not in exclude_cols_for_candidates]

# 確保所有候選特徵都在 merged 資料框中 (這一步通常是多餘的，但作為雙重檢查)
feature_candidates = [f for f in feature_candidates if f in merged.columns]
print(f"特徵選擇前的最終候選特徵集 (已排除所有指定欄位及 OPS, ERA): {feature_candidates}")

# ======= 特徵工程 (新增加的區塊) =======
engineered_features = []

# 確保 OBP 和 SLG 存在於數據中才能創建 Offensive_Productivity_Index
if 'OBP' in merged.columns and 'SLG' in merged.columns:
    merged['Offensive_Productivity_Index'] = merged['OBP'] + merged['SLG']
    engineered_features.append('Offensive_Productivity_Index')
    print("已創建組合指標: Offensive_Productivity_Index (OBP + SLG)")
else:
    print("無法創建 Offensive_Productivity_Index，因為 OBP 或 SLG 不存在。")

# 確保 FIP 和 WHIP 存在於數據中才能創建 Defensive_Efficiency_Index
if 'FIP' in merged.columns and 'WHIP' in merged.columns:
    merged['Defensive_Efficiency_Index'] = merged['FIP'] + merged['WHIP']
    engineered_features.append('Defensive_Efficiency_Index')
    print("已創建組合指標: Defensive_Efficiency_Index (FIP + WHIP)")
elif 'WHIP' in merged.columns and 'ER' in merged.columns: # 新增替代方案: 如果 FIP 不存在，則嘗試使用 WHIP 和 ER
    merged['Defensive_Efficiency_Index'] = merged['WHIP'] + merged['ER']
    engineered_features.append('Defensive_Efficiency_Index')
    print("已創建組合指標: Defensive_Efficiency_Index (WHIP + ER, 作為 FIP 不存在時的替代)")
else:
    print("無法創建 Defensive_Efficiency_Index，因為 FIP, WHIP 或 ER 不存在。")

if engineered_features:
    print(f"最終合併使用的變數 (組合指標): {engineered_features}")
else:
    print("沒有創建任何組合指標。")


# 將新創建的組合指標添加到 feature_candidates 中，讓它們也能參與後續的 RFECV 等選擇
feature_candidates.extend(engineered_features)
# 去重以防萬一
feature_candidates = list(set(feature_candidates))


# ======= 建模函式 =======
# 將建模函式移到前面，以便在特徵選擇時使用
def run_model(X, y):
    # 檢查數據量是否足夠進行訓練和測試分割
    if len(X) < 2 or len(y) < 2:
        return {'Random Forest': {'MSE': np.nan, 'R2': np.nan},
                'Decision Tree': {'MSE': np.nan, 'R2': np.nan}}, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    dt = DecisionTreeRegressor(random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    dt_pred = dt.predict(X_test)
    results = {
        'Random Forest': {'MSE': mean_squared_error(y_test, rf_pred), 'R2': r2_score(y_test, rf_pred)},
        'Decision Tree': {'MSE': np.nan, 'R2': np.nan} # Assuming Decision Tree is not the focus for comparison with RFECV etc.
    }
    return results, rf, dt # 返回訓練好的模型以便獲取特徵重要性


# ======= 定義特徵選擇閾值 =======
importance_threshold = 0.01  # 隨機森林特徵重要性閾值
correlation_threshold = 0.5  # 皮爾森、斯皮爾曼、肯德爾相關係數絕對值閾值

# ======= 各種特徵選擇方法 =======
feature_sets_for_comparison = {}
primary_selected_features = [] # 用於單一球隊、MLB總體分析及決策樹繪圖的特徵集

# --- 1. 隨機森林特徵重要性選擇 ---
print("\n--- 隨機森林特徵重要性選擇 ---")
# 將 X_all_features 和 y_target 定義在此處，確保它們在判斷前已被初始化
X_all_features = merged[feature_candidates]
y_target = merged[target_col]

if not X_all_features.empty and not y_target.empty and len(X_all_features) > 1:
    _, rf_model_for_importance, _ = run_model(X_all_features, y_target)
    if rf_model_for_importance:
        feature_importances = pd.Series(rf_model_for_importance.feature_importances_, index=feature_candidates)
        selected_features_rf = feature_importances[feature_importances >= importance_threshold].index.tolist()
        # 確保選出的特徵不包含原本應該被排除的
        selected_features_rf = [f for f in selected_features_rf if f not in exclude_cols_initial]
        if not selected_features_rf:
            print("隨機森林：沒有特徵達到重要性閾值，回退到預設的 OPS 和 ERA。")
            selected_features_rf = [col for col in ['OPS', 'ERA'] if col in merged.columns]
        else:
            print(f"隨機森林選擇的特徵：{selected_features_rf}")
    else:
        print("隨機森林：模型訓練失敗，回退到預設的 OPS 和 ERA。")
        selected_features_rf = [col for col in ['OPS', 'ERA'] if col in merged.columns]
else:
    print("隨機森林：資料量不足以訓練模型，回退到預設的 OPS 和 ERA。")
    selected_features_rf = [col for col in ['OPS', 'ERA'] if col in merged.columns]

feature_sets_for_comparison['Random Forest Importance'] = selected_features_rf
primary_selected_features = selected_features_rf # 將隨機森林的選擇結果作為主要特徵集

# --- 2. 皮爾森相關係數選擇 ---
print("\n--- 皮爾森相關係數選擇 ---")
if not merged[feature_candidates + [target_col]].empty:
    pearson_corr = merged[feature_candidates + [target_col]].corr(method='pearson')[target_col].drop(target_col).abs()
    selected_features_pearson = pearson_corr[pearson_corr >= correlation_threshold].index.tolist()
    selected_features_pearson = [f for f in selected_features_pearson if f not in exclude_cols_initial]
    if not selected_features_pearson:
        print("皮爾森：沒有特徵達到相關性閾值，回退到預設的 OPS 和 ERA。")
        selected_features_pearson = [col for col in ['OPS', 'ERA'] if col in merged.columns]
    else:
        print(f"皮爾森選擇的特徵：{selected_features_pearson}")
else:
    print("皮爾森：資料集為空，回退到預設的 OPS 和 ERA。")
    selected_features_pearson = [col for col in ['OPS', 'ERA'] if col in merged.columns]

feature_sets_for_comparison['Pearson Correlation'] = selected_features_pearson

# --- 3. 斯皮爾曼相關係數選擇 ---
print("\n--- 斯皮爾曼相關係數選擇 ---")
if not merged[feature_candidates + [target_col]].empty:
    spearman_corr = merged[feature_candidates + [target_col]].corr(method='spearman')[target_col].drop(target_col).abs()
    selected_features_spearman = spearman_corr[spearman_corr >= correlation_threshold].index.tolist()
    selected_features_spearman = [f for f in selected_features_spearman if f not in exclude_cols_initial]
    if not selected_features_spearman:
        print("斯皮爾曼：沒有特徵達到相關性閾值，回退到預設的 OPS 和 ERA。")
        selected_features_spearman = [col for col in ['OPS', 'ERA'] if col in merged.columns]
    else:
        print(f"斯皮爾曼選擇的特徵：{selected_features_spearman}")
else:
    print("斯皮爾曼：資料集為空，回退到預設的 OPS 和 ERA。")
    selected_features_spearman = [col for col in ['OPS', 'ERA'] if col in merged.columns]

feature_sets_for_comparison['Spearman Correlation'] = selected_features_spearman

# --- 4. 肯德爾相關係數選擇 ---
print("\n--- 肯德爾相關係數選擇 ---")
if not merged[feature_candidates + [target_col]].empty:
    kendall_corr = merged[feature_candidates + [target_col]].corr(method='kendall')[target_col].drop(target_col).abs()
    selected_features_kendall = kendall_corr[kendall_corr >= correlation_threshold].index.tolist()
    selected_features_kendall = [f for f in selected_features_kendall if f not in exclude_cols_initial]
    if not selected_features_kendall:
        print("肯德爾：沒有特徵達到相關性閾值，回退到預設的 OPS 和 ERA。")
        selected_features_kendall = [col for col in ['OPS', 'ERA'] if col in merged.columns]
    else:
        print(f"肯德爾選擇的特徵：{selected_features_kendall}")
else:
    print("肯德爾：資料集為空，回退到預設的 OPS 和 ERA。")
    selected_features_kendall = [col for col in ['OPS', 'ERA'] if col in merged.columns]

feature_sets_for_comparison['Kendall Correlation'] = selected_features_kendall


# --- 5. RFECV 遞歸特徵消除與交叉驗證 ---
print("\n--- RFECV 遞歸特徵消除與交叉驗證 ---")
selected_features_rfecv = []
if not X_all_features.empty and not y_target.empty and len(X_all_features) > 1:
    try:
        estimator = RandomForestRegressor(random_state=42)
        # 使用 R-squared 作為評分指標，CV 為 5 折交叉驗證
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='r2', n_jobs=-1)
        rfecv.fit(X_all_features, y_target)

        selected_features_rfecv = X_all_features.columns[rfecv.support_].tolist()
        # 確保選出的特徵不包含原本應該被排除的
        selected_features_rfecv = [f for f in selected_features_rfecv if f not in exclude_cols_initial]
        print(f"RFECV 選擇的最佳特徵數量: {rfecv.n_features_}")
        print(f"RFECV 選擇的特徵: {selected_features_rfecv}")

        # 繪製 RFECV 評分曲線
        plt.figure(figsize=(10, 6))
        plt.title('RFECV: Optimal Number of Features')
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (R2)")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.tight_layout()
        rfecv_plot_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/rfecv_score_plot.png"
        plt.savefig(rfecv_plot_path, dpi=300)
        plt.show()
        print(f"RFECV 評分曲線已儲存至 {rfecv_plot_path}")

    except Exception as e:
        print(f"RFECV 運行時發生錯誤: {e}，回退到預設的 OPS 和 ERA。")
        selected_features_rfecv = [col for col in ['OPS', 'ERA'] if col in merged.columns]
else:
    print("RFECV：資料量不足以訓練模型，回退到預設的 OPS 和 ERA。")
    selected_features_rfecv = [col for col in ['OPS', 'ERA'] if col in merged.columns]

feature_sets_for_comparison['RFECV'] = selected_features_rfecv

# --- 6. 組合指標 (Engineered Features) ---
if engineered_features:
    feature_sets_for_comparison['Engineered Features'] = engineered_features
else:
    print("沒有可用的組合指標來進行比較。")

# --- 7. OPS+ERA 對照組 ---
ops_era_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]
if ops_era_features:
    feature_sets_for_comparison['OPS+ERA (Control)'] = ops_era_features
else:
    print("OPS 或 ERA 欄位不存在，無法建立 OPS+ERA 對照組。")


# ======= 單球隊逐年資料 (使用主要特徵集，仍為 Random Forest Importance 選出的特徵) =======
teams = merged['Team'].unique()
results_single = []
for team in teams:
    df_team = merged[merged['Team'] == team]
    if len(df_team) < 5 or not primary_selected_features: # 確保有足夠的資料和已選取的特徵
        continue
    current_selected_features = [f for f in primary_selected_features if f in df_team.columns]
    if not current_selected_features:
        print(f"警告: 球隊 {team} 沒有可用的主要選取特徵，跳過。")
        continue
    X = df_team[current_selected_features]
    y = df_team[target_col]
    res, _, _ = run_model(X, y)
    for model in res:
        results_single.append({'Team': team, 'Model': model, 'MSE': res[model]['MSE'], 'R2': res[model]['R2']})
df_single = pd.DataFrame(results_single)


# ======= 比較不同特徵選擇方法在單球隊10年平均資料上的表現 =======
all_comparison_results = []

# 初始化 res_10y, rf_10y, dt_10y 以防 'Random Forest Importance' 情況未執行
res_10y = None
rf_10y = None
dt_10y = None

for method_name, current_selected_features in feature_sets_for_comparison.items():
    print(f"\n--- 評估方法: {method_name} ---")
    if not current_selected_features:
        print(f"注意: {method_name} 沒有選取到特徵，跳過模型評估。")
        continue

    # 為每個方法創建 10 年平均資料
    df_10y_method = merged.groupby('Team')[current_selected_features + [target_col]].mean().reset_index()

    # 確保選取的特徵在 df_10y_method 中存在
    valid_selected_features_method = [f for f in current_selected_features if f in df_10y_method.columns]
    if not valid_selected_features_method:
        print(f"警告: {method_name} 在 10 年平均資料中沒有有效特徵，跳過。")
        continue

    X_10y_method = df_10y_method[valid_selected_features_method]
    y_10y_method = df_10y_method[target_col]

    # 確保有足夠的數據進行模型訓練
    if len(X_10y_method) > 1 and len(y_10y_method) > 1:
        res_10y_method, rf_model_method, dt_model_method = run_model(X_10y_method, y_10y_method)
        for model_type, metrics in res_10y_method.items():
            all_comparison_results.append({
                'Feature_Set': method_name,
                'Model': model_type,
                'MSE': metrics['MSE'],
                'R2': metrics['R2']
            })
        # 為了繪製決策樹，我們需要保留其中一個模型的結果。這裡用主要特徵集的模型
        if method_name == 'Random Forest Importance': # 確保只為一個方法設置 res_10y, rf_10y, dt_10y
            res_10y = res_10y_method
            rf_10y = rf_model_method
            dt_10y = dt_model_method
    else:
        print(f"警告: {method_name} 數據量不足以訓練模型，跳過。")

df_compare = pd.DataFrame(all_comparison_results)

# ======= 全MLB逐年平均資料 (使用主要特徵集) =======
df_all_10y = merged.groupby('year')[primary_selected_features + [target_col]].mean().reset_index()
current_selected_features_all_10y = [f for f in primary_selected_features if f in df_all_10y.columns]
if not current_selected_features_all_10y:
    print("錯誤：在全MLB逐年平均資料中沒有可用的選取特徵。")
    res_all_10y = {'Random Forest': {'MSE': np.nan, 'R2': np.nan}, 'Decision Tree': {'MSE': np.nan, 'R2': np.nan}}
else:
    X_all_10y = df_all_10y[current_selected_features_all_10y]
    y_all_10y = df_all_10y[target_col]
    res_all_10y, _, _ = run_model(X_all_10y, y_all_10y)
df_all_10y_results = pd.DataFrame([{'Team': 'MLB_years', 'Model': k, 'MSE': v['MSE'], 'R2': v['R2']} for k, v in res_all_10y.items()])

# ======= 合併結果與繪圖 (此部分用於單一球隊及總體分析) =======
# 需要確保df_10y_results 和 df_all_10y_results 有被定義。
# 為了避免重複計算和複雜性，df_10y_results 和 df_all_10y_results 現在只基於 primary_selected_features
# 確保 res_10y 在此之前被賦值，如果沒有則給定默認值
if res_10y is None:
    res_10y = {'Random Forest': {'MSE': np.nan, 'R2': np.nan}, 'Decision Tree': {'MSE': np.nan, 'R2': np.nan}}
df_10y_results = pd.DataFrame([{'Team': 'All', 'Model': k, 'MSE': v['MSE'], 'R2': v['R2']} for k, v in res_10y.items()])
df_all_results = pd.concat([df_single, df_10y_results, df_all_10y_results], ignore_index=True)


plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', data=df_all_results)
plt.title("Model MSE Comparison (Across all data aggregations)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='R2', data=df_all_results)
plt.title("Model R2 Comparison (Across all data aggregations)")
plt.show()

# ======= 畫出特徵重要性熱力圖 (新增加的部分) =======
if 'feature_importances' in locals() and not feature_importances.empty:
    plt.figure(figsize=(8, len(feature_importances) * 0.6))
    sns.heatmap(
        feature_importances.sort_values(ascending=False).to_frame(),
        annot=True, cmap='viridis', fmt=".4f",
        cbar=True, yticklabels=True, xticklabels=['Importance'],
        linewidths=0.5, linecolor='gray'
    )
    plt.title("Random Forest Feature Importances")
    plt.yticks(rotation=0)
    plt.tight_layout()
    importance_heatmap_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/feature_importance_heatmap.png"
    plt.savefig(importance_heatmap_path, dpi=300)
    plt.show()
    print(f"特徵重要性熱力圖已儲存至 {importance_heatmap_path}")
else:
    print("沒有特徵重要性資料可畫圖。")

# ======= 熱力圖：與勝率最相關變數 (基於皮爾森、斯皮爾曼、肯德爾) =======
# 這裡將繪製一個包含所有三種相關係數的熱力圖，方便比較
if feature_candidates:
    # 計算所有候選特徵與目標變數的相關係數
    corr_pearson_series = merged[feature_candidates + [target_col]].corr(method='pearson')[target_col].drop(target_col)
    corr_spearman_series = merged[feature_candidates + [target_col]].corr(method='spearman')[target_col].drop(target_col)
    kendall_corr_data = [] # 儲存 Kendall 相關係數，處理可能的不適用情況
    for col in feature_candidates:
        try:
            # Kendall correlation might return NaN if there's no variance or too few unique values
            kendall_tau, _ = stats.kendalltau(merged[col], merged[target_col])
            kendall_corr_data.append(kendall_tau)
        except ValueError: # Handle cases where Kendall tau cannot be computed (e.g., all values are same)
            kendall_corr_data.append(np.nan)
    kendall_corr_series = pd.Series(kendall_corr_data, index=feature_candidates)

    # 選擇與勝率絕對相關值最高的 N 個特徵來繪圖
    # 可以選擇所有三種相關性中至少有一個達到閾值的特徵
    all_correlated_features = set()
    for s in [corr_pearson_series, corr_spearman_series, kendall_corr_series]:
        # 過濾掉 NaN 值再進行閾值判斷
        valid_s = s.dropna()
        all_correlated_features.update(valid_s[valid_s.abs() >= correlation_threshold].index.tolist())

    if all_correlated_features:
        # 將這些相關係數組合成一個 DataFrame 以便繪製熱力圖
        df_correlations = pd.DataFrame({
            'Pearson': corr_pearson_series.loc[list(all_correlated_features)],
            'Spearman': corr_spearman_series.loc[list(all_correlated_features)],
            'Kendall': kendall_corr_series.loc[list(all_correlated_features)]
        }).dropna(how='all') # 移除所有相關性都是 NaN 的行

        if not df_correlations.empty:
            plt.figure(figsize=(6, len(df_correlations) * 0.7))
            sns.heatmap(
                df_correlations.sort_values(by='Pearson', ascending=False), # 依皮爾森排序以便觀察
                annot=True, cmap='coolwarm', center=0, cbar=True,
                yticklabels=True, xticklabels=True,
                linewidths=0.5, linecolor='gray', fmt=".2f"
            )
            plt.title(f"Top Correlated Features with {target_col} (Pearson, Spearman, Kendall)")
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path_multi_corr = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/heatmap_multi_correlation.png"
            plt.savefig(save_path_multi_corr, dpi=300)
            plt.show()
            print(f"多重相關性熱力圖已儲存至 {save_path_multi_corr}")
        else:
            print("沒有足夠的特徵達到相關性閾值以繪製多重相關性熱力圖。")
    else:
        print("沒有特徵達到任何相關性閾值以繪製多重相關性熱力圖。")
else:
    print("沒有候選特徵，無法繪製相關性熱力圖。")


# ======= 模型表現比較圖 (針對特徵選擇方法) =======
if not df_compare.empty:
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='R2', hue='Model', data=df_compare)
    plt.title("R² Comparison Across Feature Selection Methods")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='MSE', hue='Model', data=df_compare)
    plt.title("MSE Comparison Across Feature Selection Methods")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可比較的模型結果資料。")


# ======= 可視化決策樹 (使用 primary_selected_features) =======
if 'dt_10y' in locals() and dt_10y and 'rf_10y' in locals() and rf_10y and primary_selected_features:
    # 確保 X_10y 被正確定義且包含 primary_selected_features
    df_10y_for_tree = merged.groupby('Team')[primary_selected_features + [target_col]].mean().reset_index()
    X_10y_for_tree = df_10y_for_tree[[f for f in primary_selected_features if f in df_10y_for_tree.columns]]

    if not X_10y_for_tree.empty:
        # 1. Decision Tree (單一)
        plt.figure(figsize=(20, 10))
        plot_tree(dt_10y, feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title("Decision Tree (max_depth=3) - Primary Features")
        plt.tight_layout()
        plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/decision_tree_10y.png", dpi=300)
        plt.close()

        plt.figure(figsize=(20, 10))
        plot_tree(rf_10y.estimators_[0], feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title("Random Forest - One Tree (max_depth=3) - Primary Features")
        plt.tight_layout()
        plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/random_forest_tree_10y.png", dpi=300)
        plt.close()
    else:
        print("無法繪製決策樹，因為用於繪圖的特徵數據為空。")
else:
    print("無法繪製決策樹，因為模型未訓練或主要選取特徵為空。")


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
    df_compare_formatted.to_excel(writer, sheet_name='Feature_Method_Compare', index=False)
    df_all_results_formatted.to_excel(writer, sheet_name='AllTeams_PrimaryFeatures', index=False)

print(f"\n✅ 結果已成功儲存為 Excel 檔：{excel_output_path}")
