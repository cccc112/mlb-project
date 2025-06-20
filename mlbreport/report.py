import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, KFold # 引入 KFold 用於 RFECV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import scipy.stats as stats
from sklearn.feature_selection import RFECV # 引入 RFECV
from sklearn.linear_model import LogisticRegression
import os

# 設定路徑
base_path_data = "C:/Users/richc/OneDrive/桌面/專題/mlbdata"
base_path_report = "C:/Users/richc/OneDrive/桌面/專題/mlbreport"

# ======= Prepare Feature Sets for Comparison =======
# 已將初始化移到最頂部，確保在任何地方使用前已定義
feature_sets_for_comparison = {} 

# ======= Data Loading and Preprocessing =======
# 假設這些檔案已經存在，如果不存在，請確保它們的路徑正確
try:
    # 讀取資料
    df_combined = pd.read_csv(os.path.join(base_path_data, "OPS", "mlb_player_stats_10years_combined.csv"))
    df_w = pd.read_csv(os.path.join(base_path_data, "W", "mlb_player_stats_10years_combined-W.csv"))
    df_era = pd.read_csv(os.path.join(base_path_data, "ERA", "mlb_player_stats_10years_combined-era.csv"))

    print("ERA columns:", df_era.columns)
    print("OPS columns:", df_combined.columns)
    print("勝率 columns:", df_w.columns)

    # 統一格式：清除欄位名稱空白，將 Team 轉為大寫，年份轉為整數
    for df in [df_combined, df_w, df_era]:
        df.columns = df.columns.str.strip()
        df['Team'] = df['Team'].str.strip().str.upper()
        df['year'] = df['year'].astype(int)

    # 平均每隊每年球員數據：計算每支球隊每年各項數值的平均值
    df_combined_grouped = df_combined.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()
    df_era_grouped = df_era.groupby(['Team', 'year']).mean(numeric_only=True).reset_index()

    # 勝率資料保留 Team 和 year 當 key (確保 Team 和 year 的格式一致)
    df_w_grouped = df_w.copy()
    df_w_grouped['year'] = df_w_grouped['year'].astype(int)
    df_w_grouped['Team'] = df_w_grouped['Team'].str.strip().str.upper()

    # 合併資料：根據 Team 和 year 合併 OPS、勝率和 ERA 數據
    merged = df_combined_grouped.merge(df_w_grouped, on=['Team', 'year'], how='inner', suffixes=('', '_W'))
    merged = merged.merge(df_era_grouped, on=['Team', 'year'], how='inner', suffixes=('', '_ERA'))

    # 儲存合併結果到 CSV 檔案
    merged_output_path = os.path.join(base_path_data, "merged_output_team_year_avg.csv")
    merged.to_csv(merged_output_path, index=False)
    print(f"合併成功，共 {len(merged)} 筆資料。儲存至 {merged_output_path}")

    # 現在重新讀取合併後的資料，確保所有欄位都在同一個 DataFrame 中
    # 並用於後續的標準化或其他處理
    merged = pd.read_csv(merged_output_path)
    print("重新讀取合併後的資料。")

except FileNotFoundError as e:
    print(f"錯誤: 找不到資料檔案。請檢查路徑和檔案是否存在: {e}")
    # 為了讓程式碼能繼續運行，創建一個空的 DataFrame (或根據實際情況處理)
    merged = pd.DataFrame()
    print("已創建空的 DataFrame，請檢查檔案路徑。")

# 檢查 merged DataFrame 是否為空
if merged.empty:
    print("錯誤: 合併後的資料為空，無法進行後續分析。請檢查資料載入過程。")
    exit() # 如果資料載入失敗，直接退出程式

target_col = 'Win_Pct' # 原始連續型目標變數
# 創建二元分類目標變數：Win_Pct > 0.5 為 1 (勝), 否則為 0 (負)
merged['Win_Pct_Binary'] = (merged[target_col] > 0.5).astype(int)
binary_target_col = 'Win_Pct_Binary'

# 排除目標欄位、勝場、敗場、W、L 以及其他可能導致數據洩漏的相關變數
exclude_cols_initial = [
    'Wins', 'Losses', 'W', 'L', target_col, binary_target_col,
    'Total_WinDiff', 'Home_Losses', 'Home_Wins', 'Away_Losses', 'Home_WinDiff', 'Away_Wins', 'Away_WinDiff'
]

# 過濾出實際存在於 merged.columns 中的排除欄位
exclude_cols_for_candidates = [col for col in exclude_cols_initial if col in merged.columns]

print(f"實際從數據中排除的初始欄位: {exclude_cols_for_candidates}")

num_cols = merged.select_dtypes(include='number').columns.tolist()
# 從數值欄位中過濾掉排除的欄位，得到用於其他選擇方法的最終特徵候選集
feature_candidates_for_general_selection = [c for c in num_cols if c not in exclude_cols_for_candidates]

# 確保所有候選特徵都存在於 merged DataFrame 中 (冗餘但有利於雙重檢查)
feature_candidates_for_general_selection = [f for f in feature_candidates_for_general_selection if f in merged.columns]
print(f"特徵選擇前的初始候選特徵集 (已排除直接結果變數): {feature_candidates_for_general_selection}")


# ======= Feature Engineering (組合指標) =======
engineered_features = []

# 如果 OBP 和 SLG 存在，則創建 Offensive_Productivity_Index
if 'OBP' in merged.columns and 'SLG' in merged.columns:
    merged['Offensive_Productivity_Index'] = merged['OBP'] + merged['SLG']
    engineered_features.append('Offensive_Productivity_Index')
    print("已創建組合指標: Offensive_Productivity_Index (OBP + SLG)")
else:
    print("無法創建 Offensive_Productivity_Index，因為 OBP 或 SLG 不存在。")

# 如果 FIP 和 WHIP 存在，則創建 Defensive_Efficiency_Index，否則回溯到 WHIP 和 ER
if 'FIP' in merged.columns and 'WHIP' in merged.columns:
    merged['Defensive_Efficiency_Index'] = merged['FIP'] + merged['WHIP']
    engineered_features.append('Defensive_Efficiency_Index')
    print("已創建組合指標: Defensive_Efficiency_Index (FIP + WHIP)")
elif 'WHIP' in merged.columns and 'ER' in merged.columns:
    merged['Defensive_Efficiency_Index'] = merged['WHIP'] + merged['ER']
    engineered_features.append('Defensive_Efficiency_Index')
    print("已創建組合指標: Defensive_Efficiency_Index (WHIP + ER, 作為 FIP 不存在時的替代)")
else:
    print("無法創建 Defensive_Efficiency_Index，因為 FIP, WHIP 或 ER 不存在。")

if engineered_features:
    print(f"最終合併使用的變數 (組合指標): {engineered_features}")
else:
    print("沒有創建任何組合指標。")

# === 加在你原本的 Feature Engineering 區塊之後 ===

# ✅ 1. 建立 RFECV 選出特徵的加權組合：BB, ER, SV
rfecv_base_features = ['BB', 'ER', 'SV']

# 確保這三個變數都存在於資料中
if all(f in merged.columns for f in rfecv_base_features):
    print("\n--- 檢查：所有 RFECV 選出的基礎變數都存在 ---")

    # ✅ 2. 使用皮爾森相關係數與 Win_Pct 的絕對值作為加權
    # 確保只計算數值列的相關性，並且處理NaN
    temp_df_for_corr = merged[rfecv_base_features + [target_col]].dropna()
    if not temp_df_for_corr.empty:
        # 將 .abs() 移除，以保留相關係數的正負號
        corr_matrix = temp_df_for_corr.corr(method='pearson')
        correlations = corr_matrix[target_col].drop(target_col)

        # 印出相關係數（即加權）
        print("使用的權重（來自皮爾森相關性）:")
        print(correlations)

        # 運用這些權重建立加權指標
        # 處理原始特徵中的 NaN 值，這裡使用 0 填充作為加權前的處理
        # 實際應用中可能需要更精細的填充策略
        merged['RFECV_Weighted_Index'] = sum([
            correlations[feat] * merged[feat].fillna(0) for feat in rfecv_base_features
        ])

        # ✅ 3. 加入到特徵比較集
        feature_sets_for_comparison['RFECV Weighted Composite'] = ['RFECV_Weighted_Index']
        # Also add to engineered_features so it's considered for interactive prediction if needed.
        # This prevents redundant code in interactive prediction logic to re-create it.
        if 'RFECV_Weighted_Index' not in engineered_features:
            engineered_features.append('RFECV_Weighted_Index') 
        print("\n✅ 已建立 'RFECV_Weighted_Index'，並加入模型比較。")
    else:
        print("\n❌ 無法計算相關係數，因為 RFECV 基礎變數或目標變數包含太多缺失值。")
else:
    print("\n❌ 無法建立 RFECV_Weighted_Index，因為 ['BB', 'ER', 'SV'] 中有變數缺失。")

# === 整合區塊結束 ===


# ======= Modeling Function - Regression Models =======
def run_regression_models(X, y_reg):
    """
    運行決策樹回歸器和隨機森林回歸器模型。
    Args:
        X (pd.DataFrame): 特徵數據。
        y_reg (pd.Series): 連續型目標變數 (Win_Pct)。
    Returns:
        tuple: 模型結果字典，訓練好的隨機森林模型，訓練好的決策樹模型。
    """
    if len(X) < 2 or len(y_reg) < 2 or X.empty or y_reg.empty:
        print("警告: 數據量不足以訓練回歸模型。")
        return {'Random Forest Regressor': {'MSE': np.nan, 'R2': np.nan},
                'Decision Tree Regressor': {'MSE': np.nan, 'R2': np.nan}}, None, None

    # 處理 NaN 值 (在訓練之前確保數據乾淨)
    X = X.fillna(X.mean(numeric_only=True))
    y_reg = y_reg.fillna(y_reg.mean())

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, random_state=42)

    # Random Forest Regressor
    # 參數解釋：
    # n_estimators: 森林中樹的數量。增加通常可提高性能但增加計算成本。
    # max_depth: 每棵樹的最大深度。限制深度可防止過擬合。
    # min_samples_split: 分裂節點所需的最少樣本數。增加可防止過擬合。
    # min_samples_leaf: 葉節點所需的最少樣本數。增加可防止過擬合。
    # max_features: 在每次分裂時考慮的特徵數量。'sqrt' 通常為 good starting point。
    # random_state: 隨機種子，確保結果可重現。
    rf_reg = RandomForestRegressor(
        n_estimators=100,      # 樹的數量，可嘗試 50-200
        max_depth=8,           # 樹的最大深度，可嘗試 5-15
        min_samples_split=5,   # 分裂節點所需的最小樣本數，可嘗試 2-10
        min_samples_leaf=2,    # 葉節點所需的最小樣本數，可嘗試 1-5
        max_features='sqrt',   # 每次分裂考慮的特徵數，可嘗試 'log2' 或 0.5 (比例)
        random_state=42
    )
    rf_reg.fit(X_train, y_train_reg)
    rf_reg_pred = rf_reg.predict(X_test)
    rf_reg_mse = mean_squared_error(y_test_reg, rf_reg_pred)
    rf_reg_r2 = r2_score(y_test_reg, rf_reg_pred)

    # Decision Tree Regressor
    # 參數解釋：
    # max_depth: 樹的最大深度。這是防止過擬合最關鍵的參數。
    # min_samples_split: 分裂節點所需的最少樣本數。
    # min_samples_leaf: 葉節點所需的最少樣本數。
    # random_state: 隨機種子。
    dt_reg = DecisionTreeRegressor(
        max_depth=5,           # 可嘗試 3-10
        min_samples_split=2,   # 可嘗試 2-5
        min_samples_leaf=1,    # 可嘗試 1-3
        random_state=42
    )
    dt_reg.fit(X_train, y_train_reg)
    dt_reg_pred = dt_reg.predict(X_test)
    dt_reg_mse = mean_squared_error(y_test_reg, dt_reg_pred)
    dt_reg_r2 = r2_score(y_test_reg, dt_reg_pred)

    results = {
        'Random Forest Regressor': {'MSE': rf_reg_mse, 'R2': rf_reg_r2},
        'Decision Tree Regressor': {'MSE': dt_reg_mse, 'R2': dt_reg_r2}
    }
    return results, rf_reg, dt_reg

# ======= Modeling Function - Classification Model =======
def run_classification_model(X, y_binary):
    """
    運行羅吉斯回歸分類器模型。
    Args:
        X (pd.DataFrame): 特徵數據。
        y_binary (pd.Series): 二元目標變數 (Win_Pct_Binary)。
    Returns:
        tuple: 模型結果字典，訓練好的羅吉斯回歸模型。
    """
    if len(X) < 2 or len(y_binary) < 2 or X.empty or y_binary.empty:
        print("警告: 數據量不足以訓練分類模型。")
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    # 處理 NaN 值 (在訓練之前確保數據乾淨)
    X = X.fillna(X.mean(numeric_only=True))
    y_binary = y_binary.fillna(y_binary.mode()[0] if not y_binary.empty else 0) # 用眾數填充，如果為空則用 0

    if len(y_binary.unique()) < 2:
        print("警告: 目標變數的類別不足，無法進行分類。")
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    X_train, X_test, y_train_binary, y_test_binary = train_test_split(X, y_binary, random_state=42)

    if len(y_train_binary.unique()) < 2:
        print("警告: 訓練集中目標變數的類別不足，無法進行分類。")
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    lr_clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=200) # max_iter 可調整以確保收斂
    try:
        lr_clf.fit(X_train, y_train_binary)
        lr_clf_pred = lr_clf.predict(X_test)
        lr_clf_accuracy = accuracy_score(y_test_binary, lr_clf_pred)
        lr_clf_f1 = f1_score(y_test_binary, lr_clf_pred, zero_division=0)

        results = {
            'Logistic Regression Classifier': {'Accuracy': lr_clf_accuracy, 'F1-Score': lr_clf_f1}
        }
        return results, lr_clf
    except Exception as e:
        print(f"羅吉斯回歸訓練失敗: {e}")
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None


# ======= Define Feature Selection Thresholds (for reference, not main comparison logic) =======
importance_threshold = 0.01
correlation_threshold = 0.5


# 定義備用特徵集，如果其他方法產生空結果時使用。
# 這將是 'Engineered Features' (如果可用)，否則為 OPS+ERA。
# 這確保在方法未能選取特徵時始終使用一個非空預設集。
fallback_features = engineered_features if engineered_features else ([col for col in ['OPS', 'ERA'] if col in merged.columns])
if not fallback_features:
    print("警告: 沒有可用的備用特徵集 (Engineered Features 或 OPS+ERA 都不存在)。某些特徵選擇方法可能無法運行。")

# 1. Random Forest Importance (fallback to engineered_features)
print("\n--- 隨機森林特徵重要性選擇 (基於連續型目標) ---")
X_all_features = merged[feature_candidates_for_general_selection]
y_target_continuous = merged[target_col]

selected_features_rf_importance = []
if not X_all_features.empty and not y_target_continuous.empty and len(X_all_features) > 1:
    _, rf_model_for_importance, _ = run_regression_models(X_all_features, y_target_continuous)
    if rf_model_for_importance:
        feature_importances = pd.Series(rf_model_for_importance.feature_importances_, index=feature_candidates_for_general_selection)
        selected_features_rf_importance = feature_importances[feature_importances >= importance_threshold].index.tolist()
        # 確保選出的特徵不包含原本應該被排除的
        selected_features_rf_importance = [f for f in selected_features_rf_importance if f not in exclude_cols_initial]
        if not selected_features_rf_importance:
            print("隨機森林：沒有特徵達到重要性閾值。將使用備用特徵集。")
            selected_features_rf_importance = fallback_features # Fallback
        else:
            print(f"隨機森林選擇的特徵：{selected_features_rf_importance}")
    else:
        print("隨機森林：模型訓練失敗。將使用備用特徵集。")
        selected_features_rf_importance = fallback_features # Fallback
else:
    print("隨機森林：資料量不足以訓練模型。將使用備用特徵集。")
    selected_features_rf_importance = fallback_features # Fallback
feature_sets_for_comparison['Random Forest Importance'] = selected_features_rf_importance

# 2. Pearson Correlation (fallback to engineered_features)
print("\n--- 皮爾森相關係數選擇 (基於連續型目標) ---")
selected_features_pearson = []
# 檢查 df_temp 是否存在且非空，以避免錯誤
df_temp = merged[feature_candidates_for_general_selection + [target_col]]
if not df_temp.empty:
    pearson_corr = df_temp.corr(method='pearson')[target_col].drop(target_col).abs()
    selected_features_pearson = pearson_corr[pearson_corr >= correlation_threshold].index.tolist()
    selected_features_pearson = [f for f in selected_features_pearson if f not in exclude_cols_initial]
    if not selected_features_pearson:
        print("皮爾森：沒有特徵達到相關性閾值。將使用備用特徵集。")
        selected_features_pearson = fallback_features # Fallback
    else:
        print(f"皮爾森選擇的特徵：{selected_features_pearson}")
else:
    print("皮爾森：資料集為空。將使用備用特徵集。")
    selected_features_pearson = fallback_features # Fallback
feature_sets_for_comparison['Pearson Correlation'] = selected_features_pearson

# 3. Spearman Correlation (fallback to engineered_features)
print("\n--- 斯皮爾曼相關係數選擇 (基於連續型目標) ---")
selected_features_spearman = []
df_temp = merged[feature_candidates_for_general_selection + [target_col]]
if not df_temp.empty:
    spearman_corr = df_temp.corr(method='spearman')[target_col].drop(target_col).abs()
    selected_features_spearman = spearman_corr[spearman_corr >= correlation_threshold].index.tolist()
    selected_features_spearman = [f for f in selected_features_spearman if f not in exclude_cols_initial]
    if not selected_features_spearman:
        print("斯皮爾曼：沒有特徵達到相關性閾值。將使用備用特徵集。")
        selected_features_spearman = fallback_features # Fallback
    else:
        print(f"斯皮爾曼選擇的特徵：{selected_features_spearman}")
else:
    print("斯皮爾曼：資料集為空。將使用備用特徵集。")
    selected_features_spearman = fallback_features # Fallback
feature_sets_for_comparison['Spearman Correlation'] = selected_features_spearman

# 4. Kendall Correlation (fallback to engineered_features)
print("\n--- 肯德爾相關係數選擇 (基於連續型目標) ---")
selected_features_kendall = []
df_temp = merged[feature_candidates_for_general_selection + [target_col]]
if not df_temp.empty:
    kendall_corr = df_temp.corr(method='kendall')[target_col].drop(target_col).abs()
    selected_features_kendall = kendall_corr[kendall_corr >= correlation_threshold].index.tolist()
    selected_features_kendall = [f for f in selected_features_kendall if f not in exclude_cols_initial]
    if not selected_features_kendall:
        print("肯德爾：沒有特徵達到相關性閾值。將使用備用特徵集。")
        selected_features_kendall = fallback_features # Fallback
    else:
        print(f"肯德爾選擇的特徵：{selected_features_kendall}")
else:
    print("肯德爾：資料集為空。將使用備用特徵集。")
    selected_features_kendall = fallback_features # Fallback
feature_sets_for_comparison['Kendall Correlation'] = selected_features_kendall

# 5. RFECV (遞歸特徵消除與交叉驗證)
print("\n--- RFECV 遞歸特徵消除與交叉驗證 ---")
selected_features_rfecv = []
if not X_all_features.empty and not y_target_continuous.empty and len(X_all_features) > 1:
    try:
        estimator = RandomForestRegressor(random_state=42)
        # 使用 R-squared 作為評分指標，5 折交叉驗證
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
        rfecv.fit(X_all_features, y_target_continuous)

        selected_features_rfecv = X_all_features.columns[rfecv.support_].tolist()
        # 確保選出的特徵不包含那些最初被排除的特徵
        selected_features_rfecv = [f for f in selected_features_rfecv if f not in exclude_cols_initial]

        if not selected_features_rfecv:
            print("RFECV：沒有特徵被選中。將使用備用特徵集。")
            selected_features_rfecv = fallback_features # Fallback
        else:
            print(f"RFECV 選擇的最佳特徵數量: {len(selected_features_rfecv)}") # 使用實際選取特徵的數量
            print(f"RFECV **自動選出**的特徵組合: {selected_features_rfecv}") # 明確印出自動選出的組合

            # 繪製 RFECV 評分曲線
            plt.figure(figsize=(10, 6))
            plt.title('RFECV: Optimal Number of Features')
            plt.xlabel("Number of Features Selected") # Changed to English
            plt.ylabel("Cross-validation score (R2)") # Changed to English
            plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
            plt.tight_layout()
            rfecv_plot_path = os.path.join(base_path_report, "rfecv_score_plot.png")
            plt.savefig(rfecv_plot_path, dpi=300)
            plt.show()
            print(f"RFECV 評分曲線已儲存至 {rfecv_plot_path}")

    except Exception as e:
        print(f"RFECV 運行時發生錯誤: {e}。將使用備用特徵集。")
        selected_features_rfecv = fallback_features # Fallback
else:
    print("RFECV：資料量不足以訓練模型。將使用備用特徵集。")
    selected_features_rfecv = fallback_features # Fallback
feature_sets_for_comparison['RFECV'] = selected_features_rfecv

# --- 新增邏輯：將 RFECV 選出的數值特徵組合成一個單一特徵 ---
rfecv_combined_feature_name = 'RFECV_Combined_Feature'
if selected_features_rfecv: # 確保有選出特徵
    # 只包含 RFECV 選出的數值特徵來進行求和
    numeric_rfecv_selected_features = [f for f in selected_features_rfecv if f in merged.select_dtypes(include=np.number).columns]
    
    if numeric_rfecv_selected_features:
        # 處理 NaN 值，只對選出的特徵進行求和
        temp_df_for_combine = merged[numeric_rfecv_selected_features].fillna(merged[numeric_rfecv_selected_features].mean(numeric_only=True))
        merged[rfecv_combined_feature_name] = temp_df_for_combine.sum(axis=1)
        print(f"\n已創建新的單一特徵 '{rfecv_combined_feature_name}'，由 {numeric_rfecv_selected_features} (RFECV選出的數值特徵) 組合而成。")
        feature_sets_for_comparison['RFECV Combined Single Feature'] = [rfecv_combined_feature_name]
    else:
        print("\nRFECV 沒有選出任何數值特徵，無法創建 'RFECV_Combined_Feature'。")
else:
    print("\nRFECV 沒有選出特徵，無法創建 'RFECV_Combined_Feature'。")

# 6. Engineered Features (如果可用則始終包含)
if engineered_features:
    feature_sets_for_comparison['Engineered Features'] = engineered_features
    print(f"\n用於比較: 'Engineered Features' -> {engineered_features}")
else:
    print("\n沒有可用的組合指標來進行比較。")

# 7. OPS+ERA 對照組 (如果可用則始終包含)
ops_era_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]
if ops_era_features:
    feature_sets_for_comparison['OPS+ERA (Control)'] = ops_era_features
    print(f"用於比較: 'OPS+ERA (Control)' -> {ops_era_features}")
else:
    print("OPS 或 ERA 欄位不存在，無法建立 OPS+ERA 對照組。")


# 初始化 rf_10y, dt_10y 用於決策樹可視化
rf_10y = None
dt_10y = None

# 設定 primary_selected_features 用於決策樹可視化
# 這將優先使用 Engineered Features，然後在 Engineered Features 為空時回溯到 OPS+ERA
primary_selected_features = engineered_features if engineered_features else ops_era_features
if not primary_selected_features:
    print("警告: 沒有主要特徵被選中用於一般分析或決策樹可視化。")


# ======= Single Team Yearly Data Analysis (現在僅針對指定比較項目) =======
teams = merged['Team'].unique()
results_single_regression = []
results_single_classification = []

for team in teams:
    df_team = merged[merged['Team'] == team]

    # 為 feature_sets_for_comparison 中的所有特徵集運行模型
    for fs_name, current_fs_features in feature_sets_for_comparison.items():
        valid_fs_features = [f for f in current_fs_features if f in df_team.columns]
        if len(df_team) >= 5 and valid_fs_features:
            X_fs = df_team[valid_fs_features]
            y_reg = df_team[target_col]
            y_binary = df_team[binary_target_col]

            # 回歸模型
            res_reg, _, _ = run_regression_models(X_fs, y_reg)
            for model_name, metrics in res_reg.items():
                results_single_regression.append({'Team': team, 'Feature_Set': fs_name, 'Model': model_name, 'MSE': metrics['MSE'], 'R2': metrics['R2']})

            # 分類模型
            if len(y_binary.unique()) > 1:
                res_clf, _ = run_classification_model(X_fs, y_binary)
                for model_name, metrics in res_clf.items():
                    results_single_classification.append({'Team': team, 'Feature_Set': fs_name, 'Model': model_name, 'Accuracy': metrics['Accuracy'], 'F1-Score': metrics['F1-Score']})
            else:
                # 如果該球隊/特徵集無法進行分類，則追加 NaN 結果
                results_single_classification.append({'Team': team, 'Feature_Set': fs_name, 'Model': 'Logistic Regression Classifier', 'Accuracy': np.nan, 'F1-Score': np.nan})
        # else:
            # print(f"警告: 球隊 {team} ({fs_name}) 的數據量不足或無有效特徵，跳過模型訓練。")

df_single_regression = pd.DataFrame(results_single_regression)
df_single_classification = pd.DataFrame(results_single_classification)


# ======= Compare Different Feature Selection Methods on 10-Year Average Data =======
all_regression_comparison_results = []
all_classification_comparison_results = []

for method_name, current_selected_features in feature_sets_for_comparison.items():
    print(f"\n--- 評估方法 (10年平均): {method_name} ---")
    if not current_selected_features:
        print(f"注意: {method_name} 沒有選取到特徵，跳過模型評估。")
        continue

    # 為每種方法創建 10 年平均數據
    df_10y_method = merged.groupby('Team')[current_selected_features + [target_col, binary_target_col]].mean(numeric_only=True).reset_index()

    # 確保選取的特徵存在於 df_10y_method 中
    valid_selected_features_method = [f for f in current_selected_features if f in df_10y_method.columns]
    if not valid_selected_features_method:
        print(f"警告: {method_name} 在 10 年平均資料中沒有有效特徵，跳過。")
        continue

    X_10y_method = df_10y_method[valid_selected_features_method]
    y_10y_reg = df_10y_method[target_col]
    y_10y_binary = (df_10y_method[target_col] > 0.5).astype(int)

    # 運行回歸模型
    if len(X_10y_method) > 1 and len(y_10y_reg) > 1:
        res_reg_method, rf_model_for_tree, dt_model_for_tree = run_regression_models(X_10y_method, y_10y_reg)
        for model_type, metrics in res_reg.items():
            all_regression_comparison_results.append({
                'Feature_Set': method_name,
                'Model': model_type,
                'MSE': metrics['MSE'],
                'R2': metrics['R2']
            })
        # 對於決策樹可視化，我們將保留來自 'Engineered Features' 或 'RFECV' 或 RFECV Combined 的模型結果
        if method_name == 'Engineered Features' or method_name == 'RFECV' or method_name == 'RFECV Combined Single Feature':
            rf_10y = rf_model_for_tree
            dt_10y = dt_model_for_tree
    else:
        print(f"警告: {method_name} 數據量不足以訓練回歸模型，跳過。")

    # 運行分類模型
    if len(X_10y_method) > 1 and len(y_10y_binary) > 1 and len(y_10y_binary.unique()) > 1:
        res_clf_method, _ = run_classification_model(X_10y_method, y_10y_binary)
        for model_type, metrics in res_clf_method.items():
            all_classification_comparison_results.append({
                'Feature_Set': method_name,
                'Model': model_type,
                'Accuracy': metrics['Accuracy'],
                'F1-Score': metrics['F1-Score']
            })
    else:
        print(f"警告: {method_name} 數據量或類別不足以訓練分類模型，跳過。")

df_regression_compare = pd.DataFrame(all_regression_comparison_results)
df_classification_compare = pd.DataFrame(all_classification_comparison_results)


# ======= Total MLB Yearly Average Data (現在僅針對指定比較項目) =======
# 定義要平均的欄位：所有數值特徵和目標欄位
cols_for_yearly_average = feature_candidates_for_general_selection + engineered_features + ops_era_features + [rfecv_combined_feature_name] + [target_col, binary_target_col]
# 確保唯一性並存在於合併欄位中，然後再平均
cols_for_yearly_average = list(set([col for col in cols_for_yearly_average if col in merged.columns]))

# 僅對這些數值欄位進行 groupby 和 mean 操作
df_all_10y = merged[cols_for_yearly_average].groupby('year').mean(numeric_only=True).reset_index()


results_all_10y_regression = []
results_all_10y_classification = []

for method_name, current_selected_features in feature_sets_for_comparison.items():
    print(f"\n--- 評估方法 (全MLB逐年平均): {method_name} ---")
    if not current_selected_features:
        print(f"注意: {method_name} 沒有選取到特徵，跳過模型評估。")
        continue

    valid_selected_features_all_10y = [f for f in current_selected_features if f in df_all_10y.columns]
    if not valid_selected_features_all_10y:
        print(f"警告: {method_name} 在全MLB逐年平均資料中沒有可用的選取特徵，跳過。")
        continue

    X_all_10y = df_all_10y[valid_selected_features_all_10y]
    y_all_10y_reg = df_all_10y[target_col]
    y_all_10y_binary = (df_all_10y[target_col] > 0.5).astype(int)

    # 運行回歸模型
    if len(X_all_10y) > 1 and len(y_all_10y_reg) > 1:
        res_reg_all_10y, _, _ = run_regression_models(X_all_10y, y_all_10y_reg)
        for model_type, metrics in res_reg_all_10y.items():
            results_all_10y_regression.append({'Team': 'MLB_years', 'Feature_Set': method_name, 'Model': model_type, 'MSE': metrics['MSE'], 'R2': metrics['R2']})
    else:
        print(f"警告: {method_name} 全MLB逐年數據量不足以訓練回歸模型，跳過。")

    # 運行分類模型
    if len(X_all_10y) > 1 and len(y_all_10y_binary) > 1 and len(y_all_10y_binary.unique()) > 1:
        res_clf_all_10y, _ = run_classification_model(X_all_10y, y_all_10y_binary)
        for model_type, metrics in res_clf_all_10y.items():
            all_classification_comparison_results.append({
                'Feature_Set': method_name,
                'Model': model_type,
                'Accuracy': metrics['Accuracy'],
                'F1-Score': metrics['F1-Score']
            })
    else:
        print(f"警告: {method_name} 全MLB逐年數據量或類別不足以訓練分類模型，跳過。")

df_all_10y_results_reg = pd.DataFrame(results_all_10y_regression)
df_all_10y_results_clf = pd.DataFrame(results_all_10y_classification)


# ======= Combine All Results (Regression Models) =======
# 現在，這些圖表將包含 'feature_sets_for_comparison' 中的所有特徵集
df_all_results_regression = pd.concat([df_single_regression, df_regression_compare, df_all_10y_results_reg], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', hue='Feature_Set', data=df_all_results_regression)
plt.title("Regression Model MSE Comparison (Different Feature Sets)") # Changed to English
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='R2', hue='Feature_Set', data=df_all_results_regression)
plt.title("Regression Model R2 Comparison (Different Feature Sets)") # Changed to English
plt.show()

# ======= Combine All Results (Classification Models) =======
df_all_results_classification = pd.concat([df_single_classification, df_classification_compare, df_all_10y_results_clf], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Accuracy', hue='Feature_Set', data=df_all_results_classification)
plt.title("Classification Model Accuracy Comparison (Different Feature Sets)") # Changed to English
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='F1-Score', hue='Feature_Set', data=df_all_results_classification)
plt.title("Classification Model F1-Score Comparison (Different Feature Sets)") # Changed to English
plt.ylim(0, 1)
plt.show()


# ======= Plot Feature Importance Heatmap (for general candidates, not just selected) =======
# 此部分仍然根據 'feature_candidates_for_general_selection' 計算重要性
# 可能不會直接反映 'Engineered Features'。保留用於一般特徵理解。
X_all_features_for_importance = merged[feature_candidates_for_general_selection]
y_target_continuous_for_importance = merged[target_col]

if not X_all_features_for_importance.empty and not y_target_continuous_for_importance.empty and len(X_all_features_for_importance) > 1:
    _, rf_model_for_importance_plot, _ = run_regression_models(X_all_features_for_importance, y_target_continuous_for_importance)
    if rf_model_for_importance_plot:
        feature_importances = pd.Series(rf_model_for_importance_plot.feature_importances_, index=feature_candidates_for_general_selection)
        plt.figure(figsize=(8, len(feature_importances) * 0.6))
        sns.heatmap(
            feature_importances.sort_values(ascending=False).to_frame(),
            annot=True, cmap='viridis', fmt=".4f",
            cbar=True, yticklabels=True, xticklabels=['Importance'],
            linewidths=0.5, linecolor='gray'
        )
        plt.title("Random Forest Feature Importance (All Initial Candidate Features)") # Changed to English
        plt.yticks(rotation=0)
        plt.tight_layout()
        importance_heatmap_path = os.path.join(base_path_report, "feature_importance_heatmap.png")
        plt.savefig(importance_heatmap_path, dpi=300)
        plt.show()
        print(f"特徵重要性熱力圖已儲存至 {importance_heatmap_path}")
    else:
        print("無法繪製特徵重要性熱力圖: 無法訓練所有候選特徵的隨機森林模型。")
else:
    print("無法繪製特徵重要性熱力圖: 數據不足以處理所有候選特徵。")


# ======= Heatmap: Most Correlated Variables with Win_Pct (Pearson, Spearman, Kendall) =======
# 此部分也保留用於所有候選特徵的一般相關性理解。
if feature_candidates_for_general_selection:
    corr_pearson_series = merged[feature_candidates_for_general_selection + [target_col]].corr(method='pearson')[target_col].drop(target_col)
    corr_spearman_series = merged[feature_candidates_for_general_selection + [target_col]].corr(method='spearman')[target_col].drop(target_col)
    kendall_corr_data = []
    for col in feature_candidates_for_general_selection:
        try:
            kendall_tau, _ = stats.kendalltau(merged[col], merged[target_col])
            kendall_corr_data.append(kendall_tau)
        except ValueError:
            kendall_corr_data.append(np.nan)
    kendall_corr_series = pd.Series(kendall_corr_data, index=feature_candidates_for_general_selection)

    all_correlated_features = set()
    for s in [corr_pearson_series, corr_spearman_series, kendall_corr_series]:
        valid_s = s.dropna()
        all_correlated_features.update(valid_s[valid_s.abs() >= correlation_threshold].index.tolist())

    if all_correlated_features:
        df_correlations = pd.DataFrame({
            'Pearson': corr_pearson_series.loc[list(all_correlated_features)],
            'Spearman': spearman_corr_series.loc[list(all_correlated_features)],
            'Kendall': kendall_corr_series.loc[list(all_correlated_features)]
        }).dropna(how='all')

        if not df_correlations.empty:
            plt.figure(figsize=(6, len(df_correlations) * 0.7))
            sns.heatmap(
                df_correlations.sort_values(by='Pearson', ascending=False),
                annot=True, cmap='coolwarm', center=0, cbar=True,
                yticklabels=True, xticklabels=True,
                linewidths=0.5, linecolor='gray', fmt=".2f"
            )
            plt.title(f"Most Correlated Features with {target_col} (Pearson, Spearman, Kendall) - All Initial Candidate Features") # Changed to English
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path_multi_corr = os.path.join(base_path_report, "heatmap_multi_correlation.png")
            plt.savefig(save_path_multi_corr, dpi=300)
            plt.show()
            print(f"多重相關性熱力圖已儲存至 {save_path_multi_corr}")
        else:
            print("沒有足夠的特徵達到相關性閾值以繪製多重相關性熱力圖。")
    else:
        print("沒有特徵達到任何相關性閾值以繪製多重相關性熱力圖。")
else:
    print("沒有初始候選特徵可繪製相關性熱力圖。")


# ======= Model Performance Comparison Chart (for Feature Selection Methods) =======
# 現在，這些圖表將包含 'feature_sets_for_comparison' 中的所有特徵集
if not df_regression_compare.empty:
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='R2', hue='Model', data=df_regression_compare)
    plt.title("Regression Model R² Comparison (Different Feature Sets)") # Changed to English
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='MSE', hue='Model', data=df_regression_compare)
    plt.title("Regression Model MSE Comparison (Different Feature Sets)") # Changed to English
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可比較的迴歸模型結果資料。")

if not df_classification_compare.empty:
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='Accuracy', hue='Model', data=df_classification_compare)
    plt.title("Classification Model Accuracy Comparison (Different Feature Sets)") # Changed to English
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='F1-Score', hue='Model', data=df_classification_compare)
    plt.title("Classification Model F1-Score Comparison (Different Feature Sets)") # Changed to English
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可比較的分類模型結果資料。")

# ======= Output Specific Accuracy Data =======
print("\n=== Logistic Regression Classifier Accuracy Comparison ===") # Changed to English
if 'Logistic Regression Classifier' in df_classification_compare['Model'].unique():
    lr_results = df_classification_compare[df_classification_compare['Model'] == 'Logistic Regression Classifier']
    if 'Engineered Features' in lr_results['Feature_Set'].values:
        eng_acc = lr_results[lr_results['Feature_Set'] == 'Engineered Features']['Accuracy'].mean()
        eng_f1 = lr_results[lr_results['Feature_Set'] == 'Engineered Features']['F1-Score'].mean()
        print(f"Logistic Regression Classifier using 'Engineered Features':") # Changed to English
        print(f"  Average Accuracy: {eng_acc:.4f}") # Changed to English
        print(f"  Average F1-Score: {eng_f1:.4f}") # Changed to English
    else:
        print("'Engineered Features' has no Logistic Regression Classifier results.") # Changed to English

    if 'OPS+ERA (Control)' in lr_results['Feature_Set'].values:
        ops_era_acc = lr_results[lr_results['Feature_Set'] == 'OPS+ERA (Control)']['Accuracy'].mean()
        ops_era_f1 = lr_results[lr_results['Feature_Set'] == 'OPS+ERA (Control)']['F1-Score'].mean()
        print(f"Logistic Regression Classifier using 'OPS+ERA (Control)':") # Changed to English
        print(f"  Average Accuracy: {ops_era_acc:.4f}") # Changed to English
        print(f"  Average F1-Score: {ops_era_f1:.4f}") # Changed to English
    else:
        print("'OPS+ERA (Control)' has no Logistic Regression Classifier results.") # Changed to English

    if 'RFECV' in lr_results['Feature_Set'].values:
        rfecv_acc = lr_results[lr_results['Feature_Set'] == 'RFECV']['Accuracy'].mean()
        rfecv_f1 = lr_results[lr_results['Feature_Set'] == 'RFECV']['F1-Score'].mean()
        print(f"Logistic Regression Classifier using features selected by 'RFECV':") # Changed to English
        print(f"  Average Accuracy: {rfecv_acc:.4f}") # Changed to English
        print(f"  Average F1-Score: {rfecv_f1:.4f}") # Changed to English
    else:
        print("'RFECV' has no Logistic Regression Classifier results.") # Changed to English
    
    if 'RFECV Combined Single Feature' in lr_results['Feature_Set'].values:
        rfecv_combined_acc = lr_results[lr_results['Feature_Set'] == 'RFECV Combined Single Feature']['Accuracy'].mean()
        rfecv_combined_f1 = lr_results[lr_results['Feature_Set'] == 'RFECV Combined Single Feature']['F1-Score'].mean()
        print(f"Logistic Regression Classifier using 'RFECV Combined Single Feature':") # Changed to English
        print(f"  Average Accuracy: {rfecv_combined_acc:.4f}") # Changed to English
        print(f"  Average F1-Score: {rfecv_combined_f1:.4f}") # Changed to English
    else:
        print("'RFECV Combined Single Feature' has no Logistic Regression Classifier results.") # Changed to English

else:
    print("No Logistic Regression Classifier results available for comparison.") # Changed to English


# ======= Output Regression Model Performance Summary =======
print("\n=== Regression Model Performance Summary (Engineered Features vs OPS+ERA vs RFECV vs RFECV Combined vs RFECV Weighted Composite) ===") # Changed to English and added new feature set
summary_reg_df = df_all_results_regression[df_all_results_regression['Feature_Set'].isin(['Engineered Features', 'OPS+ERA (Control)', 'RFECV', 'RFECV Combined Single Feature', 'RFECV Weighted Composite'])].copy() # Added new feature set

if not summary_reg_df.empty:
    grouped_summary = summary_reg_df.groupby(['Feature_Set', 'Model']).agg(
        Avg_MSE=('MSE', 'mean'),
        Avg_R2=('R2', 'mean')
    ).reset_index()

    for index, row in grouped_summary.iterrows():
        print(f"Feature Set: {row['Feature_Set']}, Model: {row['Model']}:") # Changed to English
        print(f"  Average MSE: {row['Avg_MSE']:.4f}") # Changed to English
        print(f"  Average R2: {row['Avg_R2']:.4f}") # Changed to English
else:
    print("No regression model performance data available for summary comparison.") # Changed to English


# ======= Visualize Decision Tree (Using Engineered Features if available, else OPS+ERA or RFECV) =======
final_features_for_tree = []
dt_model_for_vis = None
rf_model_for_vis = None

# Priority: RFECV Weighted Composite > Engineered Features > RFECV > RFECV Combined Single Feature > OPS+ERA
if 'RFECV_Weighted_Index' in merged.columns:
    final_features_for_tree = ['RFECV_Weighted_Index']
    print("\nDecision tree will be visualized using 'RFECV Weighted Composite'.") # Changed to English
    X_tree_data = merged[['RFECV_Weighted_Index']]
    y_tree_data = merged[target_col]
    _, rf_model_for_vis, dt_model_for_vis = run_regression_models(X_tree_data, y_tree_data)

elif engineered_features and all(f in merged.columns for f in engineered_features):
    final_features_for_tree = engineered_features
    print("\nDecision tree will be visualized using 'Engineered Features'.") # Changed to English
    X_tree_data = merged[engineered_features]
    y_tree_data = merged[target_col]
    _, rf_model_for_vis, dt_model_for_vis = run_regression_models(X_tree_data, y_tree_data)

elif selected_features_rfecv and all(f in merged.columns for f in selected_features_rfecv):
    final_features_for_tree = selected_features_rfecv
    print("\nDecision tree will be visualized using 'RFECV' selected original features.") # Changed to English
    X_tree_data = merged[selected_features_rfecv]
    y_tree_data = merged[target_col]
    _, rf_model_for_vis, dt_model_for_vis = run_regression_models(X_tree_data, y_tree_data)

elif rfecv_combined_feature_name in merged.columns:
    final_features_for_tree = [rfecv_combined_feature_name]
    print(f"\nDecision tree will be visualized using '{rfecv_combined_feature_name}'.") # Changed to English
    X_tree_data = merged[[rfecv_combined_feature_name]]
    y_tree_data = merged[target_col]
    _, rf_model_for_vis, dt_model_for_vis = run_regression_models(X_tree_data, y_tree_data)

elif ops_era_features and all(f in merged.columns for f in ops_era_features):
    final_features_for_tree = ops_era_features
    print("\nDecision tree will be visualized using 'OPS+ERA (Control)'.") # Changed to English
    X_tree_data = merged[ops_era_features]
    y_tree_data = merged[target_col]
    _, rf_model_for_vis, dt_model_for_vis = run_regression_models(X_tree_data, y_tree_data)

else:
    print("\nNo sufficient features available to plot decision tree.") # Changed to English

if dt_model_for_vis and rf_model_for_vis and final_features_for_tree:
    df_10y_for_tree = merged.groupby('Team')[final_features_for_tree + [target_col]].mean(numeric_only=True).reset_index()
    X_10y_for_tree = df_10y_for_tree[[f for f in final_features_for_tree if f in df_10y_for_tree.columns]]

    if not X_10y_for_tree.empty:
        # 1. Decision Tree (Single)
        plt.figure(figsize=(20, 10))
        plot_tree(dt_model_for_vis, feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title(f"Decision Tree (max_depth=3) - Features: {', '.join(final_features_for_tree)}") # Changed to English
        plt.tight_layout()
        plt.savefig(os.path.join(base_path_report, "decision_tree_10y.png"), dpi=300)
        plt.show() # <<< 修改處
        plt.close()

        # 2. Random Forest (Single Tree)
        plt.figure(figsize=(20, 10))
        # Plot the first tree in Random Forest
        plot_tree(rf_model_for_vis.estimators_[0], feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title(f"Random Forest - Single Tree (max_depth=3) - Features: {', '.join(final_features_for_tree)}") # Changed to English
        plt.tight_layout()
        plt.savefig(os.path.join(base_path_report, "random_forest_tree_10y.png"), dpi=300)
        plt.show() # <<< 修改處
        plt.close()
    else:
        print("Cannot plot decision tree: Feature data for plotting is empty.") # Changed to English
else:
    print("Cannot plot decision tree: Model not trained or final selected features are empty.") # Changed to English


# ======= Save Results to CSV File (Multiple Sheets) =======
output_path_csv = os.path.join(base_path_report, "mlb_model_results.csv")

# 格式化數值結果
df_all_results_regression_formatted = df_all_results_regression.copy()
df_all_results_regression_formatted['MSE'] = df_all_results_regression_formatted['MSE'].round(9)
df_all_results_regression_formatted['R2'] = df_all_results_regression_formatted['R2'].round(9)

df_all_results_classification_formatted = df_all_results_classification.copy()
df_all_results_classification_formatted['Accuracy'] = df_all_results_classification_formatted['Accuracy'].round(9)
df_all_results_classification_formatted['F1-Score'] = df_all_results_classification_formatted['F1-Score'].round(9)

df_regression_compare_formatted = df_regression_compare.copy()
df_regression_compare_formatted['MSE'] = df_regression_compare_formatted['MSE'].round(9)
df_regression_compare_formatted['R2'] = df_regression_compare_formatted['R2'].round(9)

df_classification_compare_formatted = df_classification_compare.copy()
df_classification_compare_formatted['Accuracy'] = df_classification_compare_formatted['Accuracy'].round(9)
df_classification_compare_formatted['F1-Score'] = df_classification_compare_formatted['F1-Score'].round(9)

'''
# 將結果保存到 Excel
with pd.ExcelWriter(output_path_csv.replace('.csv', '.xlsx'), engine='xlsxwriter') as writer:
    df_all_results_regression_formatted.to_excel(writer, sheet_name='All_Regression_Results', index=False)
    df_all_results_classification_formatted.to_excel(writer, sheet_name='All_Classification_Results', index=False)
    df_regression_compare_formatted.to_excel(writer, sheet_name='Regression_Compare', index=False)
    df_classification_compare_formatted.to_excel(writer, sheet_name='Classification_Compare', index=False)
print(f"\n 結果已成功儲存為 Excel 檔：{output_path_csv.replace('.csv', '.xlsx')}")
'''

# --- 訓練用於互動式預測的特定模型 ---
# 將互動式模型訓練放在這裡，確保在所有分析完成後再進行
interactive_predictor_model_eng = None
interactive_predictor_model_ops_era = None
interactive_predictor_model_rfecv = None # RFECV 選出的原始特徵模型
interactive_predictor_model_rfecv_combined = None # RFECV 選出並組合成單一特徵的模型
interactive_predictor_model_rfecv_weighted_composite = None # 新增 RFECV Weighted Composite 模型

# 1. 使用 Engineered Features 訓練模型
if engineered_features and all(f in merged.columns for f in engineered_features):
    X_for_interactive_model_eng = merged[engineered_features]
    y_for_interactive_model_eng = merged[target_col]

    if not X_for_interactive_model_eng.empty and not y_for_interactive_model_eng.empty and len(X_for_interactive_model_eng) > 1:
        # 處理 NaN 值，以防萬一
        if X_for_interactive_model_eng.isnull().any().any():
            X_for_interactive_model_eng = X_for_interactive_model_eng.fillna(X_for_interactive_model_eng.mean(numeric_only=True))
        if y_for_interactive_model_eng.isnull().any():
            y_for_interactive_model_eng = y_for_interactive_model_eng.fillna(y_for_interactive_model_eng.mean())

        interactive_predictor_model_eng = RandomForestRegressor(random_state=42)
        interactive_predictor_model_eng.fit(X_for_interactive_model_eng, y_for_interactive_model_eng)
        print("\n用於互動式預測的隨機森林模型 (使用 Engineered Features) 已訓練完成。")
    else:
        print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 Engineered Features)。")
else:
    print("\n警告: 沒有組合指標 (Engineered Features) 可用來訓練互動式預測模型。")


# 2. 使用 OPS+ERA 訓練模型
if ops_era_features and all(f in merged.columns for f in ops_era_features):
    X_for_interactive_model_ops_era = merged[ops_era_features]
    y_for_interactive_model_ops_era = merged[target_col]

    if not X_for_interactive_model_ops_era.empty and not y_for_interactive_model_ops_era.empty and len(X_for_interactive_model_ops_era) > 1:
        # 處理 NaN 值，以防萬一
        if X_for_interactive_model_ops_era.isnull().any().any():
            X_for_interactive_model_ops_era = X_for_interactive_model_ops_era.fillna(X_for_interactive_model_ops_era.mean(numeric_only=True))
        if y_for_interactive_model_ops_era.isnull().any():
            y_for_interactive_model_ops_era = y_for_interactive_model_ops_era.fillna(y_for_interactive_model_ops_era.mean())

        interactive_predictor_model_ops_era = RandomForestRegressor(random_state=42)
        interactive_predictor_model_ops_era.fit(X_for_interactive_model_ops_era, y_for_interactive_model_ops_era)
        print("\n用於互動式預測的隨機森林模型 (使用 OPS+ERA) 已訓練完成。")
    else:
        print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 OPS+ERA)。")
else:
    print("\n警告: OPS 或 ERA 欄位不存在，無法訓練互動式預測模型 (使用 OPS+ERA)。")

# 3. 使用 RFECV 選出的原始特徵訓練模型
if selected_features_rfecv and all(f in merged.columns for f in selected_features_rfecv):
    X_for_interactive_model_rfecv = merged[selected_features_rfecv]
    y_for_interactive_model_rfecv = merged[target_col]

    if not X_for_interactive_model_rfecv.empty and not y_for_interactive_model_rfecv.empty and len(X_for_interactive_model_rfecv) > 1:
        # 處理 NaN 值，以防萬一
        if X_for_interactive_model_rfecv.isnull().any().any():
            X_for_interactive_model_rfecv = X_for_interactive_model_rfecv.fillna(X_for_interactive_model_rfecv.mean(numeric_only=True))
        if y_for_interactive_model_rfecv.isnull().any():
            y_for_interactive_model_rfecv = y_for_interactive_model_rfecv.fillna(y_for_interactive_model_rfecv.mean())

        interactive_predictor_model_rfecv = RandomForestRegressor(random_state=42)
        interactive_predictor_model_rfecv.fit(X_for_interactive_model_rfecv, y_for_interactive_model_rfecv)
        print("\n用於互動式預測的隨機森林模型 (使用 RFECV 選出的原始特徵) 已訓練完成。")
    else:
        print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 RFECV 選出的原始特徵)。")
else:
    print("\n警告: RFECV 未選出原始特徵或特徵不存在，無法訓練互動式預測模型。")

# 4. 使用 RFECV 選出並組合成單一特徵的訓練模型
if rfecv_combined_feature_name in merged.columns:
    X_for_interactive_model_rfecv_combined = merged[[rfecv_combined_feature_name]]
    y_for_interactive_model_rfecv_combined = merged[target_col]

    if not X_for_interactive_model_rfecv_combined.empty and not y_for_interactive_model_rfecv_combined.empty and len(X_for_interactive_model_rfecv_combined) > 1:
        X_for_interactive_model_rfecv_combined = X_for_interactive_model_rfecv_combined.fillna(X_for_interactive_model_rfecv_combined.mean(numeric_only=True))
        y_for_interactive_model_rfecv_combined = y_for_interactive_model_rfecv_combined.fillna(y_for_interactive_model_rfecv_combined.mean())
        
        interactive_predictor_model_rfecv_combined = RandomForestRegressor(random_state=42)
        interactive_predictor_model_rfecv_combined.fit(X_for_interactive_model_rfecv_combined, y_for_interactive_model_rfecv_combined)
        print(f"\n用於互動式預測的隨機森林模型 (使用 '{rfecv_combined_feature_name}') 已訓練完成。")
    else:
        print(f"\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 '{rfecv_combined_feature_name}')。")
else:
    print(f"\n警告: '{rfecv_combined_feature_name}' 欄位不存在，無法訓練互動式預測模型。")

# 5. 使用 RFECV Weighted Composite 訓練模型
rfecv_weighted_composite_name = 'RFECV_Weighted_Index'
if rfecv_weighted_composite_name in merged.columns:
    X_for_interactive_model_rfecv_weighted_composite = merged[[rfecv_weighted_composite_name]]
    y_for_interactive_model_rfecv_weighted_composite = merged[target_col]

    if not X_for_interactive_model_rfecv_weighted_composite.empty and not y_for_interactive_model_rfecv_weighted_composite.empty and len(X_for_interactive_model_rfecv_weighted_composite) > 1:
        X_for_interactive_model_rfecv_weighted_composite = X_for_interactive_model_rfecv_weighted_composite.fillna(X_for_interactive_model_rfecv_weighted_composite.mean(numeric_only=True))
        y_for_interactive_model_rfecv_weighted_composite = y_for_interactive_model_rfecv_weighted_composite.fillna(y_for_interactive_model_rfecv_weighted_composite.mean())
        
        interactive_predictor_model_rfecv_weighted_composite = RandomForestRegressor(random_state=42)
        interactive_predictor_model_rfecv_weighted_composite.fit(X_for_interactive_model_rfecv_weighted_composite, y_for_interactive_model_rfecv_weighted_composite)
        print(f"\n用於互動式預測的隨機森林模型 (使用 '{rfecv_weighted_composite_name}') 已訓練完成。")
    else:
        print(f"\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 '{rfecv_weighted_composite_name}')。")
else:
    print(f"\n警告: '{rfecv_weighted_composite_name}' 欄位不存在，無法訓練互動式預測模型。")


# --- 新增互動式預測函數 ---
def run_interactive_prediction(eng_model, ops_era_model, rfecv_model, rfecv_combined_model, rfecv_weighted_composite_model, engineered_features_list, ops_era_features_list, rfecv_features_list, rfecv_combined_feature_name_func, rfecv_weighted_composite_feature_name_func, merged_data, target_col_name):
    """
    提供互動式介面，讓使用者輸入所有 RFECV 選中的特徵的值，
    並使用訓練好的模型預測勝率。
    Args:
        eng_model (sklearn.ensemble.RandomForestRegressor): 使用 Engineered Features 訓練好的模型。
        ops_era_model (sklearn.ensemble.RandomForestRegressor): 使用 OPS+ERA 特徵訓練好的模型。
        rfecv_model (sklearn.ensemble.RandomForestRegressor): 使用 RFECV 選出的原始特徵訓練好的模型。
        rfecv_combined_model (sklearn.ensemble.RandomForestRegressor): 使用 RFECV 選出並組合成單一特徵訓練好的模型。
        rfecv_weighted_composite_model (sklearn.ensemble.RandomForestRegressor): 使用 RFECV Weighted Composite 特徵訓練好的模型。
        engineered_features_list (list): 組合特徵的列表 (例如 ['Offensive_Productivity_Index', 'Defensive_Efficiency_Index'])。
        ops_era_features_list (list): OPS 和 ERA 特徵的列表 (例如 ['OPS', 'ERA'])。
        rfecv_features_list (list): RFECV 選出的原始特徵列表。
        rfecv_combined_feature_name_func (str): RFECV 組合特徵的名稱。
        rfecv_weighted_composite_feature_name_func (str): RFECV 加權組合特徵的名稱。
        merged_data (pd.DataFrame): 完整的資料集。
        target_col_name (str): 目標變數的名稱。
    """
    print("\n" * 2) # 加入空行增加可讀性
    print("=" * 40)
    print("      ⚾ MLB Win Probability Interactive Prediction ⚾") # Changed to English
    print("=" * 40)
    print("\nPlease enter the values for the following raw variables to predict win percentage.") # Changed to English
    print("Enter 'exit' to quit the program.") # Changed to English
    print("\n--- Note: Model accuracy is presented in the charts above and in the Excel report; single predictions cannot determine absolute accuracy. ---") # Changed to English

    while True:
        try:
            print("\n" + "-" * 30)
            user_raw_inputs = {}
            prompted_features_set = set() # 用於追蹤哪些特徵已經被提示輸入過

            # 預設的核心棒球數據及其在 merged_data 中的平均值（用於建議值）
            # 注意：這裡只定義了哪些特徵是核心，實際提示會在後面的 RFECV 特徵迭代中處理
            core_feature_defaults = {
                'OBP': (0.330, 'On-Base Percentage'), # Changed to English
                'SLG': (0.420, 'Slugging Percentage'), # Changed to English
                'OPS': (0.750, 'On-Base Plus Slugging'), # Changed to English
                'WHIP': (1.25, 'Walks Plus Hits Per Inning Pitched'), # Changed to English
                'ER': (0.5, 'Earned Runs'), # Changed to English
                'ERA': (3.80, 'Earned Run Average'), # Changed to English
                'FIP': (3.80, 'Fielding Independent Pitching'), # Changed to English
                'BA': (0.250, 'Batting Average'), # Changed to English
                'HR': (20, 'Home Runs'), # Changed to English
                'SO': (100, 'Strikeouts'), # Changed to English
                'BB': (50, 'Walks'), # Changed to English
                'RBI': (70, 'Runs Batted In'), # Changed to English
                'SB': (10, 'Stolen Bases'), # Changed to English
                'IP': (150, 'Innings Pitched'), # Changed to English
                'SV': (20, 'Saves'), # Added for RFECV Weighted Composite
                # 您可以根據數據中可能出現的其他關鍵數值特徵繼續添加
            }

            # 確保要提示的特徵列表包含 RFECV 模型所需的所有原始輸入特徵
            # 這包括 Engineered Features 的組成部分以及 RFECV 直接選出的其他原始特徵
            features_to_prompt_for = set()
            
            # 從 engineered_features_list 獲取需要提示的原始特徵
            for feat in engineered_features_list:
                if feat == 'Offensive_Productivity_Index':
                    features_to_prompt_for.add('OBP')
                    features_to_prompt_for.add('SLG')
                elif feat == 'Defensive_Efficiency_Index':
                    if 'FIP' in merged_data.columns:
                        features_to_prompt_for.add('FIP')
                    features_to_prompt_for.add('WHIP')
                    features_to_prompt_for.add('ER')
                elif feat == rfecv_weighted_composite_feature_name_func: # New: Add components for RFECV Weighted Composite
                    features_to_prompt_for.update(rfecv_base_features) # BB, ER, SV
                else:
                    if feat in merged_data.columns and np.issubdtype(merged_data[feat].dtype, np.number):
                        features_to_prompt_for.add(feat)

            # 從 ops_era_features_list 獲取需要提示的原始特徵
            for feat in ops_era_features_list:
                if feat in merged_data.columns and np.issubdtype(merged_data[feat].dtype, np.number):
                    features_to_prompt_for.add(feat)

            # 從 rfecv_features_list 獲取需要提示的原始特徵
            for feat in rfecv_features_list:
                if feat in merged_data.columns and np.issubdtype(merged_data[feat].dtype, np.number):
                    features_to_prompt_for.add(feat)


            # 實際提示用戶輸入各個特徵值
            for feature in sorted(list(features_to_prompt_for)): # 排序以便輸出順序一致
                if feature in core_feature_defaults:
                    avg_val, desc = core_feature_defaults[feature]
                    # 從實際數據中獲取平均值，如果存在
                    if feature in merged_data.columns and not merged_data[feature].isnull().all():
                        avg_val = merged_data[feature].mean()
                    
                    input_str = input(f"  Enter {desc} ({feature}, suggested value {avg_val:.3f}): ") # Changed to English
                    if input_str.lower() == 'exit': return
                    user_raw_inputs[feature] = float(input_str)
                    prompted_features_set.add(feature)
                elif feature in merged_data.columns and np.issubdtype(merged_data[feature].dtype, np.number):
                    # 對於 RFECV 選出但不在核心預設列表中的數值特徵
                    avg_val = merged_data[feature].mean() if not merged_data[feature].isnull().all() else 0.0
                    input_str = input(f"  Enter {feature} (suggested value {avg_val:.3f}): ") # Changed to English
                    if input_str.lower() == 'exit': return
                    user_raw_inputs[feature] = float(input_str)
                    prompted_features_set.add(feature)
                else:
                    print(f"  Warning: Feature '{feature}' is undefined or non-numeric, unable to get input. Using default value 0.") # Changed to English
                    user_raw_inputs[feature] = 0.0
                    prompted_features_set.add(feature)

            print(f"\n--- Prediction Results ---") # Changed to English
            # 打印所有已接收或處理的原始輸入
            print("Raw input values:", {k: f"{v:.3f}" if isinstance(v, (int, float)) else v for k, v in user_raw_inputs.items()}) # Changed to English


            # --- 使用 Engineered Features 模型預測 ---
            predicted_win_pct_eng = np.nan
            if eng_model and engineered_features_list:
                input_values_eng = {}
                # Only include features that the model was actually trained on
                for feature_name_in_model in eng_model.feature_names_in_:
                    if feature_name_in_model == 'Offensive_Productivity_Index':
                        input_values_eng['Offensive_Productivity_Index'] = user_raw_inputs.get('OBP', 0) + user_raw_inputs.get('SLG', 0)
                    elif feature_name_in_model == 'Defensive_Efficiency_Index':
                        if 'FIP' in user_raw_inputs and 'FIP' in merged_data.columns:
                            input_values_eng['Defensive_Efficiency_Index'] = user_raw_inputs.get('FIP', 0) + user_raw_inputs.get('WHIP', 0)
                        else:
                            input_values_eng['Defensive_Efficiency_Index'] = user_raw_inputs.get('WHIP', 0) + user_raw_inputs.get('ER', 0)
                    elif feature_name_in_model == rfecv_weighted_composite_feature_name_func: # Added new composite feature
                        composite_val = 0.0
                        for base_feat in rfecv_base_features: # Use the defined base features for the weighted index
                            if base_feat in user_raw_inputs and base_feat in merged_data.columns:
                                # Recalculate correlation for this specific prediction instance if desired,
                                # or use the pre-calculated one. For simplicity, we'll re-calculate.
                                # In a real app, you'd save correlations from training.
                                # The 'correlations' variable already contains the non-absolute correlations now.
                                current_corr = correlations.get(base_feat, 0)
                                composite_val += current_corr * user_raw_inputs.get(base_feat, 0)
                            else:
                                print(f"  Warning: Component '{base_feat}' for RFECV Weighted Composite not provided or invalid, using 0.")
                        input_values_eng[feature_name_in_model] = composite_val
                    elif feature_name_in_model in user_raw_inputs:
                        input_values_eng[feature_name_in_model] = user_raw_inputs[feature_name_in_model]
                    else: # Fallback for other engineered features if they are in eng_model.feature_names_in_ but not handled above
                        print(f"  Warning: Feature '{feature_name_in_model}' for Engineered Features model missing, using 0.")
                        input_values_eng[feature_name_in_model] = 0.0

                input_df_eng = pd.DataFrame([input_values_eng])
                try:
                    if set(input_df_eng.columns) == set(eng_model.feature_names_in_):
                        input_df_eng = input_df_eng[eng_model.feature_names_in_]
                        predicted_win_pct_eng = eng_model.predict(input_df_eng)[0]
                        print(f"Predicted Win Percentage using 'Engineered Features': {predicted_win_pct_eng:.4f}") # Changed to English
                    else:
                        print(f"Warning: 'Engineered Features' model input features mismatch. Cannot predict. Expected: {eng_model.feature_names_in_.tolist()}. Actual: {input_df_eng.columns.tolist()}.") # Changed to English
                except Exception as e:
                    print(f"Prediction failed using 'Engineered Features': {e}") # Changed to English
            else:
                print("Cannot predict using 'Engineered Features' (model not trained or insufficient features).") # Changed to English

            # --- 使用 OPS+ERA (Control) 模型預測 ---
            predicted_win_pct_ops_era = np.nan
            if ops_era_model and ops_era_features_list:
                input_values_ops_era = {}
                for feature_name_in_model in ops_era_model.feature_names_in_:
                    input_values_ops_era[feature_name_in_model] = user_raw_inputs.get(feature_name_in_model, 0) # Directly use raw input
                
                input_df_ops_era = pd.DataFrame([input_values_ops_era])
                try:
                    if set(input_df_ops_era.columns) == set(ops_era_model.feature_names_in_):
                        input_df_ops_era = input_df_ops_era[ops_era_model.feature_names_in_]
                        predicted_win_pct_ops_era = ops_era_model.predict(input_df_ops_era)[0]
                        print(f"Predicted Win Percentage using 'OPS+ERA (Control)': {predicted_win_pct_ops_era:.4f}") # Changed to English
                    else:
                        print(f"Warning: 'OPS+ERA (Control)' model input features mismatch. Cannot predict. Expected: {ops_era_model.feature_names_in_.tolist()}. Actual: {input_df_ops_era.columns.tolist()}.") # Changed to English
                except Exception as e:
                    print(f"Prediction failed using 'OPS+ERA (Control)': {e}") # Changed to English
            else:
                print("Cannot predict using 'OPS+ERA (Control)' (model not trained or insufficient features).") # Changed to English

            # --- 使用 RFECV 選出的原始特徵模型預測 ---
            predicted_win_pct_rfecv = np.nan
            if rfecv_model and rfecv_features_list:
                input_values_rfecv = {}
                for feature in rfecv_model.feature_names_in_: # 遍歷模型訓練時的特徵順序
                    if feature in user_raw_inputs:
                        input_values_rfecv[feature] = user_raw_inputs[feature]
                    elif feature == 'Offensive_Productivity_Index':
                        input_values_rfecv[feature] = user_raw_inputs.get('OBP', 0) + user_raw_inputs.get('SLG', 0)
                    elif feature == 'Defensive_Efficiency_Index':
                        if 'FIP' in user_raw_inputs and 'FIP' in merged_data.columns:
                            input_values_rfecv[feature] = user_raw_inputs.get('FIP', 0) + user_raw_inputs.get('WHIP', 0)
                        else:
                            input_values_rfecv[feature] = user_raw_inputs.get('WHIP', 0) + user_raw_inputs.get('ER', 0)
                    elif feature == rfecv_weighted_composite_feature_name_func: # Added new composite feature
                        composite_val = 0.0
                        for base_feat in rfecv_base_features:
                            if base_feat in user_raw_inputs and base_feat in merged_data.columns:
                                # The 'correlations' variable already contains the non-absolute correlations now.
                                current_corr = correlations.get(base_feat, 0)
                                composite_val += current_corr * user_raw_inputs.get(base_feat, 0)
                        input_values_rfecv[feature] = composite_val
                    else:
                        print(f"  Warning: RFECV raw feature '{feature}' missing input value, using default 0.0.") # Changed to English
                        input_values_rfecv[feature] = 0.0

                input_df_rfecv = pd.DataFrame([input_values_rfecv])
                
                # 確保 input_df_rfecv 的欄位順序與訓練模型時的順序一致
                if set(input_df_rfecv.columns) == set(rfecv_model.feature_names_in_):
                    input_df_rfecv = input_df_rfecv[rfecv_model.feature_names_in_]
                    if not input_df_rfecv.empty and not input_df_rfecv.isnull().any().any():
                        try:
                            predicted_win_pct_rfecv = rfecv_model.predict(input_df_rfecv)[0]
                            print(f"Predicted Win Percentage using 'RFECV' raw features: {predicted_win_pct_rfecv:.4f}") # Changed to English
                        except Exception as e:
                            print(f"Prediction failed using 'RFECV' raw features: {e}") # Changed to English
                    else:
                        print("Cannot predict using 'RFECV' raw features (input data is empty or contains NaN).") # Changed to English
                else:
                    print(f"Error: RFECV raw feature model input features mismatch. Expected: {rfecv_model.feature_names_in_.tolist()}. Actual: {input_df_rfecv.columns.tolist()}. Cannot predict.") # Changed to English
                    predicted_win_pct_rfecv = np.nan
            else:
                print("Cannot predict using 'RFECV' raw features (model not trained or insufficient features).") # Changed to English

            # --- 使用 RFECV Combined Single Feature 模型預測 ---
            predicted_win_pct_rfecv_combined = np.nan
            if rfecv_combined_model and rfecv_features_list: # rfecv_features_list 用於計算組合特徵的成分
                combined_feature_value = 0.0
                # rfecv_base_features used here instead of numeric_rfecv_selected_features_for_sum as this is a specific weighted composite.
                # The existing rfecv_features_list is the source of the RFECV_Combined_Feature components.
                numeric_rfecv_selected_features_for_sum_local = [f for f in rfecv_features_list if f in merged_data.select_dtypes(include=np.number).columns]

                if not numeric_rfecv_selected_features_for_sum_local:
                    print(f"Warning: RFECV did not select any numeric features to create '{rfecv_combined_feature_name_func}'.") # Changed to English
                else:
                    for feature in numeric_rfecv_selected_features_for_sum_local:
                        if feature in user_raw_inputs:
                            combined_feature_value += user_raw_inputs[feature]
                        elif feature == 'Offensive_Productivity_Index':
                            combined_feature_value += (user_raw_inputs.get('OBP', 0) + user_raw_inputs.get('SLG', 0))
                        elif feature == 'Defensive_Efficiency_Index':
                            if 'FIP' in user_raw_inputs and 'FIP' in merged_data.columns:
                                combined_feature_value += (user_raw_inputs.get('FIP', 0) + user_raw_inputs.get('WHIP', 0))
                            else:
                                combined_feature_value += (user_raw_inputs.get('WHIP', 0) + user_raw_inputs.get('ER', 0))
                        elif feature == rfecv_weighted_composite_feature_name_func: # Added new composite feature
                             # If the single combined feature is also a component of the combined feature (meta!), handle it
                             # For simplicity, if it's already generated, we'll try to use it.
                            if rfecv_weighted_composite_feature_name_func in user_raw_inputs:
                                combined_feature_value += user_raw_inputs.get(rfecv_weighted_composite_feature_name_func, 0)
                            else:
                                # This case means RFECV_Weighted_Index was a component of RFECV_Combined_Feature,
                                # but RFECV_Weighted_Index was not directly prompted.
                                print(f"  Warning: RFECV Combined Single Feature component '{feature}' (which is RFECV Weighted Composite) not directly provided, using 0.")
                                combined_feature_value += 0.0
                        else:
                            if feature in merged_data.columns and not merged_data[feature].isnull().all():
                                val = merged_data[feature].mean()
                                combined_feature_value += val
                                print(f"  Component '{feature}' for RFECV combined feature not directly provided, using its mean {val:.3f}.") # Changed to English
                            else:
                                combined_feature_value += 0.0
                                print(f"  Component '{feature}' for RFECV combined feature not directly provided and data is empty, using default 0.") # Changed to English

                input_df_rfecv_combined = pd.DataFrame({rfecv_combined_feature_name_func: [combined_feature_value]})
                
                try:
                    if set(input_df_rfecv_combined.columns) == set(rfecv_combined_model.feature_names_in_):
                        input_df_rfecv_combined = input_df_rfecv_combined[rfecv_combined_model.feature_names_in_]
                        if not input_df_rfecv_combined.empty and not input_df_rfecv_combined.isnull().any().any():
                            predicted_win_pct_rfecv_combined = rfecv_combined_model.predict(input_df_rfecv_combined)[0]
                            print(f"Predicted Win Percentage using '{rfecv_combined_feature_name_func}': {predicted_win_pct_rfecv_combined:.4f}") # Changed to English
                        else:
                            print(f"Cannot predict using '{rfecv_combined_feature_name_func}' (input data is empty or contains NaN).") # Changed to English
                    else:
                        print(f"Warning: '{rfecv_combined_feature_name_func}' model input features mismatch. Cannot predict.") # Changed to English
                except Exception as e:
                    print(f"Prediction failed using '{rfecv_combined_feature_name_func}': {e}") # Changed to English
            else:
                print(f"Cannot predict using '{rfecv_combined_feature_name_func}' (model not trained or insufficient features).") # Changed to English

            # --- 使用 RFECV Weighted Composite 模型預測 (已更新為使用帶正負號的 correlations) ---
            predicted_win_pct_rfecv_weighted_composite = np.nan
            if rfecv_weighted_composite_model: # Check if the specific weighted composite model exists
                try:
                    # 使用全域的 'correlations' 變數，它現在已經包含正負號
                    if 'BB' in correlations and 'ER' in correlations and 'SV' in correlations:
                        val_weighted_composite = (
                            correlations['BB'] * float(user_raw_inputs.get('BB', 0)) +
                            correlations['ER'] * float(user_raw_inputs.get('ER', 0)) +
                            correlations['SV'] * float(user_raw_inputs.get('SV', 0))
                        )
                        input_df_rfecv_weighted_composite_pred = pd.DataFrame([[val_weighted_composite]], columns=[rfecv_weighted_composite_feature_name_func])
                        
                        if set(input_df_rfecv_weighted_composite_pred.columns) == set(rfecv_weighted_composite_model.feature_names_in_):
                            input_df_rfecv_weighted_composite_pred = input_df_rfecv_weighted_composite_pred[rfecv_weighted_composite_model.feature_names_in_]
                            if not input_df_rfecv_weighted_composite_pred.empty and not input_df_rfecv_weighted_composite_pred.isnull().any().any():
                                predicted_win_pct_rfecv_weighted_composite = rfecv_weighted_composite_model.predict(input_df_rfecv_weighted_composite_pred)[0]
                                print(f"Predicted Win Percentage using 'RFECV Weighted Composite': {predicted_win_pct_rfecv_weighted_composite:.4f}") # Changed to English
                            else:
                                print(f"Cannot predict using 'RFECV Weighted Composite' (input data is empty or contains NaN).") # Changed to English
                        else:
                            print(f"Warning: 'RFECV Weighted Composite' model input features mismatch. Cannot predict. Expected: {rfecv_weighted_composite_model.feature_names_in_.tolist()}. Actual: {input_df_rfecv_weighted_composite_pred.columns.tolist()}.") # Changed to English
                    else:
                        print("Warning: Correlations for BB, ER, SV are not available. Cannot predict using 'RFECV Weighted Composite'.") # Changed to English
                except Exception as e:
                    print(f"Prediction failed using 'RFECV Weighted Composite': {e}") # Changed to English
            else:
                print(f"Cannot predict using 'RFECV Weighted Composite' (model not trained or insufficient features).") # Changed to English

            print("-" * 30)

        except ValueError:
            print("Invalid input. Please enter a number.") # Changed to English
        except Exception as e:
            print(f"An error occurred during prediction: {e}") # Changed to English

    print("\nPrediction program ended.") # Changed to English

# --- 在所有分析和圖表生成後，運行互動式預測 ---
# 只有當至少一個模型訓練成功時才運行互動式預測
if interactive_predictor_model_eng or interactive_predictor_model_ops_era or interactive_predictor_model_rfecv or interactive_predictor_model_rfecv_combined or interactive_predictor_model_rfecv_weighted_composite:
    run_interactive_prediction(
        interactive_predictor_model_eng, 
        interactive_predictor_model_ops_era, 
        interactive_predictor_model_rfecv, 
        interactive_predictor_model_rfecv_combined,
        interactive_predictor_model_rfecv_weighted_composite, # Pass the new model
        engineered_features, 
        ops_era_features, 
        selected_features_rfecv, 
        rfecv_combined_feature_name, 
        'RFECV_Weighted_Index', # Pass the new feature name
        merged, 
        target_col
    )
else:
    print("\nInteractive prediction not started as no models are available for prediction.") # Changed to English