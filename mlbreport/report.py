import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score # Import classification metrics
import scipy.stats as stats # Import scipy.stats for Spearman and Kendall correlations
from sklearn.feature_selection import RFECV # Import RFECV
from sklearn.linear_model import LogisticRegression # Import Logistic Regression

# ======= Data Loading and Preprocessing =======
merged = pd.read_csv("C:/Users/richc/OneDrive/桌面/專題/mlbdata/merged_output_team_year_avg_standardized.csv")
target_col = 'Win_Pct' # Original continuous target variable

# Create binary classification target variable: Win_Pct > 0.5 is 1 (win), else 0 (loss)
merged['Win_Pct_Binary'] = (merged[target_col] > 0.5).astype(int)
binary_target_col = 'Win_Pct_Binary'

# Exclude target columns, Wins, Losses, W, L, and other related variables that imply data leakage
exclude_cols_initial = [
    'Wins', 'Losses', 'W', 'L', target_col, binary_target_col,
    'Total_WinDiff', 'Home_Losses', 'Home_Wins', 'Away_Losses', 'Home_WinDiff', 'Away_Wins', 'Away_WinDiff'
]

# Filter out excluded columns that actually exist in merged.columns
exclude_cols_for_candidates = [col for col in exclude_cols_initial if col in merged.columns]

print(f"實際從數據中排除的初始欄位: {exclude_cols_for_candidates}")

num_cols = merged.select_dtypes(include='number').columns.tolist()
# Filter out excluded columns from numerical columns to get the final feature candidates for other selection methods
feature_candidates_for_general_selection = [c for c in num_cols if c not in exclude_cols_for_candidates]

# Ensure all candidate features exist in the merged DataFrame (redundant but good for double-checking)
feature_candidates_for_general_selection = [f for f in feature_candidates_for_general_selection if f in merged.columns]
print(f"特徵選擇前的初始候選特徵集 (已排除直接結果變數): {feature_candidates_for_general_selection}")

# ======= Feature Engineering =======
engineered_features = []

# Create Offensive_Productivity_Index if OBP and SLG exist
if 'OBP' in merged.columns and 'SLG' in merged.columns:
    merged['Offensive_Productivity_Index'] = merged['OBP'] + merged['SLG']
    engineered_features.append('Offensive_Productivity_Index')
    print("已創建組合指標: Offensive_Productivity_Index (OBP + SLG)")
else:
    print("無法創建 Offensive_Productivity_Index，因為 OBP 或 SLG 不存在。")

# Create Defensive_Efficiency_Index if FIP and WHIP exist, or fall back to WHIP and ER
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


# ======= Modeling Function - Regression Models =======
def run_regression_models(X, y_reg):
    """
    Runs Decision Tree Regressor and Random Forest Regressor models.
    Args:
        X (pd.DataFrame): Feature data.
        y_reg (pd.Series): Continuous target variable (Win_Pct).
    Returns:
        tuple: Dictionary of model results, trained Random Forest model, trained Decision Tree model.
    """
    if len(X) < 2 or len(y_reg) < 2:
        return {'Random Forest Regressor': {'MSE': np.nan, 'R2': np.nan},
                'Decision Tree Regressor': {'MSE': np.nan, 'R2': np.nan}}, None, None

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, random_state=42)

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train_reg)
    rf_reg_pred = rf_reg.predict(X_test)
    rf_reg_mse = mean_squared_error(y_test_reg, rf_reg_pred)
    rf_reg_r2 = r2_score(y_test_reg, rf_reg_pred)

    # Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(random_state=42)
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
    Runs Logistic Regression Classifier model.
    Args:
        X (pd.DataFrame): Feature data.
        y_binary (pd.Series): Binary target variable (Win_Pct_Binary).
    Returns:
        tuple: Dictionary of model results, trained Logistic Regression model.
    """
    if len(X) < 2 or len(y_binary) < 2:
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    if len(y_binary.unique()) < 2:
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    X_train, X_test, y_train_binary, y_test_binary = train_test_split(X, y_binary, random_state=42)

    if len(y_train_binary.unique()) < 2:
        return {'Logistic Regression Classifier': {'Accuracy': np.nan, 'F1-Score': np.nan}}, None

    lr_clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
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


# ======= Prepare Feature Sets for Comparison =======
feature_sets_for_comparison = {}

# Define fallback feature set if other methods yield empty results
# This will be 'Engineered Features' if available, otherwise OPS+ERA.
# This ensures a default non-empty set is always used if a method fails to select.
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
if not merged[feature_candidates_for_general_selection + [target_col]].empty:
    pearson_corr = merged[feature_candidates_for_general_selection + [target_col]].corr(method='pearson')[target_col].drop(target_col).abs()
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
if not merged[feature_candidates_for_general_selection + [target_col]].empty:
    spearman_corr = merged[feature_candidates_for_general_selection + [target_col]].corr(method='spearman')[target_col].drop(target_col).abs()
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
if not merged[feature_candidates_for_general_selection + [target_col]].empty:
    kendall_corr = merged[feature_candidates_for_general_selection + [target_col]].corr(method='kendall')[target_col].drop(target_col).abs()
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

# 5. RFECV (fallback to engineered_features)
print("\n--- RFECV 遞歸特徵消除與交叉驗證 ---")
selected_features_rfecv = []
if not X_all_features.empty and not y_target_continuous.empty and len(X_all_features) > 1:
    try:
        estimator = RandomForestRegressor(random_state=42)
        # Using R-squared as scoring metric, 5-fold cross-validation
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='r2', n_jobs=-1)
        rfecv.fit(X_all_features, y_target_continuous)

        selected_features_rfecv = X_all_features.columns[rfecv.support_].tolist()
        # Ensure selected features do not include those initially excluded
        selected_features_rfecv = [f for f in selected_features_rfecv if f not in exclude_cols_initial]
        
        if not selected_features_rfecv:
            print("RFECV：沒有特徵被選中。將使用備用特徵集。")
            selected_features_rfecv = fallback_features # Fallback
        else:
            print(f"RFECV 選擇的最佳特徵數量: {len(selected_features_rfecv)}") # Use len of actual selected features
            print(f"RFECV 選擇的特徵: {selected_features_rfecv}")

            # Plot RFECV scoring curve
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
        print(f"RFECV 運行時發生錯誤: {e}。將使用備用特徵集。")
        selected_features_rfecv = fallback_features # Fallback
else:
    print("RFECV：資料量不足以訓練模型。將使用備用特徵集。")
    selected_features_rfecv = fallback_features # Fallback
feature_sets_for_comparison['RFECV'] = selected_features_rfecv

# 6. Engineered Features (always include if available)
if engineered_features:
    feature_sets_for_comparison['Engineered Features'] = engineered_features
    print(f"\n用於比較: 'Engineered Features' -> {engineered_features}")
else:
    print("\n沒有可用的組合指標來進行比較。")

# 7. OPS+ERA Control Group (always include if available)
ops_era_features = [col for col in ['OPS', 'ERA'] if col in merged.columns]
if ops_era_features:
    feature_sets_for_comparison['OPS+ERA (Control)'] = ops_era_features
    print(f"用於比較: 'OPS+ERA (Control)' -> {ops_era_features}")
else:
    print("OPS 或 ERA 欄位不存在，無法建立 OPS+ERA 對照組。")


# Initialize rf_10y, dt_10y for decision tree visualization
rf_10y = None
dt_10y = None

# Set primary_selected_features for decision tree visualization
# This will prioritize Engineered Features, then fall back to OPS+ERA if engineered_features is empty
primary_selected_features = engineered_features if engineered_features else ops_era_features
if not primary_selected_features:
    print("警告: 沒有主要特徵被選中用於一般分析或決策樹可視化。")


# ======= Single Team Yearly Data Analysis (Now only for specified comparison items) =======
teams = merged['Team'].unique()
results_single_regression = []
results_single_classification = []

for team in teams:
    df_team = merged[merged['Team'] == team]

    # Run models for all feature sets in feature_sets_for_comparison
    for fs_name, current_fs_features in feature_sets_for_comparison.items():
        valid_fs_features = [f for f in current_fs_features if f in df_team.columns]
        if len(df_team) >= 5 and valid_fs_features:
            X_fs = df_team[valid_fs_features]
            y_reg = df_team[target_col]
            y_binary = df_team[binary_target_col]

            # Regression Models
            res_reg, _, _ = run_regression_models(X_fs, y_reg)
            for model_name, metrics in res_reg.items():
                results_single_regression.append({'Team': team, 'Feature_Set': fs_name, 'Model': model_name, 'MSE': metrics['MSE'], 'R2': metrics['R2']})

            # Classification Models
            if len(y_binary.unique()) > 1:
                res_clf, _ = run_classification_model(X_fs, y_binary)
                for model_name, metrics in res_clf.items():
                    results_single_classification.append({'Team': team, 'Feature_Set': fs_name, 'Model': model_name, 'Accuracy': metrics['Accuracy'], 'F1-Score': metrics['F1-Score']})
            else:
                # Append NaN results if classification not possible for this team/feature set
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

    # Create 10-year average data for each method
    df_10y_method = merged.groupby('Team')[current_selected_features + [target_col, binary_target_col]].mean().reset_index()

    # Ensure selected features exist in df_10y_method
    valid_selected_features_method = [f for f in current_selected_features if f in df_10y_method.columns]
    if not valid_selected_features_method:
        print(f"警告: {method_name} 在 10 年平均資料中沒有有效特徵，跳過。")
        continue

    X_10y_method = df_10y_method[valid_selected_features_method]
    y_10y_reg = df_10y_method[target_col]
    y_10y_binary = (df_10y_method[target_col] > 0.5).astype(int)

    # Run Regression Models
    if len(X_10y_method) > 1 and len(y_10y_reg) > 1:
        res_reg_method, rf_model_for_tree, dt_model_for_tree = run_regression_models(X_10y_method, y_10y_reg)
        for model_type, metrics in res_reg_method.items():
            all_regression_comparison_results.append({
                'Feature_Set': method_name,
                'Model': model_type,
                'MSE': metrics['MSE'],
                'R2': metrics['R2']
            })
        # For decision tree visualization, we'll retain the model results from 'Engineered Features'
        if method_name == 'Engineered Features':
            rf_10y = rf_model_for_tree
            dt_10y = dt_model_for_tree
    else:
        print(f"警告: {method_name} 數據量不足以訓練回歸模型，跳過。")

    # Run Classification Models
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


# ======= Total MLB Yearly Average Data (Now only for specified comparison items) =======
# Define columns to average: all numerical features and target columns
cols_for_yearly_average = feature_candidates_for_general_selection + engineered_features + [target_col, binary_target_col]
# Ensure uniqueness and presence in merged columns before averaging
cols_for_yearly_average = list(set([col for col in cols_for_yearly_average if col in merged.columns]))

# Perform groupby and mean only on these numerical columns
df_all_10y = merged[cols_for_yearly_average].groupby('year').mean().reset_index()


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

    # Run Regression Models
    if len(X_all_10y) > 1 and len(y_all_10y_reg) > 1:
        res_reg_all_10y, _, _ = run_regression_models(X_all_10y, y_all_10y_reg)
        for model_type, metrics in res_reg_all_10y.items():
            results_all_10y_regression.append({'Team': 'MLB_years', 'Feature_Set': method_name, 'Model': model_type, 'MSE': metrics['MSE'], 'R2': metrics['R2']})
    else:
        print(f"警告: {method_name} 全MLB逐年數據量不足以訓練回歸模型，跳過。")

    # Run Classification Models
    if len(X_all_10y) > 1 and len(y_all_10y_binary) > 1 and len(y_all_10y_binary.unique()) > 1:
        res_clf_all_10y, _ = run_classification_model(X_all_10y, y_all_10y_binary)
        for model_type, metrics in res_clf_all_10y.items():
            results_all_10y_classification.append({'Team': 'MLB_years', 'Feature_Set': method_name, 'Model': model_type, 'Accuracy': metrics['Accuracy'], 'F1-Score': metrics['F1-Score']})
    else:
        print(f"警告: {method_name} 全MLB逐年數據量或類別不足以訓練分類模型，跳過。")

df_all_10y_results_reg = pd.DataFrame(results_all_10y_regression)
df_all_10y_results_clf = pd.DataFrame(results_all_10y_classification)


# ======= Combine All Results (Regression Models) =======
# Now, these plots will include all feature sets in 'feature_sets_for_comparison'
df_all_results_regression = pd.concat([df_single_regression, df_regression_compare, df_all_10y_results_reg], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', hue='Feature_Set', data=df_all_results_regression)
plt.title("迴歸模型 MSE 比較 (不同特徵集)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='R2', hue='Feature_Set', data=df_all_results_regression)
plt.title("迴歸模型 R2 比較 (不同特徵集)")
plt.show()

# ======= Combine All Results (Classification Models) =======
df_all_results_classification = pd.concat([df_single_classification, df_classification_compare, df_all_10y_results_clf], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Accuracy', hue='Feature_Set', data=df_all_results_classification)
plt.title("分類模型準確度比較 (不同特徵集)")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='F1-Score', hue='Feature_Set', data=df_all_results_classification)
plt.title("分類模型 F1-Score 比較 (不同特徵集)")
plt.ylim(0, 1)
plt.show()


# ======= Plot Feature Importance Heatmap (for general candidates, not just selected) =======
# This section still calculates importance based on 'feature_candidates_for_general_selection'
# and may not directly reflect the 'Engineered Features'. Retained for general feature understanding.
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
        plt.title("隨機森林特徵重要性 (所有初始候選特徵)")
        plt.yticks(rotation=0)
        plt.tight_layout()
        importance_heatmap_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/feature_importance_heatmap.png"
        plt.savefig(importance_heatmap_path, dpi=300)
        plt.show()
        print(f"特徵重要性熱力圖已儲存至 {importance_heatmap_path}")
    else:
        print("無法繪製特徵重要性熱力圖: 無法訓練所有候選特徵的隨機森林模型。")
else:
    print("無法繪製特徵重要性熱力圖: 數據不足以處理所有候選特徵。")


# ======= Heatmap: Most Correlated Variables with Win_Pct (Pearson, Spearman, Kendall) =======
# This section also remains for general correlation understanding across all candidates.
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
            plt.title(f"與 {target_col} 最相關特徵 (皮爾森、斯皮爾曼、肯德爾) - 所有初始候選特徵")
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
    print("沒有初始候選特徵可繪製相關性熱力圖。")


# ======= Model Performance Comparison Chart (for Feature Selection Methods) =======
# Now, these plots will include all feature sets in 'feature_sets_for_comparison'
if not df_regression_compare.empty:
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='R2', hue='Model', data=df_regression_compare)
    plt.title("迴歸模型 R² 比較 (不同特徵集)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='MSE', hue='Model', data=df_regression_compare)
    plt.title("迴歸模型 MSE 比較 (不同特徵集)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可比較的迴歸模型結果資料。")

if not df_classification_compare.empty:
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='Accuracy', hue='Model', data=df_classification_compare)
    plt.title("分類模型準確度比較 (不同特徵集)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Feature_Set', y='F1-Score', hue='Model', data=df_classification_compare)
    plt.title("分類模型 F1-Score 比較 (不同特徵集)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可比較的分類模型結果資料。")

# ======= Output Specific Accuracy Data =======
print("\n=== 羅吉斯迴歸分類器準確率比較 ===")
if 'Logistic Regression Classifier' in df_classification_compare['Model'].unique():
    lr_results = df_classification_compare[df_classification_compare['Model'] == 'Logistic Regression Classifier']
    if 'Engineered Features' in lr_results['Feature_Set'].values:
        eng_acc = lr_results[lr_results['Feature_Set'] == 'Engineered Features']['Accuracy'].mean()
        eng_f1 = lr_results[lr_results['Feature_Set'] == 'Engineered Features']['F1-Score'].mean()
        print(f"使用 'Engineered Features' 的羅吉斯迴歸分類器:")
        print(f"  平均準確率 (Accuracy): {eng_acc:.4f}")
        print(f"  平均 F1-Score: {eng_f1:.4f}")
    else:
        print("'Engineered Features' 沒有羅吉斯迴歸分類器結果。")

    if 'OPS+ERA (Control)' in lr_results['Feature_Set'].values:
        ops_era_acc = lr_results[lr_results['Feature_Set'] == 'OPS+ERA (Control)']['Accuracy'].mean()
        ops_era_f1 = lr_results[lr_results['Feature_Set'] == 'OPS+ERA (Control)']['F1-Score'].mean()
        print(f"使用 'OPS+ERA (Control)' 的羅吉斯迴歸分類器:")
        print(f"  平均準確率 (Accuracy): {ops_era_acc:.4f}")
        print(f"  平均 F1-Score: {ops_era_f1:.4f}")
    else:
        print("'OPS+ERA (Control)' 沒有羅吉斯迴歸分類器結果。")
else:
    print("沒有羅吉斯迴歸分類器結果可供比較。")


# ======= Output Regression Model Performance Summary =======
print("\n=== 迴歸模型性能摘要 (Engineered Features vs OPS+ERA) ===")
# Filter for 'Engineered Features' and 'OPS+ERA (Control)' only
summary_reg_df = df_all_results_regression[df_all_results_regression['Feature_Set'].isin(['Engineered Features', 'OPS+ERA (Control)'])].copy()

if not summary_reg_df.empty:
    # Group by Feature_Set and Model, then calculate mean for MSE and R2
    grouped_summary = summary_reg_df.groupby(['Feature_Set', 'Model']).agg(
        Avg_MSE=('MSE', 'mean'),
        Avg_R2=('R2', 'mean')
    ).reset_index()

    for index, row in grouped_summary.iterrows():
        print(f"特徵集: {row['Feature_Set']}, 模型: {row['Model']}:")
        print(f"  平均 MSE: {row['Avg_MSE']:.4f}")
        print(f"  平均 R2: {row['Avg_R2']:.4f}")
else:
    print("沒有可用的迴歸模型性能數據進行摘要比較。")


# ======= Visualize Decision Tree (Using Engineered Features if available, else OPS+ERA) =======
# The decision tree will now be plotted based on 'Engineered Features' or 'OPS+ERA',
# prioritizing 'Engineered Features' if they exist.
final_features_for_tree = []
if engineered_features and all(f in merged.columns for f in engineered_features):
    final_features_for_tree = engineered_features
    print("\n決策樹將使用 'Engineered Features' 進行可視化。")
elif ops_era_features and all(f in merged.columns for f in ops_era_features):
    final_features_for_tree = ops_era_features
    print("\n決策樹將使用 'OPS+ERA (Control)' 進行可視化。")
else:
    print("\n沒有足夠的特徵來繪製決策樹。")


if 'dt_10y' in locals() and dt_10y and 'rf_10y' in locals() and rf_10y and final_features_for_tree:
    # Ensure X_10y for tree is correctly defined and contains final_features_for_tree
    df_10y_for_tree = merged.groupby('Team')[final_features_for_tree + [target_col]].mean().reset_index()
    X_10y_for_tree = df_10y_for_tree[[f for f in final_features_for_tree if f in df_10y_for_tree.columns]]

    if not X_10y_for_tree.empty:
        # 1. Decision Tree (Single)
        plt.figure(figsize=(20, 10))
        plot_tree(dt_10y, feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title(f"決策樹 (max_depth=3) - 特徵: {', '.join(final_features_for_tree)}")
        plt.tight_layout()
        plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/decision_tree_10y.png", dpi=300)
        plt.close()

        plt.figure(figsize=(20, 10))
        plot_tree(rf_10y.estimators_[0], feature_names=X_10y_for_tree.columns.tolist(), filled=True, rounded=True, max_depth=3)
        plt.title(f"隨機森林 - 單一樹 (max_depth=3) - 特徵: {', '.join(final_features_for_tree)}")
        plt.tight_layout()
        plt.savefig(r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/random_forest_tree_10y.png", dpi=300)
        plt.close()
    else:
        print("無法繪製決策樹: 用於繪圖的特徵數據為空。")
else:
    print("無法繪製決策樹: 模型未訓練或最終選取特徵為空。")


# ======= Save Results to Excel File (Multiple Sheets) =======
excel_output_path = r"C:/Users/richc/OneDrive/桌面/專題/mlbreport/mlb_model_results.xlsx"

# Format numerical results
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


# Save to Excel (multiple sheets)
with pd.ExcelWriter(excel_output_path) as writer:
    df_regression_compare_formatted.to_excel(writer, sheet_name='Regression_Feature_Compare', index=False)
    df_classification_compare_formatted.to_excel(writer, sheet_name='Classification_Feature_Compare', index=False)
    df_all_results_regression_formatted.to_excel(writer, sheet_name='AllTeams_Regression', index=False)
    df_all_results_classification_formatted.to_excel(writer, sheet_name='AllTeams_Classification', index=False)

print(f"\n✅ 結果已成功儲存為 Excel 檔：{excel_output_path}")

# --- 訓練用於互動式預測的特定模型 ---
# 1. 使用 Engineered Features 訓練模型
interactive_predictor_model_eng = None
if engineered_features and all(f in merged.columns for f in engineered_features):
    X_for_interactive_model_eng = merged[engineered_features]
    y_for_interactive_model_eng = merged[target_col]

    # Handle NaN values for X and y
    if X_for_interactive_model_eng.isnull().any().any():
        X_for_interactive_model_eng = X_for_interactive_model_eng.fillna(X_for_interactive_model_eng.mean())
    if y_for_interactive_model_eng.isnull().any():
        y_for_interactive_model_eng = y_for_interactive_model_eng.fillna(y_for_interactive_model_eng.mean())

    if not X_for_interactive_model_eng.empty and not y_for_interactive_model_eng.empty and len(X_for_interactive_model_eng) > 1:
        interactive_predictor_model_eng = RandomForestRegressor(random_state=42)
        interactive_predictor_model_eng.fit(X_for_interactive_model_eng, y_for_interactive_model_eng)
        print("\n用於互動式預測的隨機森林模型 (使用 Engineered Features) 已訓練完成。")
    else:
        print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 Engineered Features)。")
else:
    print("\n警告: 沒有組合指標 (Engineered Features) 可用來訓練互動式預測模型。")
interactive_predictor_model_eng = interactive_predictor_model_eng # Ensure it's not None if fallback happened

# 2. 使用 OPS+ERA 訓練模型
interactive_predictor_model_ops_era = None
if ops_era_features and all(f in merged.columns for f in ops_era_features):
    X_for_interactive_model_ops_era = merged[ops_era_features]
    y_for_interactive_model_ops_era = merged[target_col]

    # Handle NaN values for X and y
    if X_for_interactive_model_ops_era.isnull().any().any():
        X_for_interactive_model_ops_era = X_for_interactive_model_ops_era.fillna(X_for_interactive_model_ops_era.mean())
    if y_for_interactive_model_ops_era.isnull().any():
        y_for_interactive_model_ops_era = y_for_interactive_model_ops_era.fillna(y_for_interactive_model_ops_era.mean())

    if not X_for_interactive_model_ops_era.empty and not y_for_interactive_model_ops_era.empty and len(X_for_interactive_model_ops_era) > 1:
        interactive_predictor_model_ops_era = RandomForestRegressor(random_state=42)
        interactive_predictor_model_ops_era.fit(X_for_interactive_model_ops_era, y_for_interactive_model_ops_era)
        print("\n用於互動式預測的隨機森林模型 (使用 OPS+ERA) 已訓練完成。")
    else:
        print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 OPS+ERA)。")
else:
    print("\n警告: OPS 或 ERA 欄位不存在，無法訓練互動式預測模型 (使用 OPS+ERA)。")
interactive_predictor_model_ops_era = interactive_predictor_model_ops_era # Ensure it's not None if fallback happened


# --- 新增互動式預測函數 ---
def run_interactive_prediction(eng_model, ops_era_model, engineered_features_list, ops_era_features_list, merged_data, target_col_name):
    """
    提供互動式介面，讓使用者輸入 OBP, SLG, WHIP, ER, OPS, ERA
    並使用訓練好的模型預測勝率。
    Args:
        eng_model (sklearn.ensemble.RandomForestRegressor): 使用 Engineered Features 訓練好的模型。
        ops_era_model (sklearn.ensemble.RandomForestRegressor): 使用 OPS+ERA 特徵訓練好的模型。
        engineered_features_list (list): 組合特徵的列表 (例如 ['Offensive_Productivity_Index', 'Defensive_Efficiency_Index'])。
        ops_era_features_list (list): OPS 和 ERA 特徵的列表 (例如 ['OPS', 'ERA'])。
        merged_data (pd.DataFrame): 完整的資料集。
        target_col_name (str): 目標變數的名稱。
    """
    print("\n" * 2) # 加入空行增加可讀性
    print("=" * 40)
    print("      ⚾ MLB 勝率互動式預測 ⚾")
    print("=" * 40)
    print("\n請輸入以下原始變數的值，我將為您預測勝率。")
    print("輸入 'exit' 結束程式。")
    print("\n--- 注意：模型的準確性已在上方圖表和 Excel 報告中呈現，單次預測無法判斷絕對準確性。---")

    # 獲取 OBP、SLG、WHIP、ER、OPS、ERA 的平均值作為預設值，以方便使用者輸入
    avg_obp = merged_data['OBP'].mean() if 'OBP' in merged_data.columns else 0.330
    avg_slg = merged_data['SLG'].mean() if 'SLG' in merged_data.columns else 0.420
    avg_whip = merged_data['WHIP'].mean() if 'WHIP' in merged_data.columns else 1.25
    avg_er = merged_data['ER'].mean() if 'ER' in merged_data.columns else 0.5

    avg_ops = merged_data['OPS'].mean() if 'OPS' in merged_data.columns else (avg_obp + avg_slg) # Fallback to sum of OBP/SLG if OPS missing
    avg_era = merged_data['ERA'].mean() if 'ERA' in merged_data.columns else avg_er # Fallback to ER if ERA missing


    while True:
        try:
            print("\n" + "-" * 30)
            print("請輸入用於 'Engineered Features' 模型的原始數據:")
            obp_input_str = input(f"  上壘率 (OBP, 建議值 {avg_obp:.3f}): ")
            if obp_input_str.lower() == 'exit': break
            obp_input = float(obp_input_str)

            slg_input_str = input(f"  長打率 (SLG, 建議值 {avg_slg:.3f}): ")
            if slg_input_str.lower() == 'exit': break
            slg_input = float(slg_input_str)

            print("\n請輸入用於 'OPS+ERA (Control)' 模型的原始數據:")
            ops_input_str = input(f"  OPS (建議值 {avg_ops:.3f}): ")
            if ops_input_str.lower() == 'exit': break
            ops_input = float(ops_input_str)
            
            whip_input_str = input(f"  每局被上壘數 (WHIP, 建議值 {avg_whip:.2f}): ")
            if whip_input_str.lower() == 'exit': break
            whip_input = float(whip_input_str)

            er_input_str = input(f"  自責分 (ER, 建議值 {avg_er:.2f}): ")
            if er_input_str.lower() == 'exit': break
            er_input = float(er_input_str)

            era_input_str = input(f"  防禦率 (ERA, 建議值 {avg_era:.2f}): ")
            if era_input_str.lower() == 'exit': break
            era_input = float(era_input_str)


            # --- 計算 Engineered Features 的組合指標 ---
            offensive_prod_index = obp_input + slg_input
            defensive_eff_index = whip_input + er_input

            # --- 準備 Engineered Features 模型的輸入數據 ---
            input_values_eng = {}
            if 'Offensive_Productivity_Index' in engineered_features_list:
                input_values_eng['Offensive_Productivity_Index'] = offensive_prod_index
            if 'Defensive_Efficiency_Index' in engineered_features_list:
                input_values_eng['Defensive_Efficiency_Index'] = defensive_eff_index
            input_df_eng = pd.DataFrame([input_values_eng])

            # --- 準備 OPS+ERA (Control) 模型的輸入數據 ---
            input_values_ops_era = {}
            if 'OPS' in ops_era_features_list:
                input_values_ops_era['OPS'] = ops_input
            if 'ERA' in ops_era_features_list:
                input_values_ops_era['ERA'] = era_input
            input_df_ops_era = pd.DataFrame([input_values_ops_era])


            print(f"\n--- 預測結果 ---")
            print(f"原始輸入: OBP={obp_input:.3f}, SLG={slg_input:.3f}, WHIP={whip_input:.2f}, ER={er_input:.2f}, OPS={ops_input:.3f}, ERA={era_input:.2f}")
            
            # --- 使用 Engineered Features 模型預測 ---
            predicted_win_pct_eng = np.nan
            if eng_model and not input_df_eng.empty:
                try:
                    predicted_win_pct_eng = eng_model.predict(input_df_eng)[0]
                    print(f"使用 'Engineered Features' 預測勝率: {predicted_win_pct_eng:.4f}")
                except Exception as e:
                    print(f"使用 'Engineered Features' 預測失敗: {e}")
            else:
                print("使用 'Engineered Features' 無法進行預測 (模型未訓練或特徵不足)。")

            # --- 使用 OPS+ERA (Control) 模型預測 ---
            predicted_win_pct_ops_era = np.nan
            if ops_era_model and not input_df_ops_era.empty:
                try:
                    predicted_win_pct_ops_era = ops_era_model.predict(input_df_ops_era)[0]
                    print(f"使用 'OPS+ERA (Control)' 預測勝率: {predicted_win_pct_ops_era:.4f}")
                except Exception as e:
                    print(f"使用 'OPS+ERA (Control)' 預測失敗: {e}")
            else:
                print("使用 'OPS+ERA (Control)' 無法進行預測 (模型未訓練或特徵不足)。")

            print("-" * 30)

        except ValueError:
            print("輸入無效。請輸入數字。")
        except Exception as e:
            print(f"預測時發生錯誤: {e}")

    print("\n預測程式結束。")

# --- 在所有分析和圖表生成後，運行互動式預測 ---
# 首先，確保 engineered_features 和 ops_era_features 都至少有一個元素，否則無法訓練模型。
if not engineered_features and not ops_era_features:
    print("\n警告: 沒有任何特徵可用來訓練預測模型。互動式預測無法啟動。")
else:
    # 訓練用於互動式預測的特定模型 (使用 Engineered Features)
    interactive_predictor_model_eng = None
    if engineered_features and all(f in merged.columns for f in engineered_features):
        X_for_interactive_model_eng = merged[engineered_features]
        y_for_interactive_model_eng = merged[target_col]

        if not X_for_interactive_model_eng.empty and not y_for_interactive_model_eng.empty and len(X_for_interactive_model_eng) > 1:
            # 處理 NaN 值，以防萬一
            if X_for_interactive_model_eng.isnull().any().any():
                X_for_interactive_model_eng = X_for_interactive_model_eng.fillna(X_for_interactive_model_eng.mean())
            if y_for_interactive_model_eng.isnull().any():
                y_for_interactive_model_eng = y_for_interactive_model_eng.fillna(y_for_interactive_model_eng.mean())

            interactive_predictor_model_eng = RandomForestRegressor(random_state=42)
            interactive_predictor_model_eng.fit(X_for_interactive_model_eng, y_for_interactive_model_eng)
            print("\n用於互動式預測的隨機森林模型 (使用 Engineered Features) 已訓練完成。")
        else:
            print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 Engineered Features)。")
    else:
        print("\n警告: 沒有組合指標 (Engineered Features) 可用來訓練互動式預測模型。")

    # 訓練用於互動式預測的特定模型 (使用 OPS+ERA)
    interactive_predictor_model_ops_era = None
    if ops_era_features and all(f in merged.columns for f in ops_era_features):
        X_for_interactive_model_ops_era = merged[ops_era_features]
        y_for_interactive_model_ops_era = merged[target_col]

        if not X_for_interactive_model_ops_era.empty and not y_for_interactive_model_ops_era.empty and len(X_for_interactive_model_ops_era) > 1:
            # 處理 NaN 值，以防萬一
            if X_for_interactive_model_ops_era.isnull().any().any():
                X_for_interactive_model_ops_era = X_for_interactive_model_ops_era.fillna(X_for_interactive_model_ops_era.mean())
            if y_for_interactive_model_ops_era.isnull().any():
                y_for_interactive_model_ops_era = y_for_interactive_model_ops_era.fillna(y_for_interactive_model_ops_era.mean())

            interactive_predictor_model_ops_era = RandomForestRegressor(random_state=42)
            interactive_predictor_model_ops_era.fit(X_for_interactive_model_ops_era, y_for_interactive_model_ops_era)
            print("\n用於互動式預測的隨機森林模型 (使用 OPS+ERA) 已訓練完成。")
        else:
            print("\n警告: 數據不足以訓練用於互動式預測的隨機森林模型 (使用 OPS+ERA)。")
    else:
        print("\n警告: OPS 或 ERA 欄位不存在，無法訓練互動式預測模型 (使用 OPS+ERA)。")

    # 只有當至少一個模型訓練成功時才運行互動式預測
    if interactive_predictor_model_eng or interactive_predictor_model_ops_era:
        run_interactive_prediction(interactive_predictor_model_eng, interactive_predictor_model_ops_era, engineered_features, ops_era_features, merged, target_col)
    else:
        print("\n未啟動互動式預測，因為沒有可用的模型進行預測。")
