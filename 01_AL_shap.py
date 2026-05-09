import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import os
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(path1: str, path2: str) -> tuple:
    data = pd.read_csv(path1).dropna()
    final_data_0S_before = pd.read_csv(path2).dropna()
    return data, final_data_0S_before

def remove_outliers(data: pd.DataFrame, threshold: float = 4) -> pd.DataFrame:
    z_score = stats.zscore(data)
    outliers = (z_score > threshold).any(axis=1)
    data_no_outliers = data[~outliers].dropna()
    return data_no_outliers

def remove_high_correlation_features(data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_final = data.drop(to_drop, axis=1)
    return X_final

def standardize_selected_features(X: pd.DataFrame, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['treatment', 'TL', 'original']
    scaler = StandardScaler()
    columns_to_standardize = [col for col in X.columns if any(kw in col for kw in keywords)]
    if columns_to_standardize:
        X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
    return X

class CustomLoglossObjective(object):
    def __init__(self, penalty: float = 2, alpha: float = 0.5, mrf_weight: float = 0.5, reward_factor: float = 0.9):
        self.alpha = alpha
        self.penalty = penalty
        self.mrf_weight = mrf_weight
        self.reward_factor = reward_factor

    def calc_ders_range(self, approxes: list, targets: list, weights: list = None) -> list:
        assert len(approxes) == len(targets)
        exponents = [np.exp(a) for a in approxes]
        result = []
        for idx in range(len(targets)):
            p = exponents[idx] / (1 + exponents[idx])
            penalty = 1.0

            if (targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8):
                penalty *= self.penalty

            reward_factor = 1.0
            if targets[idx] == 1 and p > 0.8:
                reward_factor *= self.reward_factor
            elif targets[idx] == 0 and p < 0.2:
                reward_factor *= self.reward_factor

            der1_log = (1 - p) * self.penalty if targets[idx] > 0.0 else -p * self.penalty
            der2_log = -p * (1 - p)

            q = 0.6
            der1_quantile = q * (p - p**2) if targets[idx] - p >= 0 else (q - 1) * (p - p**2)
            der2_quantile = 0

            der1_combined = (der1_quantile + der1_log) / 2
            der2_combined = (der2_quantile + der2_log) / 2

            if weights is not None:
                der1_combined *= weights[idx]
                der2_combined *= weights[idx]

            result.append((der1_combined, der2_combined))
        return result

def active_learning_sample_selection_with_bald(X_train: pd.DataFrame, y_train: pd.Series, 
                                               cat_cols: list,
                                               initial_labeled_samples: int = 50, 
                                               num_iterations: int = 32, 
                                               num_samples_to_label: int = 10) -> tuple:
    n_samples = len(X_train)
    is_labeled = np.zeros(n_samples, dtype=bool)
    is_labeled[:initial_labeled_samples] = True

    custom_loss = CustomLoglossObjective()
    catboost_al = CatBoostClassifier(loss_function=custom_loss, 
                                        eval_metric='Logloss', depth=8, iterations=400, 
                                        learning_rate=0.03, verbose=0, random_seed=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    for _ in range(num_iterations):
        X_labeled = X_train[is_labeled]
        y_labeled = y_train[is_labeled]
        X_unlabeled = X_train[~is_labeled]
        
        if len(X_unlabeled) < num_samples_to_label:
            break

        train_pool = Pool(X_labeled, y_labeled, cat_features=cat_cols)
        catboost_al.fit(train_pool, verbose=0) 
        rf_model.fit(X_labeled.astype(float), y_labeled)

        unlabeled_pool = Pool(X_unlabeled, cat_features=cat_cols)
        pred_cat = catboost_al.predict_proba(unlabeled_pool)
        pred_rf = rf_model.predict_proba(X_unlabeled.astype(float))

        disagreements = np.abs(pred_cat - pred_rf).mean(axis=1)
        
        selected_local_idx = np.argsort(disagreements)[-num_samples_to_label:]
        unlabeled_global_idx = np.where(~is_labeled)[0]
        selected_global_idx = unlabeled_global_idx[selected_local_idx]
        
        is_labeled[selected_global_idx] = True

    X_labeled_final = X_train[is_labeled]
    y_labeled_final = y_train[is_labeled]

    return X_labeled_final, pd.DataFrame(y_labeled_final, columns=['OS'])

# ================= 恢复缺失的可视化图表 =================
def plot_catboost_learning_curve(model, save_dir: str):
    evals_result = model.get_evals_result()
    if 'learn' in evals_result and 'validation' in evals_result:
        train_loss = evals_result['learn']['Logloss']
        val_loss = evals_result['validation']['Logloss']
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train Logloss', color='blue', linewidth=2)
        plt.plot(val_loss, label='Validation Logloss', color='orange', linewidth=2)
        plt.xlabel('Number of Iterations', fontsize=16)
        plt.ylabel('Logloss', fontsize=16)
        plt.title('CatBoost Learning Curve', fontsize=20)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=300)
        plt.show()

def plot_top_features_bar(feature_importance_df, top_n: int = 15, save_dir: str = "data/result"):
    top_features = feature_importance_df.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'][::-1], color='skyblue', align='center')
    plt.yticks(range(len(top_features)), top_features['feature'][::-1], fontsize=14)
    plt.xlabel('SHAP Importance (Mean |SHAP|)', fontsize=16)
    plt.title(f'Top {top_n} Important Features', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_features_bar.png"), dpi=300)
    plt.show()
# =========================================================

def perform_shap_analysis(model, pool: Pool, X_df: pd.DataFrame, save_dir: str = "data/result"):
    os.makedirs(save_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)
    explanation = explainer(pool)

    shap_vals_importances = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_vals_importances).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': X_df.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("SHAP Summary Plot (Global Interpretability)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()

    plot_top_features_bar(feature_importance_df, save_dir=save_dir)

    return feature_importance_df

def main():
    path1 = r"D:\Desktop\RA\Esophageal-Cancer-Model-Code-main\data\EC_before_Treatment 7.12 OS external validation.csv"
    path2 = r"D:\Desktop\RA\Esophageal-Cancer-Model-Code-main\data\EC_before_Treatment 7.12 OS external validation.csv"

    data, final_data_0S_before = load_and_preprocess_data(path1, path2)
    data_no_outliers = remove_outliers(data, threshold=4)
    X_final = remove_high_correlation_features(data_no_outliers, threshold=0.8)
    
    X = X_final.drop(['OS', 'OS_m', 'PFS_m'], axis=1, errors='ignore') 
    y = X_final['OS'].astype(float).astype(int)
    X = standardize_selected_features(X)

    TARGET_CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose', 'ECOG', 'T', 'Chemotherapy']
    cat_cols = [col for col in X.columns if col in TARGET_CAT_FEATURES]
    
    X[cat_cols] = X[cat_cols].astype(float).astype(int).astype(str)
    num_cols = [col for col in X.columns if col not in cat_cols]
    X[num_cols] = X[num_cols].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nStarting Active Learning sample selection...")
    X_labeled_bald, y_labeled_bald = active_learning_sample_selection_with_bald(
        X_train, y_train, cat_cols,
        initial_labeled_samples=50, num_iterations=32, num_samples_to_label=10
    )[:2]

    print("\nTraining Full weighted-CatBoost Model (All Features)...")
    y_labeled_final = y_labeled_bald['OS'].astype(int)
    
    full_model = CatBoostClassifier(
        iterations=400, learning_rate=0.1, early_stopping_rounds=50, depth=8,
        loss_function=CustomLoglossObjective(), eval_metric='Logloss',
        cat_features=cat_cols, random_seed=42, verbose=0 
    )

    train_pool_full = Pool(X_labeled_bald, y_labeled_final, cat_features=cat_cols)
    test_pool_full = Pool(X_test, y_test, cat_features=cat_cols)
    full_model.fit(train_pool_full, eval_set=test_pool_full)
    
    save_dir = "data/result"
    os.makedirs(save_dir, exist_ok=True)
    plot_catboost_learning_curve(full_model, save_dir)

    print("\nGenerating SHAP plots and extracting feature importances...")
    feature_importance_df = perform_shap_analysis(full_model, train_pool_full, X_labeled_bald, save_dir)
    
    top_15_features = feature_importance_df['feature'].head(15).tolist()
    print(f"\nTop 15 Features selected by SHAP:\n{top_15_features}")
    
    X_labeled_lite = X_labeled_bald[top_15_features]
    X_test_lite = X_test[top_15_features]
    cat_cols_lite = [col for col in top_15_features if col in cat_cols]
    
    print("\nTraining Lite weighted-CatBoost Model (Top-15 Features)...")
    lite_model = CatBoostClassifier(
        iterations=400, learning_rate=0.1, early_stopping_rounds=50, depth=8,
        loss_function=CustomLoglossObjective(), eval_metric='Logloss',
        cat_features=cat_cols_lite, random_seed=42, verbose=0
    )

    train_pool_lite = Pool(X_labeled_lite, y_labeled_final, cat_features=cat_cols_lite)
    test_pool_lite = Pool(X_test_lite, y_test, cat_features=cat_cols_lite)
    lite_model.fit(train_pool_lite, eval_set=test_pool_lite)

    # 导出数据时，包含时间列
    print("\nExporting frozen splits for downstream scripts...")
    train_export = X_labeled_bald.copy()
    train_export['OS'] = y_labeled_final.values
    if 'OS_m' in X_final.columns: train_export['OS_m'] = X_final.loc[X_labeled_bald.index, 'OS_m'].values
    if 'PFS_m' in X_final.columns: train_export['PFS_m'] = X_final.loc[X_labeled_bald.index, 'PFS_m'].values
    train_export.to_csv(os.path.join(save_dir, "labeled_train.csv"), index=False)
    
    test_export = X_test.copy()
    test_export['OS'] = y_test.values
    if 'OS_m' in X_final.columns: test_export['OS_m'] = X_final.loc[X_test.index, 'OS_m'].values
    if 'PFS_m' in X_final.columns: test_export['PFS_m'] = X_final.loc[X_test.index, 'PFS_m'].values

    test_export.to_csv(os.path.join(save_dir, "test_validation.csv"), index=False)
    print(f"✅ Data explicitly saved to '{save_dir}/labeled_train.csv' and 'test_validation.csv'")

if __name__ == "__main__":
    main()
