import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import os

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
    X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
    return X

class CustomLoglossObjective(object):
    def __init__(self, penalty: float = 1.3, alpha: float = 0.5, mrf_weight: float = 0.5, reward_factor: float = 1.8):
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

            q = 0.5
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
                                               initial_labeled_samples: int = 50, 
                                               num_iterations: int = 32, 
                                               num_samples_to_label: int = 10) -> tuple:
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    n_samples = len(X_train_np)

    X_labeled = X_train_np[:initial_labeled_samples]
    y_labeled = y_train_np[:initial_labeled_samples]
    labeled_indices = set(range(min(initial_labeled_samples, n_samples)))

    custom_loss = CustomLoglossObjective()
    catboost_model = CatBoostClassifier(cat_features=list(range(13)), 
                                        loss_function=custom_loss, 
                                        eval_metric='Logloss', 
                                        depth=8, 
                                        iterations=400, 
                                        learning_rate=0.03,
                                        verbose=0)
    rf_model = RandomForestClassifier(n_estimators=100)

    for _ in range(num_iterations):
        unlabeled_indices = np.array([i for i in range(n_samples) if i not in labeled_indices], dtype=int)
        X_unlabeled = X_train_np[unlabeled_indices]
        y_unlabeled = y_train_np[unlabeled_indices]

        if len(X_unlabeled) < num_samples_to_label:
            break

        catboost_model.fit(X_labeled, y_labeled)
        rf_model.fit(X_labeled, y_labeled)

        pred_cat = catboost_model.predict_proba(X_unlabeled)
        pred_rf = rf_model.predict_proba(X_unlabeled)

        disagreements = np.abs(pred_cat - pred_rf).mean(axis=1)

        selected_indices = np.argsort(disagreements)[-num_samples_to_label:]
        selected_original_indices = unlabeled_indices[selected_indices]

        X_labeled = np.concatenate((X_labeled, X_unlabeled[selected_indices]))
        y_labeled = np.concatenate((y_labeled, y_unlabeled[selected_indices]))
        labeled_indices.update(selected_original_indices.tolist())

    removed_indices = np.array([i for i in range(n_samples) if i not in labeled_indices], dtype=int)
    X_removed = X_train_np[removed_indices] if len(removed_indices) > 0 else np.empty((0, X_train_np.shape[1]))
    y_removed = y_train_np[removed_indices] if len(removed_indices) > 0 else np.empty((0,))

    X_labeled_bald = pd.DataFrame(X_labeled, columns=X_train.columns)
    
    for col in X_train.columns:
        X_labeled_bald[col] = X_labeled_bald[col].astype(X_train[col].dtype)
        
    y_labeled_bald = pd.DataFrame(y_labeled, columns=['OS'])

    return X_labeled_bald, y_labeled_bald, X_removed, y_removed

def perform_shap_analysis(model, X_test: pd.DataFrame, save_dir: str = "data/result"):
    os.makedirs(save_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    explanation = explainer(X_test)

    # 1. 提取用于计算重要性的 SHAP 值矩阵
    # 针对二分类，CatBoost 可能返回单一数组或长度为2的列表
    if isinstance(shap_values, list):
        shap_values_for_importances = shap_values[1] 
    else:
        shap_values_for_importances = shap_values

    # 2. 计算每个特征的平均绝对 SHAP 值 (Mean |SHAP|)
    mean_abs_shap = np.abs(shap_values_for_importances).mean(axis=0)
    
    # 3. 构建特征重要性 DataFrame 并降序排列
    feature_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    # 绘制 Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Global Interpretability)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制 Waterfall Plot
    if len(X_test) > 0:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], show=False)
        plt.title("SHAP Waterfall Plot for Patient 0", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_waterfall_patient_0.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 将特征重要性排序返回
    return feature_importance_df

def main():
    # 1. 设置路径 (保持你的原始路径)
    path1 = r"D:\Desktop\RA\Esophageal-Cancer-Model-Code-main\data\EC_before_Treatment 7.12 OS external validation.csv"
    path2 = r"D:\Desktop\RA\Esophageal-Cancer-Model-Code-main\data\EC_before_Treatment 7.12 OS external validation.csv"

    # 2-4. 数据加载与预处理
    data, final_data_0S_before = load_and_preprocess_data(path1, path2)
    data_no_outliers = remove_outliers(data, threshold=4)
    X_final = remove_high_correlation_features(data_no_outliers, threshold=0.8)
    
    X = X_final.drop(['OS', 'OS_m'], axis=1, errors='ignore') 
    y = X_final['OS']
    X = standardize_selected_features(X)

    # 5. 优化后的数据类型转换
    cat_features_indices = list(range(13))
    cat_cols = X.columns[cat_features_indices].tolist() # 存为列表方便后续匹配
    X[cat_cols] = X[cat_cols].astype(str)
    
    num_cols = X.columns[13:]
    X[num_cols] = X[num_cols].astype(float)
    y = y.astype(float).astype(int)

    # 6. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7-10. 主动学习筛选 (为了快速测试，这里可以适当调低迭代次数，实际运行保持你的配置)
    print("\nStarting Active Learning sample selection...")
    X_labeled_bald, y_labeled_bald, _, _ = active_learning_sample_selection_with_bald(
        X_train, y_train, initial_labeled_samples=50, num_iterations=32, num_samples_to_label=10
    )

    # 11. 训练完整模型 (Full Model, 41特征)
    print("\nTraining Full weighted-CatBoost Model (All Features)...")
    y_labeled_final = y_labeled_bald['OS'].astype(float).astype(int)
    
    full_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=8,
        loss_function=CustomLoglossObjective(),
        eval_metric='Logloss',
        cat_features=cat_features_indices,
        verbose=0  # 关闭打印以保持输出清爽
    )
    full_model.fit(X_labeled_bald, y_labeled_final)

    # 12. 运行 SHAP 并获取特征重要性
    print("\nGenerating SHAP plots and extracting feature importances...")
    feature_importance_df = perform_shap_analysis(full_model, X_test)
    
    # =========================================================================
    # 13. 构建轻量级模型 (Lite Model, Top-15 特征)
    # =========================================================================
    # 提取排名前 15 的特征
    top_15_features = feature_importance_df['feature'].head(15).tolist()
    print(f"\nTop 15 Features selected by SHAP:\n{top_15_features}")
    
    # 截取训练集和测试集的 Top-15 子集
    X_labeled_lite = X_labeled_bald[top_15_features]
    X_test_lite = X_test[top_15_features]
    
    # 动态获取 Lite 模型中分类特征的索引位置
    cat_features_lite_indices = [i for i, col in enumerate(top_15_features) if col in cat_cols]
    
    print("\nTraining Lite weighted-CatBoost Model (Top-15 Features)...")
    lite_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=8,
        loss_function=CustomLoglossObjective(),
        eval_metric='Logloss',
        cat_features=cat_features_lite_indices,
        verbose=0
    )
    lite_model.fit(X_labeled_lite, y_labeled_final)

    # =========================================================================
    # 14. 评估并在终端输出 AUC 对比结果
    # =========================================================================
    # 预测正类 (生存风险/状态) 概率
    pred_full = full_model.predict_proba(X_test)[:, 1]
    pred_lite = lite_model.predict_proba(X_test_lite)[:, 1]

    auc_full = roc_auc_score(y_test, pred_full)
    auc_lite = roc_auc_score(y_test, pred_lite)

    print("\n================= PERFORMANCE COMPARISON =================")
    print(f"Full Model (All Features) Test AUC:  {auc_full:.4f}")
    print(f"Lite Model (Top 15 Features) Test AUC: {auc_lite:.4f}")
    print("==========================================================")
    
    if abs(auc_full - auc_lite) < 0.03:
        print("Conclusion: The Lite model achieves comparable performance to the Full model,\n"
              "proving that the top 15 SHAP features successfully capture the core survival patterns!")

if __name__ == "__main__":
    main()
