import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap
import os

# Configuration class
class ModelConfig:
    # 请替换为你的 01_Data_and_SHAP.py 跑出来的干净数据的路径
    # 建议使用主动学习筛选后的 remained.csv
    DATA_PATH = r"data/result/remained.csv" 
    
    CATBOOST_PARAMS = {
        'depth': 8,
        'iterations': 400,
        'learning_rate': 0.03,
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': 0
    }
    
    # 需要被指定为分类变量的特征列表（请根据你的实际数据调整）
    CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'ECOG', 'T', 'Chemotherapy']
    
    PLOT_FIGSIZE = (10, 8)
    PLOT_TITLE_FONTSIZE = 22
    PLOT_LABEL_FONTSIZE = 16
    
    LITE_FEATURES_COUNT = 15  # Lite 模型保留的特征数

# Custom Logloss Objective (保持你最新优化的版本)
class CustomLoglossObjective(object):
    def __init__(self, penalty=1.3, reward_factor=1.8):
        self.penalty = penalty
        self.reward_factor = reward_factor

    def calc_ders_range(self, approxes, targets, weights=None):
        assert len(approxes) == len(targets)
        exponents = [np.exp(a) for a in approxes]
        result = []
        for idx in range(len(targets)):
            p = exponents[idx] / (1 + exponents[idx])
            penalty = 1.0

            if (targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8):
                penalty *= self.penalty

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

def calculate_roc_auc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    return fpr, tpr, auc_score

def plot_unified_roc_curve(roc_data: dict, title: str, config: ModelConfig):
    plt.figure(figsize=config.PLOT_FIGSIZE)
    
    # 为不同的模型设置不同的颜色和线型，突出 CatBoost
    colors = ['red', 'darkorange', 'blue', 'green', 'purple', 'brown', 'cyan']
    
    for (label, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        linewidth = 3 if 'CatBoost' in label else 1.5
        linestyle = '-' if 'Full' in label else ('--' if 'Lite' in label else ':')
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.3f})', 
                 color=color, linewidth=linewidth, linestyle=linestyle)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate', fontsize=config.PLOT_LABEL_FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=config.PLOT_LABEL_FONTSIZE)
    plt.title(title, fontsize=config.PLOT_TITLE_FONTSIZE)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def main():
    config = ModelConfig()
    
    # 1. 加载主动学习清洗后的数据
    print("Loading prepared data...")
    data = pd.read_csv(config.DATA_PATH)
    
    # 假设标签列是 'OS' 或者是 'PFS'
    label_col = 'OS' if 'OS' in data.columns else 'PFS'
    X = data.drop([label_col], axis=1, errors='ignore')
    y = data[label_col].astype(float).astype(int)
    
    # 动态匹配分类特征
    actual_cat_features = [col for col in config.CAT_FEATURES if col in X.columns]
    X[actual_cat_features] = X[actual_cat_features].astype(str)
    
    # 将其他特征转为 float
    num_features = [col for col in X.columns if col not in actual_cat_features]
    X[num_features] = X[num_features].astype(float)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features_indices = [X.columns.get_loc(col) for col in actual_cat_features]

    # ==========================================
    # 实验 1: 训练 CatBoost (Full Model - 41 特征)
    # ==========================================
    print(f"\nTraining CatBoost (Full Model) with {X_train.shape[1]} features...")
    full_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=cat_features_indices,
        **config.CATBOOST_PARAMS
    )
    full_model.fit(X_train, y_train)
    full_probs = full_model.predict_proba(X_test)[:, 1]
    
    # ==========================================
    # 实验 2: 基于 SHAP 提取 Top-15 特征，训练 Lite 模型
    # ==========================================
    print("\nCalculating SHAP values to extract Top-15 features for Lite Model...")
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X_train)
    shap_values_for_importances = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_values_for_importances).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)
    
    top_15_features = feature_importance_df['feature'].head(config.LITE_FEATURES_COUNT).tolist()
    print(f"Top 15 Features selected: {top_15_features}")
    
    # 截取 Lite 数据集
    X_train_lite = X_train[top_15_features]
    X_test_lite = X_test[top_15_features]
    lite_cat_features_indices = [i for i, col in enumerate(top_15_features) if col in actual_cat_features]
    
    print(f"\nTraining CatBoost (Lite Model) with {config.LITE_FEATURES_COUNT} features...")
    lite_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=lite_cat_features_indices,
        **config.CATBOOST_PARAMS
    )
    lite_model.fit(X_train_lite, y_train)
    lite_probs = lite_model.predict_proba(X_test_lite)[:, 1]

    # ==========================================
    # 实验 3: 训练传统 Baseline 模型 (使用全量特征)
    # ==========================================
    print("\nTraining Baseline models (RandomForest, SVM, LogisticRegression, KNN, AdaBoost)...")
    
    # 基线模型无法处理字符串类别，需要进行 One-Hot 编码或标签编码
    # 为简单起见，我们对基线模型的数据进行 get_dummies 独热编码处理
    X_train_encoded = pd.get_dummies(X_train, columns=actual_cat_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=actual_cat_features, drop_first=True)
    
    # 确保训练集和测试集的列完全一致
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    # 填补可能的缺失值（由于对齐操作）并标准化基线模型数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    baselines = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }

    roc_results = {}
    # 保存 CatBoost 的结果
    roc_results['weighted-CatBoost (Full)'] = calculate_roc_auc(y_test, full_probs)
    roc_results['weighted-CatBoost (Lite)'] = calculate_roc_auc(y_test, lite_probs)
    
    # 保存基线模型结果
    for name, model in baselines.items():
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        roc_results[name] = calculate_roc_auc(y_test, probs)

    # ==========================================
    # 4. 绘制终极消融实验 ROC 对比图
    # ==========================================
    print("\nPlotting unified Ablation Experiment ROC Curve...")
    plot_unified_roc_curve(roc_results, "Ablation Study: Model & Feature Comparison (Test Set)", config)

if __name__ == '__main__':
    main()