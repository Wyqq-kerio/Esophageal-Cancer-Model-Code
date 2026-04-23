import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. Configuration Class
# ------------------------------------------------------------------------------
class ClinicalConfig:
    """Configuration for Clinical Evaluation (Survival, DCA, Calibration)"""
    # 替换为 01_Data_and_SHAP 跑出的数据路径
    DATA_PATH = r"data/result/labeled.csv" 
    
    # 标签配置（根据你的任务选择 'OS' 或 'PFS'）
    TARGET_EVENT = "OS"       # 生存状态 (0/1)
    TARGET_TIME = "OS_m"      # 生存时间 (月)
    TIME_CANDIDATES = ["OS_m", "PFS_m"]  # 自动兼容可用的生存时间列
    
    LITE_FEATURES_COUNT = 15  # 轻量级模型特征数
    
    CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'ECOG', 'T', 'Chemotherapy']
    
    CATBOOST_PARAMS = {
        'depth': 8,
        'iterations': 400,
        'learning_rate': 0.03,
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': 0
    }

    # Plot settings
    FONT_CONFIG = {'family': 'Times New Roman', 'fontsize': 18}
    DCA_THRESH_RANGE = (0, 1, 0.01)

# ------------------------------------------------------------------------------
# 2. Custom Objective (保持你的优化版本)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 3. Clinical Evaluation Functions (DCA & Calibration)
# ------------------------------------------------------------------------------
def calculate_net_benefit(thresh_group, y_pred_score, y_label):
    """计算模型的净收益 (Net Benefit)"""
    net_benefit = []
    n = len(y_label)
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label, labels=[0, 1]).ravel()
        if (1 - thresh) == 0:
            nb = 0.0
        else:
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit.append(nb)
    return np.array(net_benefit)

def calculate_net_benefit_all(thresh_group, y_label):
    """计算 Treat All 策略的净收益"""
    net_benefit = []
    tp_total = np.sum(y_label == 1)
    total = len(y_label)
    fp_total = total - tp_total
    for thresh in thresh_group:
        if (1 - thresh) == 0:
            nb = 0.0
        else:
            nb = (tp_total / total) - (fp_total / total) * (thresh / (1 - thresh))
        net_benefit.append(nb)
    return np.array(net_benefit)

def plot_dca_curve(y_true, prob_full, prob_lite, title):
    """绘制 DCA 决策曲线"""
    thresh_group = np.arange(*ClinicalConfig.DCA_THRESH_RANGE)
    
    nb_full = calculate_net_benefit(thresh_group, prob_full, y_true)
    nb_lite = calculate_net_benefit(thresh_group, prob_lite, y_true)
    nb_all = calculate_net_benefit_all(thresh_group, y_true)

    plt.figure(figsize=(10, 6))
    plt.plot(thresh_group, nb_full, color='red', linewidth=2, label='CatBoost (Full Model)')
    plt.plot(thresh_group, nb_lite, color='orange', linewidth=2, linestyle='--', label='CatBoost (Lite Model)')
    plt.plot(thresh_group, nb_all, color='slategrey', label='Treat All')
    plt.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat None')

    plt.xlim(0, 1)
    plt.ylim(min(nb_full.min(), -0.05), max(nb_full.max(), 0.1) + 0.1)
    plt.xlabel('Threshold Probability', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.ylabel('Net Benefit', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.title(title, fontsize=20, fontname='Times New Roman')
    plt.legend(loc='lower left')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(y_true, prob_full, prob_lite, title):
    """绘制校准曲线"""
    plt.figure(figsize=(8, 8))
    
    # Full Model
    fop_full, mpv_full = calibration_curve(y_true, prob_full, n_bins=5, strategy='quantile')
    plt.plot(mpv_full, fop_full, marker='o', color='red', linewidth=2, label='CatBoost (Full Model)')
    
    # Lite Model
    fop_lite, mpv_lite = calibration_curve(y_true, prob_lite, n_bins=5, strategy='quantile')
    plt.plot(mpv_lite, fop_lite, marker='s', color='orange', linewidth=2, linestyle='--', label='CatBoost (Lite Model)')
    
    # Perfect calibration
    plt.plot([0, 1], [0, 1], linestyle=':', color='black', label='Perfect Calibration')

    plt.xlabel('Predicted Probability', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.ylabel('Observed Fraction of Positives', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.title(title, fontsize=20, fontname='Times New Roman')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 4. Survival Analysis Functions (KM & C-Index)
# ------------------------------------------------------------------------------
def perform_survival_analysis(time_true, event_true, risk_scores, median_risk, title_prefix):
    """
    计算 C-index 并绘制 Kaplan-Meier 生存曲线
    使用训练集的风险中位数将患者划分为高风险与低风险组
    """
    # 1. 计算 C-index (注意：风险越高，生存期越短，需要传入 risk_scores)
    c_index = concordance_index(time_true, risk_scores, event_true)
    print(f"[{title_prefix}] Concordance Index (C-index): {c_index:.3f}")

    # 2. 划分高低风险组
    high_risk_mask = risk_scores >= median_risk
    low_risk_mask = risk_scores < median_risk

    # 3. 绘制 KM 曲线
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    
    if sum(high_risk_mask) > 0:
        kmf_high.fit(time_true[high_risk_mask], event_observed=event_true[high_risk_mask], label='High Risk')
        kmf_high.plot_survival_function(color='red', linewidth=2)
    
    if sum(low_risk_mask) > 0:
        kmf_low.fit(time_true[low_risk_mask], event_observed=event_true[low_risk_mask], label='Low Risk')
        kmf_low.plot_survival_function(color='blue', linewidth=2)

    plt.title(f'{title_prefix} Kaplan-Meier Survival Curve', fontsize=20, fontname='Times New Roman')
    plt.xlabel('Time (Months)', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.ylabel('Survival Probability', fontdict=ClinicalConfig.FONT_CONFIG)
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------
def main():
    config = ClinicalConfig()
    
    # 1. 数据加载与预处理
    print("Loading data for clinical evaluation...")
    data = pd.read_csv(config.DATA_PATH)
    
    # 提取时间、事件与特征（兼容不同输入文件的时间列命名）
    time_col = config.TARGET_TIME if config.TARGET_TIME in data.columns else None
    if time_col is None:
        for candidate in config.TIME_CANDIDATES:
            if candidate in data.columns:
                time_col = candidate
                break
    if time_col is None:
        raise KeyError(
            f"No survival time column found. Tried: {[config.TARGET_TIME] + config.TIME_CANDIDATES}. "
            f"Available columns: {list(data.columns)}"
        )

    time_true = data[time_col].values
    event_true = data[config.TARGET_EVENT].astype(float).astype(int).values
    X = data.drop([time_col, config.TARGET_EVENT], axis=1, errors='ignore')
    
    # 分类变量处理
    actual_cat_features = [col for col in config.CAT_FEATURES if col in X.columns]
    X[actual_cat_features] = X[actual_cat_features].astype(str)
    num_features = [col for col in X.columns if col not in actual_cat_features]
    X[num_features] = X[num_features].astype(float)
    cat_features_indices = [X.columns.get_loc(col) for col in actual_cat_features]

    # 划分训练/测试集
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
        X, event_true, time_true, test_size=0.2, random_state=42
    )

    # 2. 训练 Full Model 并获取概率
    print("\nTraining CatBoost (Full Model)...")
    full_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=cat_features_indices,
        **config.CATBOOST_PARAMS
    )
    full_model.fit(X_train, y_train)
    
    # 训练集和测试集的风险评分 (Log-odds 也可以，这里直接用预测概率作为风险分数)
    prob_train_full = full_model.predict_proba(X_train)[:, 1]
    prob_test_full = full_model.predict_proba(X_test)[:, 1]

    # 3. 利用 SHAP 提取 Top-15 训练 Lite Model
    print("Extracting SHAP values to build Lite Model...")
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X_train)
    shap_values_for_importances = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_values_for_importances).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': mean_abs_shap})
    top_15_features = feature_importance_df.sort_values(by='importance', ascending=False)['feature'].head(config.LITE_FEATURES_COUNT).tolist()
    
    X_train_lite = X_train[top_15_features]
    X_test_lite = X_test[top_15_features]
    lite_cat_features_indices = [i for i, col in enumerate(top_15_features) if col in actual_cat_features]

    print("Training CatBoost (Lite Model)...")
    lite_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=lite_cat_features_indices,
        **config.CATBOOST_PARAMS
    )
    lite_model.fit(X_train_lite, y_train)
    prob_test_lite = lite_model.predict_proba(X_test_lite)[:, 1]

    # -------------------------------------------------------------------------
    # 临床评估执行区
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(" Executing Clinical Evaluations (Test Set)")
    print("="*50)

    # 评估 A: 决策曲线 (DCA)
    print("1. Generating Decision Curve Analysis (DCA)...")
    plot_dca_curve(y_test, prob_test_full, prob_test_lite, title="Decision Curve Analysis (Test Set)")

    # 评估 B: 校准曲线 (Calibration)
    print("2. Generating Calibration Curves...")
    plot_calibration_curve(y_test, prob_test_full, prob_test_lite, title="Calibration Curve (Test Set)")

    # 评估 C: 生存分析 (KM & C-index)
    print("3. Performing Survival Analysis (Kaplan-Meier & C-index)...")
    # 获取训练集风险概率的中位数作为高低风险划分阈值 (论文标准做法)
    median_risk_threshold = np.median(prob_train_full) 
    
    perform_survival_analysis(
        time_test, y_test, prob_test_full, median_risk_threshold, title_prefix="Full Model"
    )
    perform_survival_analysis(
        time_test, y_test, prob_test_lite, median_risk_threshold, title_prefix="Lite Model"
    )

    # ------------------ 新增：导出真实的 DCA 数据供 Streamlit 使用 ------------------
    print("4. Exporting real DCA metrics for Streamlit Analyzer...")
    thresh_group = np.arange(*ClinicalConfig.DCA_THRESH_RANGE)
    nb_full = calculate_net_benefit(thresh_group, prob_test_full, y_test)
    nb_lite = calculate_net_benefit(thresh_group, prob_test_lite, y_test)
    nb_all = calculate_net_benefit_all(thresh_group, y_test)
    
    # 将数据保存为 CSV
    dca_df = pd.DataFrame({
        'threshold': thresh_group,
        'nb_full': nb_full,
        'nb_lite_15': nb_lite, # 真实的 15 特征 Lite 模型净获益
        'nb_all': nb_all
    })
    
    # 确保输出目录存在 (可以存放在我们之前建的 cipec_export 文件夹里)
    os.makedirs("cipec_export", exist_ok=True)
    dca_csv_path = os.path.join("cipec_export", "dca_real_results.csv")
    dca_df.to_csv(dca_csv_path, index=False)
    print(f"✅ Real DCA data exported successfully to: {dca_csv_path}")

if __name__ == '__main__':
    main()