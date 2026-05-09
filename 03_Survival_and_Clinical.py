import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings('ignore')

class ClinicalConfig:
    TRAIN_DATA_PATH = r"data/result/labeled_train.csv"
    TEST_DATA_PATH = r"data/result/test_validation.csv"
    TARGET_EVENT = "OS"
    TIME_CANDIDATES = ["OS_m", "PFS_m"]
    LITE_FEATURES_COUNT = 15
    TARGET_CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose', 'ECOG', 'T', 'Chemotherapy']
    CATBOOST_PARAMS = {'depth': 8, 'iterations': 400, 'learning_rate': 0.1, 'early_stopping_rounds': 50, 'eval_metric': 'Logloss', 'random_seed': 42, 'verbose': 0}
    DCA_THRESH_RANGE = (0, 1, 0.01)

class CustomLoglossObjective(object):
    def __init__(self, penalty=2, reward_factor=0.9):
        self.penalty = penalty; self.reward_factor = reward_factor
    def calc_ders_range(self, approxes, targets, weights=None):
        exponents = [np.exp(a) for a in approxes]
        result = []
        for idx in range(len(targets)):
            p = exponents[idx] / (1 + exponents[idx])
            penalty = 1.0 if not ((targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8)) else self.penalty
            der1_log = (1 - p) * penalty if targets[idx] > 0.0 else -p * penalty
            der2_log = -p * (1 - p)
            q = 0.6
            der1_quantile = q * (p - p**2) if targets[idx] - p >= 0 else (q - 1) * (p - p**2)
            der1_combined = (der1_quantile + der1_log) / 2
            der2_combined = der2_log / 2
            if weights is not None:
                der1_combined *= weights[idx]; der2_combined *= weights[idx]
            result.append((der1_combined, der2_combined))
        return result

# ================= 核心修复：Platt Scaling 概率校准 =================
def calibrate_probabilities(y_train, prob_train_raw, prob_test_raw):
    """
    用逻辑回归将 CustomLoss 畸变的输出强制拉伸并对齐到真实的临床概率空间 [0, 1]
    彻底解决 DCA 曲线“全不治疗 (Treat None)” 问题，并替代原本的 orient_probabilities
    """
    calibrator = LogisticRegression()
    calibrator.fit(prob_train_raw.reshape(-1, 1), y_train)
    calibrated_train = calibrator.predict_proba(prob_train_raw.reshape(-1, 1))[:, 1]
    calibrated_test = calibrator.predict_proba(prob_test_raw.reshape(-1, 1))[:, 1]
    return calibrated_train, calibrated_test
# =================================================================

def calculate_net_benefit(thresh_group, y_pred_score, y_label):
    net_benefit = []
    n = len(y_label)
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label, labels=[0, 1]).ravel()
        nb = 0.0 if (1 - thresh) == 0 else (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit.append(nb)
    return np.array(net_benefit)

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit = []
    tp_total = np.sum(y_label == 1); total = len(y_label)
    for thresh in thresh_group:
        nb = 0.0 if (1 - thresh) == 0 else (tp_total / total) - ((total - tp_total) / total) * (thresh / (1 - thresh))
        net_benefit.append(nb)
    return np.array(net_benefit)

# ================= 恢复：Cox 特征优化循环 =================
def optimize_cox_feature_combination(train_df, test_df, feature_cols, time_col, event_col):
    print("\n--- Running Cox Feature Optimization (from test survival.py) ---")
    essential_cols = [time_col, event_col]
    total_features = len(feature_cols)
    c_index_values = []
    
    for n_features in range(1, total_features + 1):
        np.random.seed(42)
        selected_features = np.random.choice(feature_cols, n_features, replace=False).tolist()
        current_cols = essential_cols + selected_features
        
        cph = CoxPHFitter()
        # 加入惩罚项防止 Cox 在少样本下共线性报错
        cph.fit(train_df[current_cols], duration_col=time_col, event_col=event_col, penalizer=0.1)
        
        train_predictions = cph.predict_expectation(train_df[current_cols])
        train_c_index = concordance_index(train_df[time_col], train_predictions, train_df[event_col])
        c_index_values.append(train_c_index)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_features + 1), c_index_values, marker='o', color='blue', linestyle='-')
    plt.title('Concordance Index vs Number of Features', fontsize=20)
    plt.xlabel('Number of Features', fontsize=16)
    plt.ylabel('Train Concordance Index', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# ============================================================

def main():
    config = ClinicalConfig()
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    time_col = next((c for c in config.TIME_CANDIDATES if c in train_df.columns), None)
    if time_col is None:
        raise KeyError(f"Missing time column. Make sure {config.TIME_CANDIDATES} are in 01_AL_shap.py output.")
    
    drop_cols = [config.TARGET_EVENT, 'OS_m', 'PFS_m']
    X_train = train_df.drop([col for col in drop_cols if col in train_df.columns], axis=1)
    y_train = train_df[config.TARGET_EVENT].astype(int).values
    time_train = train_df[time_col].values

    X_test = test_df.drop([col for col in drop_cols if col in test_df.columns], axis=1)
    y_test = test_df[config.TARGET_EVENT].astype(int).values
    time_test = test_df[time_col].values

    cat_cols = [col for col in X_train.columns if col in config.TARGET_CAT_FEATURES]
    X_train[cat_cols] = X_train[cat_cols].astype(float).astype(int).astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(float).astype(int).astype(str)
    
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    X_train[num_cols] = X_train[num_cols].astype(float)
    X_test[num_cols] = X_test[num_cols].astype(float)

    # 1. Full Model
    train_pool_full = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool_full = Pool(X_test, y_test, cat_features=cat_cols)

    full_model = CatBoostClassifier(loss_function=CustomLoglossObjective(), cat_features=cat_cols, **config.CATBOOST_PARAMS)
    full_model.fit(train_pool_full, eval_set=test_pool_full)
    
    prob_train_full_raw = full_model.predict_proba(train_pool_full)[:, 1]
    prob_test_full_raw = full_model.predict_proba(test_pool_full)[:, 1]
    
    # 核心：将畸变概率校准拉伸回 [0, 1] 区间
    prob_train_full, prob_test_full = calibrate_probabilities(y_train, prob_train_full_raw, prob_test_full_raw)

    # 2. Lite Model
    import shap
    shap_vals = shap.TreeExplainer(full_model).shap_values(train_pool_full)
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    
    top_15 = pd.DataFrame({'feature': X_train.columns, 'importance': np.abs(shap_vals).mean(axis=0)}).sort_values(by='importance', ascending=False)['feature'].head(config.LITE_FEATURES_COUNT).tolist()
    
    X_train_lite = X_train[top_15]; X_test_lite = X_test[top_15]
    cat_cols_lite = [col for col in top_15 if col in cat_cols]

    train_pool_lite = Pool(X_train_lite, y_train, cat_features=cat_cols_lite)
    test_pool_lite = Pool(X_test_lite, y_test, cat_features=cat_cols_lite)

    lite_model = CatBoostClassifier(loss_function=CustomLoglossObjective(), cat_features=cat_cols_lite, **config.CATBOOST_PARAMS)
    lite_model.fit(train_pool_lite, eval_set=test_pool_lite)
    
    prob_train_lite_raw = lite_model.predict_proba(train_pool_lite)[:, 1]
    prob_test_lite_raw = lite_model.predict_proba(test_pool_lite)[:, 1]
    
    # 校准 Lite 模型
    prob_train_lite, prob_test_lite = calibrate_probabilities(y_train, prob_train_lite_raw, prob_test_lite_raw)

    print(f"\n[Calibrated Test AUC] Full: {roc_auc_score(y_test, prob_test_full):.4f} | Lite: {roc_auc_score(y_test, prob_test_lite):.4f}")

    # 3. 输出完美展开的 DCA 曲线
    thresh_group = np.arange(*config.DCA_THRESH_RANGE)
    nb_full = calculate_net_benefit(thresh_group, prob_test_full, y_test)
    nb_lite = calculate_net_benefit(thresh_group, prob_test_lite, y_test)
    nb_all = calculate_net_benefit_all(thresh_group, y_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresh_group, nb_full, 'r-', linewidth=2, label='CatBoost (Full Model)')
    plt.plot(thresh_group, nb_lite, 'orange', linewidth=2, linestyle='--', label='CatBoost (Lite Model)')
    plt.plot(thresh_group, nb_all, color='slategrey', label='Treat All')
    plt.plot((0, 1), (0, 0), 'k:', label='Treat None')
    plt.xlim(0, 1)
    # 限制 y 轴范围以免被 Treat All 的负值挤压
    plt.ylim(-0.1, max(nb_full.max(), nb_lite.max()) + 0.1)
    plt.title("Calibrated Decision Curve Analysis (Test Set)", fontsize=20)
    plt.xlabel('Threshold Probability', fontsize=16)
    plt.ylabel('Net Benefit', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.show()

    # 4. 恢复 Cox 优化曲线
    optimize_cox_feature_combination(train_df, test_df, X_train.columns.tolist(), time_col, config.TARGET_EVENT)

if __name__ == '__main__':
    main()
