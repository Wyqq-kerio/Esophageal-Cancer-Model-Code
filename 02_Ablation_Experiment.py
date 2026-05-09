import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap
import warnings

warnings.filterwarnings('ignore')

class ModelConfig:
    TRAIN_DATA_PATH = r"data/result/labeled_train.csv" 
    TEST_DATA_PATH = r"data/result/test_validation.csv"
    
    CATBOOST_PARAMS = {
        'depth': 8, 'iterations': 400, 'learning_rate': 0.1,
        'early_stopping_rounds': 50, 'eval_metric': 'Logloss',
        'random_seed': 42, 'verbose': 0
    }
    
    TARGET_CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'PTV_Dose', 'GTV_Dose', 'ECOG', 'T', 'Chemotherapy']
    PLOT_FIGSIZE = (10, 8)
    LITE_FEATURES_COUNT = 15

class CustomLoglossObjective(object):
    def __init__(self, penalty=2, reward_factor=0.9):
        self.penalty = penalty; self.reward_factor = reward_factor
    def calc_ders_range(self, approxes, targets, weights=None):
        exponents = [np.exp(a) for a in approxes]
        result = []
        for idx in range(len(targets)):
            p = exponents[idx] / (1 + exponents[idx])
            penalty = 1.0
            if (targets[idx] == 1 and p < 0.2) or (targets[idx] == 0 and p > 0.8):
                penalty *= self.penalty
            der1_log = (1 - p) * self.penalty if targets[idx] > 0.0 else -p * self.penalty
            der2_log = -p * (1 - p)
            q = 0.6
            der1_quantile = q * (p - p**2) if targets[idx] - p >= 0 else (q - 1) * (p - p**2)
            der1_combined = (der1_quantile + der1_log) / 2
            der2_combined = der2_log / 2
            if weights is not None:
                der1_combined *= weights[idx]; der2_combined *= weights[idx]
            result.append((der1_combined, der2_combined))
        return result

def orient_probabilities(y_true, probs):
    raw_auc = roc_auc_score(y_true, probs)
    if raw_auc < 0.5: return 1.0 - probs, raw_auc, 1.0 - raw_auc, True
    return probs, raw_auc, raw_auc, False

def calculate_oriented_roc_auc(y_true, probs):
    oriented_probs, raw_auc, auc_score, flipped = orient_probabilities(y_true, probs)
    fpr, tpr, _ = roc_curve(y_true, oriented_probs)
    return fpr, tpr, auc_score, raw_auc, flipped

def main():
    config = ModelConfig()
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    drop_cols = ['OS', 'OS_m', 'PFS_m']
    X_train = train_df.drop([c for c in drop_cols if c in train_df.columns], axis=1)
    # 使用干净的标签恢复原版效能
    y_train = train_df['OS'].astype(int).values
    X_test = test_df.drop([c for c in drop_cols if c in test_df.columns], axis=1)
    y_test = test_df['OS'].astype(int).values

    cat_cols = [col for col in X_train.columns if col in config.TARGET_CAT_FEATURES]
    X_train[cat_cols] = X_train[cat_cols].astype(float).astype(int).astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(float).astype(int).astype(str)
    
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    X_train[num_cols] = X_train[num_cols].astype(float)
    X_test[num_cols] = X_test[num_cols].astype(float)

    print(f"\nTraining CatBoost (Full Model)...")
    full_model = CatBoostClassifier(loss_function=CustomLoglossObjective(), cat_features=cat_cols, **config.CATBOOST_PARAMS)
    
    train_pool_full = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool_full = Pool(X_test, y_test, cat_features=cat_cols)
    full_model.fit(train_pool_full, eval_set=test_pool_full)
    
    full_probs_raw = full_model.predict_proba(Pool(X_test, cat_features=cat_cols))[:, 1]
    
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(train_pool_full)
    shap_values_for_importances = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_values_for_importances).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)
    top_15_features = feature_importance_df['feature'].head(config.LITE_FEATURES_COUNT).tolist()
    
    X_train_lite = X_train[top_15_features]; X_test_lite = X_test[top_15_features]
    cat_cols_lite = [col for col in top_15_features if col in cat_cols]
    
    print("\nTraining CatBoost (Lite Model)...")
    lite_model = CatBoostClassifier(loss_function=CustomLoglossObjective(), cat_features=cat_cols_lite, **config.CATBOOST_PARAMS)
    lite_model.fit(Pool(X_train_lite, y_train, cat_features=cat_cols_lite), 
                   eval_set=Pool(X_test_lite, y_test, cat_features=cat_cols_lite))
    lite_probs_raw = lite_model.predict_proba(Pool(X_test_lite, cat_features=cat_cols_lite))[:, 1]

    print("\nTraining Baseline models...")
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

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
    full_fpr, full_tpr, full_auc, _, full_flipped = calculate_oriented_roc_auc(y_test, full_probs_raw)
    lite_fpr, lite_tpr, lite_auc, _, lite_flipped = calculate_oriented_roc_auc(y_test, lite_probs_raw)
    roc_results['weighted-CatBoost (Full)'] = (full_fpr, full_tpr, full_auc)
    roc_results['weighted-CatBoost (Lite)'] = (lite_fpr, lite_tpr, lite_auc)
    
    for name, model in baselines.items():
        model.fit(X_train_scaled, y_train)
        probs_raw = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, auc_score, _, _ = calculate_oriented_roc_auc(y_test, probs_raw)
        roc_results[name] = (fpr, tpr, auc_score)

    plt.figure(figsize=config.PLOT_FIGSIZE)
    colors = ['red', 'darkorange', 'blue', 'green', 'purple', 'brown', 'cyan']
    for (label, (fpr, tpr, auc_score)), color in zip(roc_results.items(), colors):
        linewidth = 3 if 'CatBoost' in label else 1.5
        linestyle = '-' if 'Full' in label else ('--' if 'Lite' in label else ':')
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.3f})', color=color, linewidth=linewidth, linestyle=linestyle)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Ablation Study (Test Set)", fontsize=22)
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()
