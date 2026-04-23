import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class ExportConfig:
    """
    Configuration for final model training and exporting for CIPEC software.
    """
    # 输入：由 01_Data_and_SHAP.py 产出的最终干净数据
    DATA_PATH = r"data/result/remained.csv" 
    
    # 输出：模型和配置文件保存的文件夹
    OUTPUT_DIR = r"cipec_export"
    
    # 标签列 (根据实际训练目标修改为 'OS' 或 'PFS')
    TARGET_LABEL = "OS"
    
    LITE_FEATURES_COUNT = 15  # Lite 模型保留的特征数
    
    # 分类特征列表 (必须与之前的处理保持一致)
    CAT_FEATURES = ['Age', 'Location', 'N', 'TNM', 'ECOG', 'T', 'Chemotherapy']
    

    CATBOOST_BEST_PARAMS = {
        'depth': 8,
        'iterations': 400,
        'learning_rate': 0.03,
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': 100  # 打印训练进度，确保模型正在收敛
    }


def standardize_selected_features(X: pd.DataFrame, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['treatment', 'TL', 'original']
    scaler = StandardScaler()
    columns_to_standardize = [col for col in X.columns if any(kw in col for kw in keywords)]
    X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
    return X


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


def main():
    config = ExportConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Starting CIPEC Model Export Pipeline")
    print("="*60)

    # 1. 加载所有可用数据 (用于最终生产模型，不再切分测试集，最大化利用数据)
    print(f"Loading data from {config.DATA_PATH}...")
    data = pd.read_csv(config.DATA_PATH)
    
    # 提取标签并清理不需要的列
    y = data[config.TARGET_LABEL].astype(float).astype(int).values
    drop_cols = [config.TARGET_LABEL, "OS_m", "PFS_m"]  # 排除所有标签和时间列
    X = data.drop([col for col in drop_cols if col in data.columns], axis=1)
    
    # 与 Active Learning_1.0.py 保持一致：标准化 + 前13列作为分类特征
    X = standardize_selected_features(X)
    cat_features_count = min(13, X.shape[1])
    full_cat_indices = list(range(cat_features_count))
    cat_cols = X.columns[full_cat_indices].tolist()
    X[cat_cols] = X[cat_cols].astype(str)
    num_cols = X.columns[cat_features_count:]
    X[num_cols] = X[num_cols].astype(float)


    #  Train & Export Full Model
    print(f"\n[1/4] Training FULL Model on all {len(X)} samples with {X.shape[1]} features...")
    full_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=full_cat_indices,
        **config.CATBOOST_BEST_PARAMS
    )
    full_model.fit(X, y)
    
    full_model_path = os.path.join(config.OUTPUT_DIR, "cipec_full_model.cbm")
    full_model.save_model(full_model_path)
    print(f"✅ Full model saved to: {full_model_path}")


    # SHAP Extraction for Lite Model
    print(f"\n[2/4] Running SHAP to extract Top-{config.LITE_FEATURES_COUNT} features for Lite Model...")
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X)
    shap_values_for_importances = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_values_for_importances).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': mean_abs_shap})
    top_15_features = feature_importance_df.sort_values(by='importance', ascending=False)['feature'].head(config.LITE_FEATURES_COUNT).tolist()
    
    print(f"Top 15 Features: {top_15_features}")


    # Train & Export Lite Model
    print(f"\n[3/4] Training LITE Model on top {config.LITE_FEATURES_COUNT} features...")
    X_lite = X[top_15_features]
    lite_cat_indices = [i for i, col in enumerate(top_15_features) if col in cat_cols]
    
    lite_model = CatBoostClassifier(
        loss_function=CustomLoglossObjective(),
        cat_features=lite_cat_indices,
        **config.CATBOOST_BEST_PARAMS
    )
    lite_model.fit(X_lite, y)
    
    lite_model_path = os.path.join(config.OUTPUT_DIR, "cipec_lite_model.cbm")
    lite_model.save_model(lite_model_path)
    print(f"✅ Lite model saved to: {lite_model_path}")


    print("\n[4/4] Exporting GUI Configuration JSON...")
    
    # 获取特征顺序，供 GUI 开发者构建输入表单时参考
    gui_config = {
        "full_model": {
            "model_file": "cipec_full_model.cbm",
            "feature_count": len(X.columns),
            "expected_features": list(X.columns),
            "categorical_features": cat_cols
        },
        "lite_model": {
            "model_file": "cipec_lite_model.cbm",
            "feature_count": len(top_15_features),
            "expected_features": top_15_features,
            "categorical_features": [col for col in top_15_features if col in cat_cols]
        }
    }
    
    config_path = os.path.join(config.OUTPUT_DIR, "cipec_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(gui_config, f, indent=4, ensure_ascii=False)
        
    print(f" GUI configuration saved to: {config_path}")
    print("\n Export pipeline completed successfully! You can now copy the 'cipec_export' folder to your software.")

if __name__ == "__main__":
    main()
