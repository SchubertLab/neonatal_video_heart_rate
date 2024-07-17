import numpy as np
from scipy.stats.stats import pearsonr


def rmse(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):
    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    predictions_flat = np.array(predicted_hr_list).flatten()
    targets_flat = np.array(target_hr_list).flatten()

    HR_MAE = mae(
        predictions_flat,
        targets_flat
    )
    HR_RMSE = rmse(
        predictions_flat,
        targets_flat
    )

    PEARSON_R, p_value = pearsonr(predictions_flat, targets_flat)

    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "PEARSON_R": PEARSON_R}
