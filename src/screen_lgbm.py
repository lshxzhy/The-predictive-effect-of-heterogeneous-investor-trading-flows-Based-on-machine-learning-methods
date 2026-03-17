import argparse

import pandas as pd
from lightgbm import LGBMClassifier

from config import get_horizon_context
from training_utils import evaluate_model, save_model_bundle


MODEL_ID = "screen_lgbm"
RANDOM_STATE = 42
N_JOBS = -1
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "auc", "ks"]
SEARCH_LOG_HELPER_COLS = [
    "meet_valid_auc_le_train_auc",
    "meet_coarse_valid_auc_floor",
    "meet_fine_selection_requirements",
    "coarse_best_valid_auc",
]
SEARCH_PARAM_NAMES = [
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "min_child_weight",
    "min_split_gain",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "path_smooth",
    "extra_trees",
    "scale_pos_weight",
]
# 调参顺序刻意采用“先学习率和树数，再树结构，再采样与正则”的单参数锁定路径，
# 每一步都在上一轮已锁定参数的基础上继续搜索。
SEARCH_PARAM_ORDER = [
    "learning_rate",
    "n_estimators",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "min_child_weight",
    "min_split_gain",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "path_smooth",
    "extra_trees",
    "scale_pos_weight",
]
# 每个超参数都集中记录“含义 + 当前搜索范围 + 一般变化趋势”，
# 既服务代码阅读，也会同步写入 best_params 报表，方便把结果和理论直觉对照起来。
SEARCH_PARAM_DOCS = {
    # 树的总轮数；粗搜 100~900；增大通常拟合更强、训练更慢，也更容易放大过拟合。
    "n_estimators": {
        "meaning": "Boosting 迭代轮数，也就是累积树的数量。",
        "search_range": "粗搜 100~900；细搜围绕粗搜最优按约 8% 步长扩展，最低 50。",
        "trend": "增大通常提升拟合能力并拉长训练时间；过大更容易过拟合，过小则可能欠拟合。",
    },
    # 学习率；粗搜 0.005~0.08；变大收敛更快但更激进，变小更平滑但通常需要更多树。
    "learning_rate": {
        "meaning": "每轮树对整体模型的步长缩放系数。",
        "search_range": "粗搜 0.005~0.08；细搜围绕粗搜最优做约 8% 微调，上限 0.2。",
        "trend": "增大时收敛更快但更冒进；减小时更新更平滑，常需要配合更多 n_estimators。",
    },
    # 叶子数；粗搜 4~64；增大提升非线性表达，但树更复杂、更容易拟合噪声。
    "num_leaves": {
        "meaning": "单棵树允许生成的最大叶子数。",
        "search_range": "粗搜 4~64；细搜围绕粗搜最优按约 12% 步长扩展。",
        "trend": "增大能增强非线性和交互刻画；过大时复杂度上升、过拟合风险更高。",
    },
    # 最大深度；粗搜 2~10 或 -1 不限深；增大允许更深分裂，减小则更保守。
    "max_depth": {
        "meaning": "单棵树的最大深度，-1 表示不限制。",
        "search_range": "粗搜 {2,3,4,5,6,8,10,-1}；细搜在最优值附近按 1 层微调。",
        "trend": "增大允许树更深、拟合更强但更不稳；减小会强制树结构更浅、更保守。",
    },
    # 叶子最小样本数；粗搜 10~200；增大抑制碎叶子，减小更容易贴合局部噪声。
    "min_child_samples": {
        "meaning": "每个叶子节点至少需要的样本数。",
        "search_range": "粗搜 10~200；细搜围绕粗搜最优按约 8% 步长扩展，最低 5。",
        "trend": "增大时分裂更谨慎、过拟合更低；减小时叶子更细、更容易学习局部波动。",
    },
    # 叶子最小 Hessian；粗搜 0.001~10；增大让继续分裂更难，减小会放松约束。
    "min_child_weight": {
        "meaning": "叶子节点最小二阶梯度和约束。",
        "search_range": "粗搜 0.001~10；细搜围绕粗搜最优按 0.4~1.6 倍缩放。",
        "trend": "增大后小样本或低权重叶子更难出现；减小时模型更容易继续分裂。",
    },
    # 最小分裂增益；粗搜 0~1，细搜上限到 3；增大只保留更“值钱”的分裂。
    "min_split_gain": {
        "meaning": "一次分裂至少要带来的损失下降阈值。",
        "search_range": "粗搜 0.0~1.0；细搜围绕粗搜最优微调，允许到 3.0。",
        "trend": "增大时树更稀疏、分裂更保守；减小时更容易继续切分。",
    },
    # 行采样比例；粗搜 0.4~1.0；减小增加随机性并抑制过拟合，过低则可能损失信号。
    "subsample": {
        "meaning": "每轮训练用于建树的样本抽样比例。",
        "search_range": "粗搜 0.4~1.0；细搜围绕粗搜最优按 0.03 微调。",
        "trend": "减小时随机性更强、往往更稳；接近 1 时训练更稳定，但更容易放大共性噪声。",
    },
    # 行采样频率；粗搜 0~7；0 表示不做 bagging，增大表示更频繁启用抽样。
    "subsample_freq": {
        "meaning": "执行 subsample 的频率。",
        "search_range": "粗搜 {0,1,2,3,5,7}；细搜围绕粗搜最优按 1 递增或递减。",
        "trend": "0 表示关闭 bagging；数值越大，行采样越频繁，随机性通常更强。",
    },
    # 列采样比例；粗搜 0.4~1.0；减小可缓解共线和过拟合，但过小会丢信息。
    "colsample_bytree": {
        "meaning": "每棵树建树时可见的特征抽样比例。",
        "search_range": "粗搜 0.4~1.0；细搜围绕粗搜最优按 0.03 微调。",
        "trend": "减小时有助于降低特征共线噪声；过小则会削弱有效信号。",
    },
    # L1 正则；粗搜 0~5；增大鼓励权重稀疏化，通常更保守。
    "reg_alpha": {
        "meaning": "L1 正则项系数。",
        "search_range": "粗搜 0.0~5.0；细搜在最优值附近按 0.1 或 0.25 微调。",
        "trend": "增大通常让模型更稀疏、更保守；过大则可能欠拟合。",
    },
    # L2 正则；粗搜 0~10；增大可平滑叶子权重，过大也会压缩有效信号。
    "reg_lambda": {
        "meaning": "L2 正则项系数。",
        "search_range": "粗搜 0.0~10.0；细搜在最优值附近按 0.1 或 0.25 微调。",
        "trend": "增大可平滑参数波动、压制过拟合；过大时会削弱模型区分能力。",
    },
    # 最大分箱数；粗搜 31~511；增大能保留更细离散化信息，但训练更慢。
    "max_bin": {
        "meaning": "连续特征离散化时允许的最大分箱数。",
        "search_range": "粗搜 {31,63,127,255,511}；细搜围绕粗搜最优按 32 微调。",
        "trend": "增大时特征切分更细、表达更充分；过大可能增加噪声拟合和训练成本。",
    },
    # 路径平滑；粗搜 0~5，细搜到 10；增大让叶子输出更平滑、更保守。
    "path_smooth": {
        "meaning": "对叶子输出做路径平滑的强度。",
        "search_range": "粗搜 0.0~5.0；细搜围绕粗搜最优按 25% 左右步长扩展，上限 10。",
        "trend": "增大时叶子估计更平滑、更抗噪；减小时更贴合局部样本。",
    },
    # 是否启用随机阈值；粗搜只有 False/True 两档；True 往往更稳，但单棵树更粗糙。
    "extra_trees": {
        "meaning": "是否使用极端随机树式的随机分裂阈值。",
        "search_range": "粗搜与细搜都只在 {False, True} 两个取值间切换。",
        "trend": "True 会增强随机性、常有助于稳健性；False 更确定、更容易追随训练样本细节。",
    },
    # 正类权重；粗搜 0.8~1.2，细搜可到 0.5~1.5；增大更偏向召回正类。
    "scale_pos_weight": {
        "meaning": "正类样本在损失中的权重缩放。",
        "search_range": "粗搜 0.8~1.2；细搜围绕粗搜最优按 0.05 微调，范围 0.5~1.5。",
        "trend": "增大通常提高正类召回倾向；减小则更偏向负类精度和平衡。",
    },
}
# 这是两轮搜索的公共起点，表示在没有搜索证据前的默认筛选模型复杂度。
INITIAL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 80,
    "min_child_weight": 1.0,
    "min_split_gain": 0.1,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "max_bin": 255,
    "path_smooth": 0.5,
    "extra_trees": False,
    "scale_pos_weight": 1.0,
}
# 第一轮粗搜只修改一个参数，其余参数保持当前锁定值；
# 因而这个字典既是粗搜候选集合，也是当前允许探索的理论范围。
COARSE_CANDIDATES = {
    "n_estimators": [100, 200, 300, 500, 700, 900],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [4, 8, 12, 16, 24, 32, 48, 64],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, -1],
    "min_child_samples": [10, 20, 40, 60, 80, 100, 150, 200],
    "min_child_weight": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    "min_split_gain": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample_freq": [0, 1, 2, 3, 5, 7],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
    "reg_lambda": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
    "max_bin": [31, 63, 127, 255, 511],
    "path_smooth": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    "extra_trees": [False, True],
    "scale_pos_weight": [0.8, 0.9, 1.0, 1.1, 1.2],
}


# 清理旧版随机森林筛选残留文件，避免当前目录里同时存在多套历史输出。
def cleanup_legacy_output_files(ctx) -> None:
    """清理已经废弃的旧筛选输出文件。"""
    legacy_files = [
        ctx.paths.screening_output_file("screening_rf_importance.csv"),
        ctx.paths.screening_output_file("screening_rf_metrics.csv"),
        ctx.paths.screening_output_file("screening_rf_best_params.csv"),
        ctx.paths.screening_output_file("screening_rf_search_results.csv"),
    ]
    for file_path in legacy_files:
        if file_path.exists():
            file_path.unlink()


# 统一筛选只读取 screening_long_panel，并且只取 30 个候选变量与固定 1d 标签。
def load_screening_long_panel(horizon: int) -> tuple[pd.DataFrame, list[str], str]:
    """读取统一筛选长面板并拆分特征列。"""
    ctx = get_horizon_context(horizon=horizon)
    screening_horizon = ctx.config.selected_feature_source_horizon()
    if horizon != screening_horizon:
        raise ValueError(
            f"统一变量筛选固定使用 {screening_horizon}d，请显式传入 --horizon {screening_horizon}"
        )

    df = pd.read_csv(ctx.paths.screening_long_panel_file(), parse_dates=["Date"])
    feature_cols = ctx.config.all_screening_supplementary_cols()
    target_col = ctx.config.target_label_col(screening_horizon)
    return (
        df.loc[:, [*ctx.config.screening_id_cols, "split", target_col, *feature_cols]].copy(),
        feature_cols,
        target_col,
    )


# 在 pooled 长面板上直接按 split 拆出 train/valid/test，
# 这里不再做任何再切分、再填充或再标准化。
def split_to_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """拆分筛选模型的训练集、验证集和测试集。"""
    train_df = df.loc[df["split"] == "train"].copy()
    valid_df = df.loc[df["split"] == "valid"].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("统一筛选长面板缺少 train、valid 或 test 样本。")

    missing_counts = df.loc[:, feature_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"统一筛选长面板仍存在特征缺失：{missing_counts.to_dict()}")

    return (
        train_df.loc[:, feature_cols].copy(),
        train_df[target_col].astype(int).copy(),
        valid_df.loc[:, feature_cols].copy(),
        valid_df[target_col].astype(int).copy(),
        test_df.loc[:, feature_cols].copy(),
        test_df[target_col].astype(int).copy(),
    )


# LightGBM 结构固定为二分类 GBDT，并统一使用 gain 重要性口径。
def build_model(params: dict) -> LGBMClassifier:
    """根据参数构造 LightGBM 筛选模型。"""
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        **params,
    )


# 单次候选参数只在 train 上拟合、在 valid 上评估，
# 返回的 train/valid 指标会被写入搜索日志供后续排序。
def fit_and_evaluate_on_holdout(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[LGBMClassifier, pd.Series, pd.Series]:
    """拟合单组参数并返回训练集与验证集评估。"""
    model = build_model(params)
    model.fit(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train, "train").iloc[0]
    valid_metrics = evaluate_model(model, X_valid, y_valid, "valid").iloc[0]
    return model, train_metrics, valid_metrics


# 把单次搜索结果整理成一行：
# 一边保留完整参数快照，一边保留 train/valid 指标和过拟合差距。
def build_search_row(
    stage: str,
    search_id: int,
    tuned_parameter: str,
    params: dict,
    train_metrics: pd.Series,
    valid_metrics: pd.Series,
) -> dict[str, object]:
    """整理单次搜索结果。"""
    row: dict[str, object] = {
        "stage": stage,
        "search_id": search_id,
        "tuned_parameter": tuned_parameter,
    }
    row.update({name: params[name] for name in SEARCH_PARAM_NAMES})

    for metric_col in METRIC_COLS:
        row[f"train_{metric_col}"] = train_metrics[metric_col]
        row[f"valid_{metric_col}"] = valid_metrics[metric_col]

    train_auc = train_metrics["auc"]
    valid_auc = valid_metrics["auc"]
    row["meet_valid_auc_le_train_auc"] = (
        pd.notna(train_auc)
        and pd.notna(valid_auc)
        and valid_auc <= train_auc
    )
    row["auc_gap"] = abs(train_auc - valid_auc) if pd.notna(train_auc) and pd.notna(valid_auc) else float("inf")
    return row


# 第一轮先筛掉 valid_auc 高于 train_auc 的候选，再在合格候选里按 valid_auc 选优。
def rank_coarse_step(step_df: pd.DataFrame) -> pd.DataFrame:
    """按第一轮规则排序单参数粗搜结果。"""
    if not step_df["meet_valid_auc_le_train_auc"].any():
        raise ValueError(
            "当前粗搜参数步不存在满足 valid_auc <= train_auc 的候选，"
            "请检查搜索网格或上游数据。"
        )
    ranked_df = step_df.sort_values(
        ["meet_valid_auc_le_train_auc", "valid_auc", "auc_gap", "train_auc", "search_id"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


# 第二轮只接受同时满足“valid_auc 不低于第一轮最优”且“valid_auc 不高于 train_auc”的候选，
# 再在达标候选里按 train-valid gap 选更稳的一组。
def rank_fine_step(step_df: pd.DataFrame, coarse_best_valid_auc: float) -> pd.DataFrame:
    """按第二轮规则排序单参数细搜结果。"""
    ranked_df = step_df.copy()
    ranked_df["meet_coarse_valid_auc_floor"] = ranked_df["valid_auc"] >= coarse_best_valid_auc
    ranked_df["meet_fine_selection_requirements"] = (
        ranked_df["meet_coarse_valid_auc_floor"]
        & ranked_df["meet_valid_auc_le_train_auc"]
    )
    ranked_df = ranked_df.sort_values(
        [
            "meet_fine_selection_requirements",
            "auc_gap",
            "valid_auc",
            "train_auc",
            "search_id",
        ],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    ranked_df["selection_rank"] = ranked_df.index + 1
    return ranked_df


# 对整数型参数在当前中心值附近做对称细搜，并强制下界合法。
def build_int_candidates(center: int, step: int, min_value: int) -> list[int]:
    """围绕整数中心值构造细搜候选。"""
    candidates = [
        center - 3 * step,
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
        center + 3 * step,
    ]
    return sorted({max(min_value, int(candidate)) for candidate in candidates})


# 对浮点型参数在中心值附近做对称细搜，同时裁到合法区间并统一保留四位小数。
def build_float_candidates(
    center: float,
    step: float,
    min_value: float,
    max_value: float,
) -> list[float]:
    """围绕浮点中心值构造细搜候选。"""
    candidates = [
        center - 3 * step,
        center - 2 * step,
        center - step,
        center,
        center + step,
        center + 2 * step,
        center + 3 * step,
    ]
    rounded_candidates = []
    for candidate in candidates:
        clipped = min(max(candidate, min_value), max_value)
        rounded_candidates.append(round(float(clipped), 4))
    return sorted(set(rounded_candidates))


# max_depth 的特殊值 -1 代表不限深，因此要单独保留一套细搜候选。
def build_max_depth_candidates(center: int) -> list[int]:
    """构造 max_depth 细搜候选。"""
    if center == -1:
        return [-1, 8, 10, 12, 14]
    return build_int_candidates(center=center, step=1, min_value=2)


# min_child_weight 更适合按倍数而不是固定步长缩放，
# 这样能同时兼顾很小值和很大值附近的搜索精度。
def build_min_child_weight_candidates(center: float) -> list[float]:
    """构造 min_child_weight 细搜候选。"""
    factors = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    candidates = []
    for factor in factors:
        candidate = max(0.0001, round(float(center) * factor, 4))
        candidates.append(candidate)
    return sorted(set(candidates))


# reg_alpha 和 reg_lambda 共用同一套细搜规则：
# 小参数附近用更小步长，大参数附近放宽步长。
def build_regularization_candidates(center: float) -> list[float]:
    """构造正则项细搜候选。"""
    step = 0.1 if center <= 1.0 else 0.25
    return build_float_candidates(center=center, step=step, min_value=0.0, max_value=10.0)


# 细搜候选全部围绕粗搜最优值生成，不再回到全局大范围搜索。
def build_fine_candidates(parameter_name: str, center_value: object) -> list[object]:
    """围绕粗搜最优值生成超小步长细搜候选。"""
    if parameter_name == "n_estimators":
        step = max(20, int(round(float(center_value) * 0.08)))
        return build_int_candidates(int(center_value), step=step, min_value=50)
    if parameter_name == "learning_rate":
        step = max(0.0025, float(center_value) * 0.08)
        return build_float_candidates(float(center_value), step=step, min_value=0.0025, max_value=0.2)
    if parameter_name == "num_leaves":
        step = max(2, int(round(float(center_value) * 0.12)))
        return build_int_candidates(int(center_value), step=step, min_value=2)
    if parameter_name == "max_depth":
        return build_max_depth_candidates(int(center_value))
    if parameter_name == "min_child_samples":
        step = max(5, int(round(float(center_value) * 0.08)))
        return build_int_candidates(int(center_value), step=step, min_value=5)
    if parameter_name == "min_child_weight":
        return build_min_child_weight_candidates(float(center_value))
    if parameter_name == "min_split_gain":
        step = max(0.02, float(center_value) * 0.2)
        return build_float_candidates(float(center_value), step=step, min_value=0.0, max_value=3.0)
    if parameter_name == "subsample":
        return build_float_candidates(float(center_value), step=0.03, min_value=0.4, max_value=1.0)
    if parameter_name == "subsample_freq":
        return build_int_candidates(int(center_value), step=1, min_value=0)
    if parameter_name == "colsample_bytree":
        return build_float_candidates(float(center_value), step=0.03, min_value=0.4, max_value=1.0)
    if parameter_name in {"reg_alpha", "reg_lambda"}:
        return build_regularization_candidates(float(center_value))
    if parameter_name == "max_bin":
        return build_int_candidates(int(center_value), step=32, min_value=31)
    if parameter_name == "path_smooth":
        step = max(0.1, float(center_value) * 0.25)
        return build_float_candidates(float(center_value), step=step, min_value=0.0, max_value=10.0)
    if parameter_name == "extra_trees":
        return [False, True]
    if parameter_name == "scale_pos_weight":
        return build_float_candidates(float(center_value), step=0.05, min_value=0.5, max_value=1.5)
    raise KeyError(f"未配置细搜参数：{parameter_name}")


# 第一轮是顺序单参数粗搜：
# 每个参数遍历大范围候选，按 valid_auc 选出当前最优，再把该值锁进下一步搜索。
def run_coarse_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    """执行第一轮粗搜，并逐个参数锁定验证集 AUC 更高的值。"""
    current_params = INITIAL_PARAMS.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1

    for parameter_name in SEARCH_PARAM_ORDER:
        step_rows: list[dict[str, object]] = []
        for candidate in COARSE_CANDIDATES[parameter_name]:
            candidate_params = current_params.copy()
            candidate_params[parameter_name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                candidate_params,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            row = build_search_row(
                stage="coarse",
                search_id=search_id,
                tuned_parameter=parameter_name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_coarse_step(pd.DataFrame(step_rows))
        best_step_row = ranked_step_df.iloc[0]
        current_params = {name: best_step_row[name] for name in SEARCH_PARAM_NAMES}
        selected_search_ids.append(int(best_step_row["search_id"]))

    coarse_df = pd.DataFrame(all_rows)
    coarse_df["step_selected"] = coarse_df["search_id"].isin(selected_search_ids)
    coarse_best_valid_auc = float(coarse_df.loc[coarse_df["search_id"].isin(selected_search_ids), "valid_auc"].iloc[-1])
    return coarse_df, current_params, coarse_best_valid_auc


# 第二轮围绕粗搜最优做微调，但只接受 valid_auc 不低于粗搜最优的候选；
# 如果所有细搜都达不到这个门槛，就直接回退到粗搜参数。
def run_fine_search(
    coarse_params: dict[str, object],
    coarse_best_valid_auc: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """执行第二轮细搜，并只接受验证集 AUC 不低于第一轮最优的参数。"""
    current_params = coarse_params.copy()
    all_rows: list[dict[str, object]] = []
    selected_search_ids: list[int] = []
    search_id = 1
    has_any_update = False

    for parameter_name in SEARCH_PARAM_ORDER:
        step_rows: list[dict[str, object]] = []
        for candidate in build_fine_candidates(parameter_name, current_params[parameter_name]):
            candidate_params = current_params.copy()
            candidate_params[parameter_name] = candidate
            _, train_metrics, valid_metrics = fit_and_evaluate_on_holdout(
                candidate_params,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            row = build_search_row(
                stage="fine",
                search_id=search_id,
                tuned_parameter=parameter_name,
                params=candidate_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            all_rows.append(row)
            step_rows.append(row)
            search_id += 1

        ranked_step_df = rank_fine_step(pd.DataFrame(step_rows), coarse_best_valid_auc)
        qualified_step_df = ranked_step_df.loc[
            ranked_step_df["meet_fine_selection_requirements"]
        ].reset_index(drop=True)
        if qualified_step_df.empty:
            continue

        best_step_row = qualified_step_df.iloc[0]
        current_params = {name: best_step_row[name] for name in SEARCH_PARAM_NAMES}
        selected_search_ids.append(int(best_step_row["search_id"]))
        has_any_update = True

    fine_df = pd.DataFrame(all_rows)
    fine_df["meet_coarse_valid_auc_floor"] = fine_df["valid_auc"] >= coarse_best_valid_auc
    fine_df["meet_fine_selection_requirements"] = (
        fine_df["meet_coarse_valid_auc_floor"]
        & fine_df["meet_valid_auc_le_train_auc"]
    )
    fine_df["step_selected"] = fine_df["search_id"].isin(selected_search_ids)
    final_stage = "fine" if has_any_update else "coarse"
    return fine_df, current_params if has_any_update else coarse_params, final_stage


# 最终评估仍保持严格 holdout 结构：模型只在 train 上拟合，
# 再分别汇报 train/valid/test 三段指标，用于解释筛选模型的泛化差异。
def fit_final_model(
    final_params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LGBMClassifier, pd.DataFrame]:
    """用最终参数拟合模型并输出三段评估。"""
    model = build_model(final_params)
    model.fit(X_train, y_train)
    metrics_df = pd.concat(
        [
            evaluate_model(model, X_train, y_train, "train"),
            evaluate_model(model, X_valid, y_valid, "valid"),
            evaluate_model(model, X_test, y_test, "test"),
        ],
        ignore_index=True,
    )
    return model, metrics_df


# 重要性表使用 LightGBM 的 gain 重要性，并只对 30 个候选变量排序。
def build_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    """把模型特征重要性整理成表。"""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("当前筛选模型不支持 feature_importances_。")

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df.loc[:, ["feature_name", "importance"]]


# best_params 报表除了最终取值，还会带上每个超参数的含义、搜索范围和调参趋势，
# 便于把搜索日志和理论解释放到同一张表里。
def build_best_params_frame(
    final_params: dict[str, object],
    metrics_df: pd.DataFrame,
    coarse_best_valid_auc: float,
    final_stage: str,
) -> pd.DataFrame:
    """整理最终参数和选择依据。"""
    train_auc = metrics_df.loc[metrics_df["dataset"] == "train", "auc"].iloc[0]
    valid_auc = metrics_df.loc[metrics_df["dataset"] == "valid", "auc"].iloc[0]
    test_auc = metrics_df.loc[metrics_df["dataset"] == "test", "auc"].iloc[0]
    auc_gap = abs(train_auc - valid_auc)

    rows = []
    for name in SEARCH_PARAM_NAMES:
        doc = SEARCH_PARAM_DOCS[name]
        rows.append(
            {
                "parameter": name,
                "value": final_params[name],
                "meaning": doc["meaning"],
                "search_range": doc["search_range"],
                "trend_note": doc["trend"],
            }
        )
    rows.extend(
        [
            {
                "parameter": "coarse_best_valid_auc",
                "value": coarse_best_valid_auc,
                "meaning": "第一轮粗搜锁定参数后的最高 valid AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_selection_stage",
                "value": final_stage,
                "meaning": "最终参数来自 coarse 还是 fine。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "selection_rule",
                "value": "第一轮逐个参数只接受 valid_auc 不高于 train_auc 的候选，再取 valid_auc 最高；第二轮只接受 valid_auc 不低于第一轮最优且 valid_auc 不高于 train_auc 的候选，再从中选 train_valid_auc_gap 最小的一组",
                "meaning": "当前统一筛选的两轮调参规则。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_train_auc",
                "value": train_auc,
                "meaning": "最终参数在训练集上的 AUC。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_valid_auc",
                "value": valid_auc,
                "meaning": "最终参数在验证集上的 AUC，是筛选规则的核心比较指标。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_test_auc",
                "value": test_auc,
                "meaning": "最终参数在测试集上的 AUC，用于事后检查泛化。",
                "search_range": "",
                "trend_note": "",
            },
            {
                "parameter": "final_train_valid_auc_gap",
                "value": auc_gap,
                "meaning": "最终 train AUC 与 valid AUC 的绝对差，用来观察过拟合程度。",
                "search_range": "",
                "trend_note": "",
            },
        ]
    )
    return pd.DataFrame(rows)


# 筛选模型固定要求显式提供 --horizon。
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    return parser.parse_args()


# screen_lgbm.py 到这里结束“统一筛选”主线：
# 输入是 screening_long_panel，输出是搜索日志、重要性、指标和筛选模型文件。
def main() -> None:
    """执行 LightGBM 两轮筛选搜索并输出重要性与评估结果。"""
    args = parse_args()
    ctx = get_horizon_context(horizon=args.horizon)
    cleanup_legacy_output_files(ctx)

    panel_df, feature_cols, target_col = load_screening_long_panel(args.horizon)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_to_xy(
        panel_df,
        feature_cols,
        target_col,
    )

    coarse_df, coarse_params, coarse_best_valid_auc = run_coarse_search(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    fine_df, final_params, final_stage = run_fine_search(
        coarse_params,
        coarse_best_valid_auc,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    final_model, metrics_df = fit_final_model(
        final_params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
    importance_df = build_importance_frame(final_model, feature_cols)
    best_params_df = build_best_params_frame(
        final_params,
        metrics_df,
        coarse_best_valid_auc=coarse_best_valid_auc,
        final_stage=final_stage,
    )
    screening_scheme_name = ctx.config.screening_scheme_name(ctx.horizon)
    model_file = ctx.paths.model_file(MODEL_ID, screening_scheme_name)

    ctx.paths.ensure_screening_output_dir()
    coarse_file = ctx.paths.screening_output_file("screening_lgbm_coarse_search.csv")
    fine_file = ctx.paths.screening_output_file("screening_lgbm_fine_search.csv")
    importance_file = ctx.paths.screening_output_file("screening_lgbm_importance.csv")
    metrics_file = ctx.paths.screening_output_file("screening_lgbm_metrics.csv")
    best_params_file = ctx.paths.screening_output_file("screening_lgbm_best_params.csv")
    save_model_bundle(
        model=final_model,
        model_file=model_file,
        model_id=MODEL_ID,
        feature_names=feature_cols,
    )

    coarse_save_df = coarse_df.drop(columns=SEARCH_LOG_HELPER_COLS, errors="ignore")
    fine_save_df = fine_df.drop(columns=SEARCH_LOG_HELPER_COLS, errors="ignore")

    coarse_save_df.to_csv(coarse_file, index=False, encoding="utf-8-sig")
    fine_save_df.to_csv(fine_file, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_file, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    best_params_df.to_csv(best_params_file, index=False, encoding="utf-8-sig")

    print("统一筛选 LightGBM 输出已保存：")
    print(coarse_file)
    print(fine_file)
    print(importance_file)
    print(metrics_file)
    print(best_params_file)
    print(model_file)
    print("\n最终参数：")
    print(best_params_df)
    print("\n最终三段评估：")
    print(metrics_df)


if __name__ == "__main__":
    main()
