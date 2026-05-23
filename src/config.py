"""配置索引。

当前维护的配置类别：
1. 路径与目录：`ROOT_DIR`、`DATA_DIR`、`RAW_DIR`、`PROCESSED_DIR`、`MODELS_DIR`、`OUTPUTS_DIR`。
2. 资产与原始数据映射：`ASSET_SPECS`、`ASSET_ORDER`。
3. 特征与目标期限：`HORIZONS`。
4. 统一筛选范围：`SCREENING_ASSETS`、`SCREENING_CANDIDATE_COLUMNS`。
5. 正式预测范围：`PAPER_ASSETS`、`PAPER_HORIZONS`、`MAINLINE_CLASSIFICATION_MODELS`、`MAINLINE_SCHEMES`、`MAINLINE_EXPERIMENT_TAGS`。
6. 固定交易流变量与正式控制变量：`FIXED_TRADEFLOW_COLUMNS`、`BEST8_CONTROL_COLUMNS`。
7. 文件命名与路径组织：各类 `*_filename`、`*_dir`、`tagged_*` 辅助函数。

配置类别与影响步骤：
- `ASSET_SPECS`：决定第 01 步读取哪个 sheet，并影响对应资产从第 01 步到第 12 步的整条分支。
- `HORIZONS`：决定第 02 步构造哪些目标列。
- `SCREENING_ASSETS`、`SCREENING_CANDIDATE_COLUMNS`：决定第 03 到第 07 步统一筛选输入范围。
- `PAPER_ASSETS`、`PAPER_HORIZONS`、`MAINLINE_EXPERIMENT_TAGS`：决定第 08 到第 12 步正式预测目录和结果命名。
- `MAINLINE_CLASSIFICATION_MODELS`、`MAINLINE_SCHEMES`：决定第 09 到第 11 步训练、汇总和章节结果覆盖范围。
- `FIXED_TRADEFLOW_COLUMNS`：决定第 02、06、08、10、11 步涉及的固定交易流变量口径。
- `BEST8_CONTROL_COLUMNS`：决定第 07 步输出的正式控制变量清单，并影响第 08 到第 12 步全部结果。

`README.md` 中的“关键修改位置”“变量筛选逻辑”和“重跑关系”均以这里的配置分类为准。
"""

from __future__ import annotations

import re
from pathlib import Path


# === 路径根目录 ===
# 这一组路径常量定义了“原始数据、中间数据、模型结果、汇总结果”分别放在哪里。
# 正常运行时通常不需要改这里；只有在整个项目目录被搬动时，这些路径才会自动跟着变。
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# === 资产主表 ===
# 每个资产同时记录了两个信息：
# 1）asset_code：写入中间表和结果表时使用的标准代码；
# 2）sheet_name：第 01 步去原始 Excel 里读取哪个 sheet。
# 新增资产时，首先在这里补上资产代码和对应的 Excel sheet 名称，
# 然后再从第 01 步开始重建该资产链路。
ASSET_SPECS = {
    "index": {"asset_code": "932077.CSI", "sheet_name": "中证能源行业指数932077"},
    "sc": {"asset_code": "SC00.INE", "sheet_name": "原油期货近月连续"},
    "lu": {"asset_code": "LU00.INE", "sheet_name": "低硫燃料油近月连续"},
    "pp": {"asset_code": "PP00.DCE", "sheet_name": "聚丙烯近月连续"},
    "eg": {"asset_code": "EG00.DCE", "sheet_name": "乙二醇近月连续"},
    "bu": {"asset_code": "BU00.SHF", "sheet_name": "沥青近月连续"},
    "jm": {"asset_code": "JM00.DCE", "sheet_name": "焦煤近月连续"},
}

ASSET_ORDER = list(ASSET_SPECS.keys())
# HORIZONS 是特征工程阶段会同时构造的未来收益和涨跌标签口径。
# 这里仍保留 1/22/44/66/88 五个期限，是为了让 feature_panel.csv 中的目标变量口径完整一致，
# 但正式建模目录只使用 PAPER_HORIZONS 指定的 1 日和 22 日。
# 新增正式预测期限时，先改 PAPER_HORIZONS；如果 feature_panel 中的目标列也要同步扩展，再检查这里。
HORIZONS = [1, 22, 44, 66, 88]
MODEL_IDS = ["dt_cls", "rf_cls", "svm_cls", "xgb_cls", "lgbm_cls"]
SCHEMES = ["with_tradeflow4", "no_tradeflow4"]

# === 关键实验位置 ===
# 下面这一组常量对应 README 和步骤头注释中反复引用的关键修改位置。
# 资产、期限、模型、方案和变量口径的主要调整都集中在这里。
# `SCREENING_ASSETS` 决定第 03 到第 06 步统一筛选和辅助诊断会读取哪些资产。
# `PAPER_ASSETS` 决定第 08 到第 11 步真正进入建模和结果整理的预测对象。
# `PAPER_HORIZONS` 决定第 08 到第 11 步会对哪些预测期限生成建模输入、训练结果和图表。
# `MAINLINE_CLASSIFICATION_MODELS` 决定第 09、10、11 步会涉及哪些分类模型。
# `MAINLINE_SCHEMES` 决定是否同时比较“加入交易流”和“不加入交易流”两种方案。
# `MAINLINE_EXPERIMENT_TAGS` 决定不同期限结果写入哪个标签目录，改这里后要同步第 08、09、10 步目录命名。
SCREENING_ASSETS = ["index", "sc", "lu", "pp", "eg", "bu", "jm"]
# 第 03 到第 07 步统一筛选固定使用这 7 个资产。
# 如果这里删减资产，应从第 03 步开始重跑统一筛选与筛选参考。
SCREENING_HORIZON = 1
# 第 04 到第 06 步统一筛选诊断和筛选模型固定使用 1 日涨跌标签口径。
# 如果这里修改，需要同步检查 screening_long_panel 的目标列口径、筛选输出命名和 README 说明。
PAPER_ASSETS = ["index", "eg", "bu", "jm", "pp"]
# 第 08 到第 12 步只保留这 5 个预测对象。
# 如果这里修改，需要同步检查图表目录和重点结果表是否需要重建。
PAPER_HORIZONS = [1, 22]
# 当前主线只保留 1 日和 22 日两个正式预测期限。
# 如果在这里新增期限，后续至少要重跑第 08 到第 12 步，并重新生成新的建模目录、训练结果和图表结果。
MAINLINE_CLASSIFICATION_MODELS = MODEL_IDS.copy()
MAINLINE_SCHEMES = SCHEMES.copy()
MAINLINE_EXPERIMENT_TAGS = {
    1: "best8__a4pp_1d",
    22: "best8__a4pp_22d",
}

# === 原始 Excel 直接读取字段 ===
CORE_SOURCE_COLUMNS = [
    "IND_SECTOR_TV_ene_norm",
    "INS_SECTOR_TV_ene_norm",
    "ITVvar",
    "sigpre",
    "sigpre30",
    "dolsha30",
    "dolsha",
    "iVX",
    "FirmBondAA10Y",
    "ChBond10Y",
    "ChBond3M",
    "R6M",
]

MARKET_SOURCE_COLUMNS = [
    "close",
    "volume",
    "amt",
    "turn",
    "MACD",
    "RSI",
    "OBV",
    "BIAS",
    "BOLL",
    "PVT",
    "DMI",
]

# === 固定保留的交易流变量 ===
# 这 4 列是主线固定的交易流变量。
# 交易流变量口径变更时，修改这里，并从第 02 步开始重跑。
FIXED_TRADEFLOW_COLUMNS = [
    "IND_SECTOR_TV_ene_norm",
    "INS_SECTOR_TV_ene_norm",
    "ITVvar",
    "ITVvar_x_dolsha",
]

# === pooled screening 的候选控制变量 ===
# 统一筛选阶段只会在这组候选控制变量里比较重要性。
# 候选控制变量新增或删除后，第 03 到第 07 步都需要重跑。
SCREENING_CANDIDATE_COLUMNS = [
    "sigpre",
    "sigpre30",
    "dolsha30",
    "dolsha",
    "iVX",
    "cred",
    "liqu",
    "volume",
    "amt",
    "turn",
    "MACD",
    "RSI",
    "OBV",
    "BIAS",
    "BOLL",
    "PVT",
    "DMI",
    "log_return_lag1",
    "log_return_abs_lag1",
    "log_return_lag5",
    "log_return_abs_lag5",
    "log_return_lag22",
    "log_return_abs_lag22",
    "vol_h5d",
    "mom_h5d",
    "vol_h22d",
    "mom_h22d",
]

# === 参考删列记录（仅作筛选说明，不直接驱动主线） ===
REFERENCE_DROP_COLUMNS = [
    "log_return_abs_lag1",
    "amt",
    "dolsha30",
    "log_return_abs_lag22",
    "log_return_abs_lag5",
    "BOLL",
    "sigpre30",
    "dolsha",
    "log_return_lag22",
    "PVT",
    "vol_h22d",
    "log_return_lag5",
    "MACD",
    "vol_h5d",
    "turn",
]

# === 当前主线固定控制变量 ===
# 这 8 个变量是当前主线正式建模保留的控制变量。
# 第 07 步默认会把这组变量写入 selected_features_best8_controls.csv，
# 第 08 步如果不额外指定变量文件，就会继续沿用这里对应的口径准备建模输入。
BEST8_CONTROL_COLUMNS = [
    "sigpre",
    "iVX",
    "volume",
    "RSI",
    "BIAS",
    "DMI",
    "log_return_lag1",
    "mom_h5d",
]

CANONICAL_MAINLINE_CONTROL_COLUMNS = BEST8_CONTROL_COLUMNS.copy()

REFERENCE_SELECTED_COLUMNS = CANONICAL_MAINLINE_CONTROL_COLUMNS.copy()

# === 变量类别说明：供筛选说明复用 ===
CONTROL_CATEGORY_RULES = {
    "macro": ["iVX", "cred", "liqu"],
    "quantity_price": ["volume", "amt", "turn"],
    "technical": ["MACD", "RSI", "OBV", "BIAS", "BOLL", "PVT", "DMI"],
    "return_vol_momentum": [
        "log_return_lag1",
        "log_return_abs_lag1",
        "log_return_lag5",
        "log_return_abs_lag5",
        "log_return_lag22",
        "log_return_abs_lag22",
        "vol_h5d",
        "mom_h5d",
        "vol_h22d",
        "mom_h22d",
    ],
    "behavior": ["sigpre", "sigpre30", "dolsha", "dolsha30"],
}

MANDATORY_COVERAGE_CATEGORIES = ["macro", "quantity_price", "technical", "return_vol_momentum"]

# === feature_panel 与 screening 面板结构 ===
FEATURE_PANEL_DROP_COLUMNS = [
    "close",
    "FirmBondAA10Y",
    "ChBond10Y",
    "ChBond3M",
    "R6M",
]

SCREENING_FEATURE_COLUMNS = FIXED_TRADEFLOW_COLUMNS + SCREENING_CANDIDATE_COLUMNS

# === 原始文件名与标准输出根目录 ===
RAW_CORE_FILE_NAME = "核心变量时间序列日度.xlsx"
RAW_MARKET_FILE_NAME = "指标数据补充.xlsx"

SCREENING_DIR = PROCESSED_DIR / "screening"
# 本项目直接在 outputs/ 和 models/ 下组织筛选结果、分类结果、汇总结果和模型文件。
# 查看结果目录时，可以直接在这些一级目录下顺着章节结果、训练结果和模型文件继续往下定位。
MAINLINE_OUTPUT_DIR = OUTPUTS_DIR
MAINLINE_MODELS_DIR = MODELS_DIR
SCREENING_OUTPUT_DIR = MAINLINE_OUTPUT_DIR / "screening"
CLASSIFICATION_OUTPUT_DIR = MAINLINE_OUTPUT_DIR / "classification"
CLASSIFICATION_SUMMARY_DIR = MAINLINE_OUTPUT_DIR / "summary"
CLASSIFICATION_SUMMARY_DEFAULT_SLUG = "default"


def asset_processed_dir(asset: str) -> Path:
    """返回单个资产在 data/processed 下的目录。"""
    # 第 01、02、08 步都会先定位到这个目录，再读写该资产的中间文件。
    return PROCESSED_DIR / asset


def normalize_name_part(name: str | None) -> str | None:
    """清洗名称片段，过滤空值并阻止路径分隔符混入。"""
    if name is None:
        return None
    normalized = str(name).strip()
    if not normalized:
        return None
    if any(char in normalized for char in ("/", "\\")):
        raise ValueError(f"invalid name part: {name}")
    return normalized


def build_name_suffix(*parts: str | None) -> str:
    """把若干名称片段拼成统一的双下划线后缀。"""
    # 这个函数主要服务实验标签和目录命名。
    # 结果目录名里出现 __best8__a4pp_1d 这类后缀时，就是这里拼出来的。
    normalized_parts = []
    for part in parts:
        normalized = normalize_name_part(part)
        if normalized:
            normalized_parts.append(normalized)
    return "".join(f"__{part}" for part in normalized_parts)


_HASHED_EXPERIMENT_TAG_PATTERN = re.compile(r"^c(?P<count>\d+)_(?P<digest>[0-9a-fA-F]{8,})(?:__(?P<rest>.+))?$")


def canonical_experiment_slug(experiment_tag: str | None) -> str | None:
    """把实验标签规整成稳定 slug，便于目录名和汇总文件名复用。"""
    normalized = normalize_name_part(experiment_tag)
    if normalized is None:
        return None
    match = _HASHED_EXPERIMENT_TAG_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized

    count = match.group("count")
    digest = match.group("digest").lower()[:8]
    raw_rest = match.group("rest")
    if not raw_rest:
        return f"subset__c{count}__h{digest}"

    tokens = raw_rest.split("__")
    if tokens[0] == "merged":
        merged_features = []
        suffix_tokens = []
        for token in tokens[1:]:
            if token in SCREENING_CANDIDATE_COLUMNS and not suffix_tokens:
                merged_features.append(token)
            else:
                suffix_tokens.append(token)
        if merged_features:
            return "__".join([f"merged_{'__'.join(merged_features)}", f"c{count}", f"h{digest}", *suffix_tokens])

    if len(tokens) > 1:
        return "__".join([tokens[0], f"c{count}", f"h{digest}", *tokens[1:]])

    return "__".join(["subset", f"c{count}", f"h{digest}", tokens[0]])


def classification_summary_slug(experiment_tag: str | None) -> str:
    """返回分类汇总文件使用的实验标签 slug。"""
    return canonical_experiment_slug(experiment_tag) or CLASSIFICATION_SUMMARY_DEFAULT_SLUG


def classification_summary_filename(
    kind: str,
    experiment_tag: str | None = None,
    scheme: str | None = None,
    model_id: str | None = None,
    extension: str = "csv",
) -> str:
    """按汇总类型、方案和实验标签生成标准汇总文件名。"""
    # 第 10 步和第 11 步都依赖这一组命名规则。
    # experiment_tag 改变后，这里生成的文件名也会随之改变，因此第 08 到第 11 步的标签必须一致。
    slug = classification_summary_slug(experiment_tag)
    if kind == "metrics":
        if not scheme:
            raise ValueError("scheme is required for metrics summaries")
        return f"classification_metrics__{scheme}__{slug}.{extension}"
    if kind == "best_valid_auc":
        if not scheme:
            raise ValueError("scheme is required for best_valid_auc summaries")
        return f"classification_best_valid_auc__{scheme}__{slug}.{extension}"
    if kind == "anomalies":
        if not scheme:
            raise ValueError("scheme is required for anomaly summaries")
        return f"classification_anomalies__{scheme}__{slug}.{extension}"
    if kind == "ablation":
        return f"classification_ablation__{slug}.{extension}"
    if kind == "ablation_overall":
        return f"classification_ablation_overall__{slug}.{extension}"
    if kind == "test_by_model":
        if model_id:
            return f"classification_test_by_model__{model_id}__{slug}.{extension}"
        return f"classification_test_by_model__{slug}.{extension}"
    raise ValueError(f"unknown classification summary kind: {kind}")


def asset_horizon_dir(asset: str, horizon: int, scheme: str | None = None, experiment_tag: str | None = None) -> Path:
    """返回单资产单期限建模输入目录。"""
    # 第 08 步会把 prepared/model_input 文件写到这里，
    # 第 09 步再从这个目录读取 train/valid/test 三份输入文件。
    base = asset_processed_dir(asset) / "horizons" / f"{horizon}d"
    experiment_suffix = build_name_suffix(experiment_tag)
    if not experiment_suffix and scheme in (None, "with_tradeflow4"):
        return base
    scheme_name = scheme or "with_tradeflow4"
    return base / f"{scheme_name}{experiment_suffix}"


def classification_model_dir(model_id: str, scheme_name: str, model_root: Path | None = None) -> Path:
    """返回单个分类 run 的模型目录。"""
    # 第 09 步训练完成后会把 joblib 模型文件写到这里。
    base_dir = model_root or (MAINLINE_MODELS_DIR / "classification")
    return base_dir / model_id / scheme_name


def classification_output_dir(model_id: str, scheme_name: str, output_root: Path | None = None) -> Path:
    """返回单个分类 run 的结果目录。"""
    # 第 09 步的 metrics.csv、pred_test.csv、feature_importance.csv 等文件都写在这里。
    base_dir = output_root or CLASSIFICATION_OUTPUT_DIR
    return base_dir / model_id / scheme_name


def classification_summary_dir(summary_root: Path | None = None) -> Path:
    """返回分类汇总目录。"""
    return summary_root or CLASSIFICATION_SUMMARY_DIR


def parse_csv_list(raw: str) -> list[str]:
    """把逗号分隔字符串解析成去空白后的列表。"""
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_assets(raw: str) -> list[str]:
    """解析资产参数，并校验是否属于当前主线资产集合。"""
    # 命令行里写 all 时，表示使用当前配置里登记的全部资产。
    if raw == "all":
        return ASSET_ORDER.copy()
    assets = parse_csv_list(raw)
    unknown = set(assets) - set(ASSET_ORDER)
    if unknown:
        raise ValueError(f"Unknown assets: {sorted(unknown)}")
    return assets


def parse_models(raw: str) -> list[str]:
    """解析模型参数，并校验是否属于当前主线模型集合。"""
    if raw == "all":
        return MODEL_IDS.copy()
    model_ids = parse_csv_list(raw)
    unknown = set(model_ids) - set(MODEL_IDS)
    if unknown:
        raise ValueError(f"Unknown model ids: {sorted(unknown)}")
    return model_ids


def parse_horizons(raw: str) -> list[int]:
    """解析期限参数，并校验是否属于当前主线期限集合。"""
    if raw == "all":
        return HORIZONS.copy()
    horizons = [int(item) for item in parse_csv_list(raw)]
    unknown = set(horizons) - set(HORIZONS)
    if unknown:
        raise ValueError(f"Unknown horizons: {sorted(unknown)}")
    return horizons


def parse_scheme(raw: str) -> str:
    """解析方案参数，并校验是否属于当前特征方案集合。"""
    if raw not in SCHEMES:
        raise ValueError(f"Unknown scheme: {raw}")
    return raw


def experiment_tag_for_horizon(horizon: int) -> str | None:
    """按期限返回当前主线默认实验标签。"""
    if horizon not in MAINLINE_EXPERIMENT_TAGS:
        raise ValueError(f"Missing experiment tag for horizon: {horizon}")
    return MAINLINE_EXPERIMENT_TAGS[horizon]


def run_name(asset: str, horizon: int, scheme: str) -> str:
    """生成不带实验标签的基础 run 名称。"""
    return f"{asset}_{horizon}d__{scheme}"


def tagged_run_name(asset: str, horizon: int, scheme: str, experiment_tag: str | None = None) -> str:
    """生成带实验标签的标准 run 名称。"""
    # 第 08、09、10 步必须共享同一套 run_name 规则，否则会出现“准备好的输入目录”和“训练结果目录”对不上的问题。
    slug = canonical_experiment_slug(experiment_tag)
    if slug is None:
        return run_name(asset, horizon, scheme)
    return f"{run_name(asset, horizon, scheme)}__{slug}"
