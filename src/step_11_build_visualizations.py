from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 11 步：生成结果图片、配套表格和结果索引
# 当前主线位置：筛选结果、训练结果、汇总结果 -> 章节图表与结果索引。
# 本文件负责：训练完成后，从已有结果独立生成图片、章节 Excel、章节结果目录表、图片质量检查表和结果索引。
# 主函数：`run()`。
# 命令行入口：`main()`，负责触发 `run()`。
# 直接输入：`outputs/screening/*`；`outputs/classification/*`；`outputs/summary/*`；`data/processed/*`；`data/processed/screening/screening_reference_mainline_controls.csv`。
# 直接输出：`outputs/visualizations/<chapter>/figures/*.png`；`outputs/visualizations/<chapter>/tables_excel/*.xlsx`；`outputs/visualizations/结果索引/章节结果目录表.csv`；`outputs/visualizations/结果索引/章节结果索引.md`；`outputs/visualizations/结果索引/图片质量检查表.csv`。
# 下游读取：第 12 步 `src.step_12_build_inventory`。
# 关键修改位置：`src/common/visual_style.py`；本文件中的章节目录、结果索引写法、重点结果表组织方式和结果记录规则。
# 变更后重跑起点：第 11 步。前提是第 08 到第 10 步的上游结果已经按当前口径准备好。
# 对应文档：`README.md` 的“主线结构总览”“结果目录与阅读顺序”和“出错时优先检查”。
import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.common.paper_common import (
    BEST8_ASSETS,
    BEST8_CONTROLS,
    MAINLINE_CLASSIFICATION_OUTPUT_DIR,
    MAINLINE_SUMMARY_DIR,
    TRADEFLOW4_COLUMNS,
    VISUALIZATION_DIR,
    ensure_best8_dirs,
)
from src.common.visual_style import (
    annotate_bars,
    annotate_bars_horizontal,
    annotate_heatmap,
    configure_matplotlib,
    gray_colors,
    save_figure,
    style_axis,
)
from src.config import (
    FIXED_TRADEFLOW_COLUMNS,
    MODEL_IDS,
    MAINLINE_EXPERIMENT_TAGS,
    OUTPUTS_DIR,
    PAPER_HORIZONS,
    PAPER_ASSETS,
    PROCESSED_DIR,
    ROOT_DIR,
    SCREENING_ASSETS,
    SCREENING_CANDIDATE_COLUMNS,
    SCREENING_OUTPUT_DIR,
    asset_processed_dir,
    classification_summary_filename,
    tagged_run_name,
)
from src.common.metrics_utils import safe_auc


ASSET_LABELS = {
    "index": "中证能源指数",
    "eg": "乙二醇",
    "bu": "沥青",
    "jm": "焦煤",
    "pp": "聚丙烯",
}
MODEL_LABELS = {
    "dt_cls": "决策树",
    "rf_cls": "随机森林",
    "svm_cls": "支持向量机",
    "xgb_cls": "XGBoost",
    "lgbm_cls": "LightGBM",
}
SCHEME_LABELS = {
    "with_tradeflow4": "加入交易流",
    "no_tradeflow4": "不加入交易流",
}
TRADEFLOW_LABELS = {
    "IND_SECTOR_TV_ene_norm": "行业交易流",
    "INS_SECTOR_TV_ene_norm": "机构交易流",
    "ITVvar": "交易分歧",
    "ITVvar_x_dolsha": "交易分歧×多空分歧",
}
FEATURE_LABEL_OVERRIDES = {
    "iVX": "隐含波动率",
    "volume": "成交量",
    "RSI": "相对强弱指标",
    "BIAS": "乖离率",
    "DMI": "动向指标",
    "log_return_lag1": "1日滞后收益",
    "mom_h5d": "5日动量",
    "mom_h22d": "22日动量",
    "vol_h22d": "22日波动率",
    "cred": "信用利差",
    "liqu": "流动性利差",
    "amt": "成交额",
    "OBV": "能量潮指标",
    "MACD": "指数平滑异同移动平均",
    "BOLL": "布林带指标",
    "PVT": "价量趋势指标",
    "log_return_lag5": "5日滞后收益",
    "log_return_abs_lag1": "1日收益绝对值",
    "log_return_abs_lag5": "5日收益绝对值",
    "log_return_lag22": "22日滞后收益",
    "log_return_abs_lag22": "22日收益绝对值",
    "vol_h5d": "5日波动率",
}
HORIZON_TO_EXPERIMENT_TAG = MAINLINE_EXPERIMENT_TAGS.copy()
TREE_MODEL_IDS = ["dt_cls", "rf_cls", "xgb_cls", "lgbm_cls"]
STATE_ORDER = ["上涨低波动", "上涨高波动", "下跌低波动", "下跌高波动"]
KEY_RESULT_TABLE_ITEM_TYPE = "key_result_table_xlsx"
KEY_RESULT_ROW_LABELS = {
    "no_tradeflow4": "基准模型",
    "with_tradeflow4": "扩展模型",
}
KEY_RESULT_METRIC_COLUMNS = [
    ("accuracy", "准确率"),
    ("precision", "精确率"),
    ("recall", "召回率"),
    ("f1", "F1"),
    ("auc", "AUC"),
]
LIGHT_GRAYS_CMAP = LinearSegmentedColormap.from_list(
    "light_grays_paper",
    ["#f6f6f6", "#e6e6e6", "#d2d2d2", "#b5b5b5", "#8a8a8a", "#5f5f5f"],
)
CHAPTER_SPECS = {
    "5.3.1 目标变量": "5.3 特征工程与数据准备/5.3.1 目标变量",
    "5.3.2 特征集": "5.3 特征工程与数据准备/5.3.2 特征集",
    "5.3.3 特征标准化": "5.3 特征工程与数据准备/5.3.3 特征标准化",
    "5.3.4 数据划分": "5.3 特征工程与数据准备/5.3.4 数据划分",
    "5.3.5 处理类别不平衡": "5.3 特征工程与数据准备/5.3.5 处理类别不平衡",
    "5.4.1 超参数搜索过程": "5.4 模型训练与超参数优化/5.4.1 超参数搜索过程",
    "5.4.2 最优模型分布与选择": "5.4 模型训练与超参数优化/5.4.2 最优模型分布与选择",
    "5.4.3 基准方案与扩展方案对比框架": "5.4 模型训练与超参数优化/5.4.3 基准方案与扩展方案对比框架",
    "5.5.1 分类任务": "5.5 预测效果评估/5.5.1 分类任务",
    "5.5.2 基准模型对比": "5.5 预测效果评估/5.5.2 基准模型对比",
    "5.6 基于变量重要性的可解释性分析": "5.6 基于变量重要性的可解释性分析",
    "5.7.1 状态划分": "5.7 交易流指标在不同市场状态下的预测表现/5.7.1 状态划分",
    "5.7.2 分组预测": "5.7 交易流指标在不同市场状态下的预测表现/5.7.2 分组预测",
    "5.7.3 稳健性检验": "5.7 交易流指标在不同市场状态下的预测表现/5.7.3 稳健性检验",
}
LEGACY_COMPOSITE_FILE_NAMES = {
    "best8_breakdown.png",
    "best8_horizon_overview.png",
    "best8_1d_breakdown.png",
    "best8_22d_breakdown.png",
    "best8_horizon_comparison.png",
}
RESULT_INDEX_DIRNAME = "结果索引"
FORMAL_PAPER_RESULT_DIRNAME = "正式论文结果"
ROOT_PAPER_RESULT_DIR = ROOT_DIR / "论文结果_混合阈值F1提升"


@dataclass
class ChapterPaths:
    chapter_title: str
    root: Path
    figures_dir: Path
    tables_excel_dir: Path
    key_result_tables_excel_dir: Path | None = None


# ===== 第一组：名称映射与目录准备 =====
# 这一组函数只负责把代码里的资产、模型和章节标识，映射成最终展示的中文名称和目录结构。
# 章节目录名、中文资产名或图表根目录的调整，优先从这里往下看。

def feature_label(name: str) -> str:
    """把特征代码映射成图表和表格里使用的中文名称。"""
    if name in TRADEFLOW_LABELS:
        return TRADEFLOW_LABELS[name]
    return FEATURE_LABEL_OVERRIDES.get(name, name)


def asset_label(asset: str) -> str:
    """把资产代码映射成中文资产名称。"""
    return ASSET_LABELS.get(asset, asset)


def model_label(model_id: str) -> str:
    """把模型代码映射成图表和表格里使用的中文名称。"""
    return MODEL_LABELS.get(model_id, model_id)


def chapter_paths() -> dict[str, ChapterPaths]:
    """返回各章节图表目录、表格目录和重点结果表目录。"""
    mapping: dict[str, ChapterPaths] = {}
    for chapter_title, folder_name in CHAPTER_SPECS.items():
        root = VISUALIZATION_DIR / folder_name
        key_result_tables_excel_dir = None
        if chapter_title == "5.5.2 基准模型对比":
            key_result_tables_excel_dir = root / "tables_excel" / "重点结果表"
        mapping[chapter_title] = ChapterPaths(
            chapter_title=chapter_title,
            root=root,
            figures_dir=root / "figures",
            tables_excel_dir=root / "tables_excel",
            key_result_tables_excel_dir=key_result_tables_excel_dir,
        )
    return mapping


def remove_files_by_pattern(directory: Path, patterns: tuple[str, ...]) -> None:
    """按通配符删除当前目录下的生成型文件，保留静态正文文本。"""
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except PermissionError:
                continue


def prepare_visualization_root() -> None:
    """按当前论文结构全量重建可视化目录。"""
    root = VISUALIZATION_DIR.resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    if root == outputs_root or outputs_root not in root.parents:
        raise ValueError(f"Refuse to clear unsafe visualization path: {root}")
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    for child in VISUALIZATION_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    result_index_dir = VISUALIZATION_DIR / RESULT_INDEX_DIRNAME
    result_index_dir.mkdir(parents=True, exist_ok=True)
    for paths in chapter_paths().values():
        paths.figures_dir.mkdir(parents=True, exist_ok=True)
        paths.tables_excel_dir.mkdir(parents=True, exist_ok=True)
        if paths.key_result_tables_excel_dir is not None:
            paths.key_result_tables_excel_dir.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, output_path: Path, float_format: str | None = None) -> Path:
    """按路径后缀写出 CSV 或 Excel 表格。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="结果表")
        return output_path
    df.to_csv(output_path, index=False, encoding="utf-8-sig", float_format=float_format)
    return output_path


def excel_sheet_name(title: str) -> str:
    """把中文标题转换成 Excel 可接受的工作表名。"""
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", " ", title).strip()
    return (cleaned or "结果表")[:31]


def write_excel_sheet(df: pd.DataFrame, workbook_path: Path, sheet_name: str) -> Path:
    """把某张结果表写入章节级 Excel 工作簿的一个工作表。"""
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheet = excel_sheet_name(sheet_name)
    if workbook_path.exists():
        with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
    else:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return workbook_path


def add_record(
    records: list[dict[str, str]],
    chapter: str,
    item_type: str,
    title: str,
    path: Path,
    source: str,
    result_status: str,
    note: str = "",
    related_title: str = "",
    related_path: Path | str | None = None,
    excel_sheet: str = "",
) -> None:
    """向章节结果目录表追加一条图表或表格产出。"""
    related_path_str = ""
    if related_path not in (None, ""):
        related_path_str = str(related_path)
    records.append(
        {
            "chapter": chapter,
            "item_type": item_type,
            "title": title,
            "path": str(path),
            "source": source,
            "result_status": result_status,
            "note": note,
            "related_title": related_title,
            "related_path": related_path_str,
            "excel_sheet": excel_sheet,
        }
    )


# ===== 第二组：静态正文挂接 =====
# 第 11 步只保留图表、Excel 和索引生成能力。
# 已经存在于 tables_excel/ 下的正文文件由人工静态维护，这里只负责把它们登记到结果索引。

def join_source_paths(paths: list[Path]) -> str:
    """Join source paths used by chapter summaries."""
    return "; ".join(str(path) for path in paths)


def best8_feature_panel_sources() -> str:
    """Return all feature_panel.csv paths used by chapter-level summaries."""
    return join_source_paths([PROCESSED_DIR / asset / "feature_panel.csv" for asset in BEST8_ASSETS])


def best8_preprocess_stats_sources() -> str:
    """Return all screening_preprocess_stats.csv paths used by chapter-level summaries."""
    return join_source_paths([PROCESSED_DIR / asset / "screening_preprocess_stats.csv" for asset in BEST8_ASSETS])


def write_figure_result_excel_sheet(
    records: list[dict[str, str]],
    paths: ChapterPaths,
    figure_title: str,
    figure_path: Path,
    df: pd.DataFrame,
    source: str,
    result_status: str,
    note: str = "",
) -> Path:
    """Write one figure result sheet into the chapter workbook."""
    workbook_path = write_excel_sheet(df, paths.tables_excel_dir / "图表结果数据表.xlsx", figure_title)
    sheet_name = excel_sheet_name(figure_title)
    add_record(
        records,
        paths.chapter_title,
        "figure_result_excel_sheet",
        f"{figure_title}数据工作表",
        workbook_path,
        source,
        result_status,
        note=note,
        related_title=figure_title,
        related_path=figure_path,
        excel_sheet=sheet_name,
    )
    return workbook_path


def static_text_metadata(txt_path: Path) -> tuple[str, str, str]:
    """Infer the record metadata for one static prose file."""
    name = txt_path.name
    if name.endswith("_图表分析.txt"):
        base_title = name.removesuffix("_图表分析.txt")
        return ("figure_analysis_txt", f"{base_title}图表分析", base_title)
    if name.endswith("_结果分析.txt"):
        base_title = name.removesuffix("_结果分析.txt")
        return ("table_analysis_txt", f"{base_title}结果分析", base_title)
    return ("static_text_txt", txt_path.stem, txt_path.stem)


def find_related_record(
    records: list[dict[str, str]],
    chapter_title: str,
    text_item_type: str,
    related_title: str,
) -> dict[str, str] | None:
    """Find the figure or table record that a static prose file belongs to."""
    preferred_item_types = ("figure",) if text_item_type == "figure_analysis_txt" else ("table_xlsx",)
    for item_type in preferred_item_types:
        for record in records:
            if record["chapter"] == chapter_title and record["item_type"] == item_type and record["title"] == related_title:
                return record
    for record in records:
        if record["chapter"] == chapter_title and record["title"] == related_title:
            return record
    return None


def register_static_text_records(records: list[dict[str, str]], paths_map: dict[str, ChapterPaths]) -> None:
    """Register the manually maintained static text files into the result index."""
    existing_paths = {record["path"] for record in records}
    for chapter_title, paths in paths_map.items():
        for txt_path in sorted(paths.tables_excel_dir.glob("*.txt")):
            if str(txt_path) in existing_paths:
                continue
            item_type, title, related_title = static_text_metadata(txt_path)
            related_record = find_related_record(records, chapter_title, item_type, related_title)
            related_path = related_record["path"] if related_record is not None else ""
            excel_sheet = ""
            if item_type == "figure_analysis_txt":
                excel_sheet = excel_sheet_name(related_title)
            elif related_record is not None and related_record["item_type"] == "table_xlsx":
                excel_sheet = related_record.get("excel_sheet", "")
            add_record(
                records,
                chapter_title,
                item_type,
                title,
                txt_path,
                str(paths.tables_excel_dir),
                "静态正文",
                related_title=related_title,
                related_path=related_path,
                excel_sheet=excel_sheet,
            )
            existing_paths.add(str(txt_path))
def compute_key_result_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    """Compute the metric set used in key result tables."""
    y_true = pred_df["y_true"].astype(int)
    y_pred = pred_df["y_pred"].astype(int)
    y_score = pd.to_numeric(pred_df["y_score"], errors="coerce")
    return {
        "准确率": float(accuracy_score(y_true, y_pred)),
        "精确率": float(precision_score(y_true, y_pred, zero_division=0)),
        "召回率": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(safe_auc(y_true, y_score)),
    }


def classification_run_dir(model_id: str, asset: str, horizon: int, scheme: str) -> Path:
    """Return one classification run directory under outputs/classification."""
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    run_name = tagged_run_name(asset, horizon, scheme, experiment_tag)
    return MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name


def load_test_predictions(model_id: str, asset: str, horizon: int, scheme: str) -> pd.DataFrame:
    """Load pred_test.csv for one model x asset x horizon x scheme."""
    return pd.read_csv(classification_run_dir(model_id, asset, horizon, scheme) / "pred_test.csv")


def compute_test_prediction_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    """Compute the standard metric set from pred_test.csv."""
    raw_metrics = compute_key_result_metrics(pred_df)
    return {
        "accuracy": raw_metrics["准确率"],
        "precision": raw_metrics["精确率"],
        "recall": raw_metrics["召回率"],
        "f1": raw_metrics["F1"],
        "auc": raw_metrics["AUC"],
    }


def key_result_table_filename(model_id: str, asset: str, horizon: int) -> str:
    """Return the key-result workbook filename used under 5.5.2."""
    return f"{horizon}日_{asset_label(asset)}_{model_label(model_id)}重点结果表.xlsx"


def build_key_result_table(model_id: str, asset: str, horizon: int) -> pd.DataFrame:
    """Build one key result table for model x asset x horizon."""
    metrics_by_scheme: dict[str, dict[str, float]] = {}
    for scheme in ["no_tradeflow4", "with_tradeflow4"]:
        pred_df = load_test_predictions(model_id, asset, horizon, scheme)
        metrics_by_scheme[scheme] = compute_test_prediction_metrics(pred_df)
    rows = []
    for scheme in ["no_tradeflow4", "with_tradeflow4"]:
        row = {"项目": KEY_RESULT_ROW_LABELS[scheme]}
        for metric_key, label in KEY_RESULT_METRIC_COLUMNS:
            row[label] = round(metrics_by_scheme[scheme][metric_key], 3)
        rows.append(row)
    diff_row = {"项目": "结果对比"}
    for metric_key, label in KEY_RESULT_METRIC_COLUMNS:
        diff_row[label] = round(metrics_by_scheme["with_tradeflow4"][metric_key] - metrics_by_scheme["no_tradeflow4"][metric_key], 3)
    rows.append(diff_row)
    return pd.DataFrame(rows)


def build_key_result_table_index_title(model_id: str, asset: str, horizon: int) -> str:
    """Return the record title used in the result index csv."""
    return f"{horizon}日_{asset_label(asset)}_{model_label(model_id)}重点结果表"


def build_key_result_tables(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """Generate the 50 key result tables kept under 5.5.2."""
    if paths.key_result_tables_excel_dir is None:
        return
    for horizon in PAPER_HORIZONS:
        experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
        for asset in BEST8_ASSETS:
            for model_id in MODEL_IDS:
                output_name = key_result_table_filename(model_id, asset, horizon)
                output_path = write_csv(build_key_result_table(model_id, asset, horizon), paths.key_result_tables_excel_dir / output_name)
                baseline_run_dir = classification_run_dir(model_id, asset, horizon, "no_tradeflow4")
                extended_run_dir = classification_run_dir(model_id, asset, horizon, "with_tradeflow4")
                add_record(
                    records,
                    paths.chapter_title,
                    KEY_RESULT_TABLE_ITEM_TYPE,
                    build_key_result_table_index_title(model_id, asset, horizon),
                    output_path,
                    f"{baseline_run_dir}; {extended_run_dir}",
                    "现有结果整理",
                    note="读取同一资产、同一期限下基准模型与扩展模型的 pred_test.csv，结果对比行等于扩展模型减去基准模型。",
                )


def build_nonlinear_feature_response_table(source: pd.DataFrame, feature_name: str, max_bins: int = 5) -> pd.DataFrame:
    """Build a quantile-bin response table for one feature."""
    df = source[[feature_name, "y_score", "y_true"]].copy()
    df = df.dropna(subset=[feature_name, "y_score", "y_true"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=["特征", "特征代码", "分位组", "bin_id", "样本量", "分箱区间", "区间下界", "区间上界", "特征均值", "平均预测分数", "实际正收益率"]
        )
    bin_count = max(1, min(max_bins, df[feature_name].nunique()))
    try:
        binned = pd.qcut(df[feature_name], q=bin_count, duplicates="drop")
    except ValueError:
        binned = pd.cut(df[feature_name], bins=bin_count, duplicates="drop", include_lowest=True)
    categories = list(binned.cat.categories)
    df["bin_id"] = binned.cat.codes.astype(int)
    rows: list[dict[str, object]] = []
    for bin_id, interval in enumerate(categories):
        bin_df = df.loc[df["bin_id"] == bin_id]
        rows.append(
            {
                "特征": feature_label(feature_name),
                "特征代码": feature_name,
                "分位组": f"第{bin_id + 1}组",
                "bin_id": bin_id,
                "样本量": int(len(bin_df)),
                "分箱区间": str(interval),
                "区间下界": float(interval.left),
                "区间上界": float(interval.right),
                "特征均值": float(bin_df[feature_name].mean()),
                "平均预测分数": float(bin_df["y_score"].mean()),
                "实际正收益率": float(bin_df["y_true"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_selected_feature_table() -> pd.DataFrame:
    """读取筛选参考表，整理成 5.3 章节展示表。"""
    selected_path = PROCESSED_DIR / "screening" / "screening_reference_mainline_controls.csv"
    selected_df = pd.read_csv(selected_path)
    merged = (
        selected_df.assign(
            变量=lambda df: df["feature_name"].map(feature_label),
            是否入选=lambda df: df["selected_for_model"].map({True: "是", False: "否", "True": "是", "False": "否"}).fillna("否"),
            统一筛选重要性排序=lambda df: df["screening_importance_rank"],
            统一筛选重要性=lambda df: df["screening_importance_gain"],
            总体缺失率=lambda df: df["missing_rate_overall"],
            线性相关排序=lambda df: df["linear_rank"],
            非线性重要性排序=lambda df: df["nonlinear_rank"],
            并集排序=lambda df: df["union_rank"],
        )
        .sort_values(["selected_for_model", "union_rank", "screening_importance_rank", "feature_name"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    return merged[
        [
            "变量",
            "feature_name",
            "是否入选",
            "drop_source",
            "统一筛选重要性排序",
            "统一筛选重要性",
            "总体缺失率",
            "线性相关排序",
            "非线性重要性排序",
            "并集排序",
        ]
    ].rename(columns={"feature_name": "变量代码", "drop_source": "筛选来源"})


def build_sample_split_table() -> pd.DataFrame:
    """统计主线资产在 1 日和 22 日期限下的样本划分与正收益占比。"""
    rows: list[dict[str, object]] = []
    for asset in BEST8_ASSETS:
        feature_panel = pd.read_csv(PROCESSED_DIR / asset / "feature_panel.csv")
        for horizon in PAPER_HORIZONS:
            target_col = f"target_label_{horizon}d"
            panel = feature_panel.loc[feature_panel[target_col].notna()].copy()
            grouped = panel.groupby("split", as_index=False).agg(样本量=("Date", "count"), 正收益占比=(target_col, "mean"))
            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "资产": asset_label(asset),
                        "资产代码": asset,
                        "期限": f"{horizon}日",
                        "划分": row["split"],
                        "样本量": int(row["样本量"]),
                        "正收益占比": float(row["正收益占比"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["资产代码", "期限", "划分"]).reset_index(drop=True)


def build_target_return_trend_table(horizon: int) -> pd.DataFrame:
    """读取各资产 feature_panel.csv，生成目标变量累计收益走势的画图表。"""
    rows: list[pd.DataFrame] = []
    target_col = f"target_return_{horizon}d"
    for asset in BEST8_ASSETS:
        # feature_panel.csv 是第02步生成的单资产特征面板，里面同时包含日期、目标收益和目标方向标签。
        panel_path = PROCESSED_DIR / asset / "feature_panel.csv"
        panel = pd.read_csv(panel_path, usecols=["Date", target_col])
        panel = panel.rename(columns={"Date": "日期", target_col: "目标收益"})
        # 先转成日期并清理缺失目标值，避免未来收益窗口末尾的空值影响累计线。
        panel["日期"] = pd.to_datetime(panel["日期"])
        panel["目标收益"] = pd.to_numeric(panel["目标收益"], errors="coerce")
        panel = panel.dropna(subset=["日期", "目标收益"]).sort_values("日期").reset_index(drop=True)
        # 累计目标收益仅用于描述目标变量走势，不代表交易策略收益。
        panel["累计目标收益"] = panel["目标收益"].cumsum()
        panel["资产"] = asset_label(asset)
        panel["资产代码"] = asset
        panel["期限"] = f"{horizon}日"
        rows.append(panel[["日期", "资产", "资产代码", "期限", "目标收益", "累计目标收益"]])
    return pd.concat(rows, ignore_index=True)


def build_target_direction_distribution_table(horizon: int) -> pd.DataFrame:
    """读取各资产 feature_panel.csv，统计目标方向标签的正负样本分布。"""
    rows: list[dict[str, object]] = []
    target_col = f"target_label_{horizon}d"
    for asset in BEST8_ASSETS:
        panel_path = PROCESSED_DIR / asset / "feature_panel.csv"
        panel = pd.read_csv(panel_path, usecols=[target_col])
        label = pd.to_numeric(panel[target_col], errors="coerce").dropna()
        positive_count = int((label == 1).sum())
        negative_count = int((label == 0).sum())
        sample_count = positive_count + negative_count
        rows.append(
            {
                "资产": asset_label(asset),
                "资产代码": asset,
                "期限": f"{horizon}日",
                "样本量": sample_count,
                "正收益样本量": positive_count,
                "负收益样本量": negative_count,
                "正收益占比": positive_count / sample_count if sample_count else np.nan,
                "负收益占比": negative_count / sample_count if sample_count else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("资产代码").reset_index(drop=True)


def build_preprocessing_standardization_table() -> pd.DataFrame:
    """汇总第02步预处理记录，说明每个建模变量如何填补缺失和是否标准化。"""
    rows: list[pd.DataFrame] = []
    keep_columns = TRADEFLOW4_COLUMNS + BEST8_CONTROLS
    for asset in BEST8_ASSETS:
        stats_path = PROCESSED_DIR / asset / "screening_preprocess_stats.csv"
        stats = pd.read_csv(stats_path)
        stats = stats.loc[stats["feature_name"].isin(keep_columns)].copy()
        stats["资产"] = asset_label(asset)
        stats["资产代码"] = asset
        stats["变量"] = stats["feature_name"].map(feature_label)
        stats["是否标准化"] = stats["scaled"].map(lambda value: "是" if bool(value) else "否")
        rows.append(
            stats[
                [
                    "资产",
                    "资产代码",
                    "变量",
                    "feature_name",
                    "fill_value",
                    "train_mean",
                    "train_std",
                    "是否标准化",
                ]
            ].rename(
                columns={
                    "feature_name": "变量代码",
                    "fill_value": "缺失填补值",
                    "train_mean": "训练集均值",
                    "train_std": "训练集标准差",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_screening_training_summary() -> pd.DataFrame:
    """把 screening LGBM 的搜索与最终指标整理成摘要表。"""
    coarse_path = SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_coarse_search.csv"
    fine_path = SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_fine_search.csv"
    metrics_path = SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_metrics.csv"
    best_params_path = SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_best_params.csv"
    coarse_df = pd.read_csv(coarse_path)
    fine_df = pd.read_csv(fine_path)
    metrics_df = pd.read_csv(metrics_path)
    best_params_df = pd.read_csv(best_params_path)
    best_row = metrics_df.iloc[0]
    params_json = json.loads(best_params_df.iloc[0]["params_json"])
    return pd.DataFrame(
        [
            {"指标": "粗搜候选数", "取值": len(coarse_df)},
            {"指标": "细搜候选数", "取值": len(fine_df)},
            {"指标": "最佳阈值", "取值": best_row["decision_threshold"]},
            {"指标": "训练集AUC", "取值": best_row["train_auc"]},
            {"指标": "验证集AUC", "取值": best_row["valid_auc"]},
            {"指标": "测试集AUC", "取值": best_row["test_auc"]},
            {"指标": "训练集F1", "取值": best_row["train_f1"]},
            {"指标": "验证集F1", "取值": best_row["valid_f1"]},
            {"指标": "测试集F1", "取值": best_row["test_f1"]},
            {"指标": "最佳参数", "取值": json.dumps(params_json, ensure_ascii=False)},
        ]
    )


def load_summary_metrics(horizon: int, scheme: str) -> pd.DataFrame:
    """Load the step 10 metrics summary for one horizon and scheme."""
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    path = MAINLINE_SUMMARY_DIR / classification_summary_filename("metrics", experiment_tag, scheme=scheme)
    return pd.read_csv(path)


def load_best_valid(horizon: int, scheme: str) -> pd.DataFrame:
    """Load the step 10 best-valid summary for one horizon and scheme."""
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    path = MAINLINE_SUMMARY_DIR / classification_summary_filename("best_valid_auc", experiment_tag, scheme=scheme)
    return pd.read_csv(path)


def merge_with_no_metrics(horizon: int) -> pd.DataFrame:
    """Pair with/no-tradeflow summary rows for one horizon and compute gains."""
    with_df = load_summary_metrics(horizon, "with_tradeflow4").copy()
    no_df = load_summary_metrics(horizon, "no_tradeflow4").copy()
    merged = with_df.merge(
        no_df,
        on=["model_id", "asset_alias", "horizon"],
        how="inner",
        suffixes=("_with", "_no"),
    )
    merged["资产"] = merged["asset_alias"].map(asset_label)
    merged["模型"] = merged["model_id"].map(model_label)
    merged["期限"] = merged["horizon"].map(lambda value: f"{int(value)}日")
    merged["test_auc_gain"] = merged["test_auc_with"] - merged["test_auc_no"]
    merged["valid_auc_gain"] = merged["valid_auc_with"] - merged["valid_auc_no"]
    merged["test_f1_gain"] = merged["test_f1_with"] - merged["test_f1_no"]
    merged["valid_f1_gain"] = merged["valid_f1_with"] - merged["valid_f1_no"]
    return merged.sort_values(["test_auc_gain", "test_f1_gain", "asset_alias", "model_id"], ascending=[False, False, True, True]).reset_index(drop=True)


FORMAL_ASSET_LABELS = {
    "index": "index(指数)",
    "eg": "eg(乙二醇)",
    "bu": "bu(沥青)",
    "jm": "jm(焦煤)",
    "pp": "pp(聚丙烯)",
}
FORMAL_MODEL_SHEETS = [
    ("表5-3_决策树", "dt_cls"),
    ("表5-4_随机森林", "rf_cls"),
    ("表5-5_支持向量机", "svm_cls"),
    ("表5-6_XGBoost", "xgb_cls"),
    ("表5-7_LightGBM", "lgbm_cls"),
]
FORMAL_PAPER_METRICS = [
    ("accuracy", "准确率"),
    ("precision", "精确率"),
    ("recall", "召回率"),
    ("f1", "F1"),
    ("auc", "AUC"),
]


def formal_asset_label(asset: str) -> str:
    return FORMAL_ASSET_LABELS.get(asset, f"{asset}({asset_label(asset)})")


def formal_round(value: Any, digits: int = 3) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        return round(float(value), digits)
    return value


def clear_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_generated_files(path: Path, patterns: tuple[str, ...]) -> None:
    """只清理指定类型的生成文件，避免误删同目录其他人工文件。"""
    path.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for child in path.glob(pattern):
            if not child.is_file():
                continue
            try:
                child.unlink()
            except PermissionError:
                continue


def load_formal_ml_metrics() -> pd.DataFrame:
    frames = []
    for horizon in PAPER_HORIZONS:
        for scheme in ["no_tradeflow4", "with_tradeflow4"]:
            frame = load_summary_metrics(horizon, scheme).copy()
            frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["model_name"] = metrics["model_id"].map(model_label)
    return metrics


def load_formal_ols_metrics() -> pd.DataFrame:
    path = OUTPUTS_DIR / "summary" / "ols" / "ols_all_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    metrics["model_name"] = "OLS"
    return metrics


def formal_pairwise_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (asset, horizon, model_id), group in metrics.groupby(["asset_alias", "horizon", "model_id"], sort=True):
        if not {"no_tradeflow4", "with_tradeflow4"}.issubset(set(group["scheme"])):
            continue
        base = group.loc[group["scheme"] == "no_tradeflow4"].iloc[0]
        expanded = group.loc[group["scheme"] == "with_tradeflow4"].iloc[0]
        row = {
            "asset_alias": asset,
            "资产": formal_asset_label(asset),
            "horizon": int(horizon),
            "期限": f"{int(horizon)}日",
            "model_id": model_id,
            "模型": base.get("model_name", model_label(model_id)),
        }
        for metric, _ in FORMAL_PAPER_METRICS:
            row[f"base_{metric}"] = base[f"test_{metric}"]
            row[f"with_{metric}"] = expanded[f"test_{metric}"]
            row[f"{metric}_gain"] = expanded[f"test_{metric}"] - base[f"test_{metric}"]
        if "test_balanced_accuracy" in base and "test_balanced_accuracy" in expanded:
            row["base_balanced_accuracy"] = base["test_balanced_accuracy"]
            row["with_balanced_accuracy"] = expanded["test_balanced_accuracy"]
            row["balanced_accuracy_gain"] = expanded["test_balanced_accuracy"] - base["test_balanced_accuracy"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["auc_gain", "f1_gain", "asset_alias", "model_id"], ascending=[False, False, True, True]).reset_index(drop=True)


def formal_improvement_rows(pairwise: pd.DataFrame, horizon: int) -> pd.DataFrame:
    subset = pairwise.loc[pairwise["horizon"] == horizon].copy()
    subset = subset.sort_values(["auc_gain", "f1_gain", "asset_alias", "model_id"], ascending=[False, False, True, True])
    columns = [
        ("资产", "资产"),
        ("模型", "模型"),
        ("base_auc", "基准AUC"),
        ("with_auc", "扩展AUC"),
        ("auc_gain", "AUC提升"),
        ("base_f1", "基准F1"),
        ("with_f1", "扩展F1"),
        ("f1_gain", "F1提升"),
        ("base_accuracy", "基准准确率"),
        ("with_accuracy", "扩展准确率"),
        ("accuracy_gain", "准确率提升"),
        ("base_precision", "基准精确率"),
        ("with_precision", "扩展精确率"),
        ("precision_gain", "精确率提升"),
        ("base_recall", "基准召回率"),
        ("with_recall", "扩展召回率"),
        ("recall_gain", "召回率提升"),
        ("base_balanced_accuracy", "基准BA"),
        ("with_balanced_accuracy", "扩展BA"),
        ("balanced_accuracy_gain", "BA提升"),
    ]
    existing = [(source, target) for source, target in columns if source in subset.columns]
    out = subset[[source for source, _ in existing]].rename(columns={source: target for source, target in existing})
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: formal_round(value, 6))
    return out


def write_formal_improvement_summary(pairwise: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        formal_improvement_rows(pairwise, 1).to_excel(writer, index=False, sheet_name="1日")
        formal_improvement_rows(pairwise, 22).to_excel(writer, index=False, sheet_name="22日")
    return path


def formal_metric_values(metrics: pd.DataFrame, asset: str, model_id: str, horizon: int, scheme: str) -> list[Any]:
    row = metrics.loc[
        (metrics["asset_alias"] == asset)
        & (metrics["model_id"] == model_id)
        & (metrics["horizon"] == horizon)
        & (metrics["scheme"] == scheme)
    ]
    if row.empty:
        return [None for _ in FORMAL_PAPER_METRICS]
    row = row.iloc[0]
    return [formal_round(row[f"test_{metric}"]) for metric, _ in FORMAL_PAPER_METRICS]


def formal_diff_values(base: list[Any], expanded: list[Any]) -> list[Any]:
    values = []
    for base_value, expanded_value in zip(base, expanded):
        if base_value is None or expanded_value is None:
            values.append(None)
        else:
            values.append(formal_round(float(expanded_value) - float(base_value)))
    return values


def write_formal_asset_paper_table(metrics: pd.DataFrame, asset: str, output_dir: Path) -> Path:
    path = output_dir / f"{formal_asset_label(asset)}_论文对应表格.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, model_id in FORMAL_MODEL_SHEETS:
            rows: list[list[Any]] = []
            for horizon, horizon_label in [(1, "向前1日"), (22, "向前1月")]:
                base = formal_metric_values(metrics, asset, model_id, horizon, "no_tradeflow4")
                expanded = formal_metric_values(metrics, asset, model_id, horizon, "with_tradeflow4")
                rows.extend(
                    [
                        [horizon_label] + [label for _, label in FORMAL_PAPER_METRICS],
                        ["基准模型"] + base,
                        ["扩展模型"] + expanded,
                        ["结果对比"] + formal_diff_values(base, expanded),
                    ]
                )
            pd.DataFrame(rows).to_excel(writer, index=False, header=False, sheet_name=excel_sheet_name(sheet_name))
    return path


def formal_auc_share_rows(pairwise: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    subset = pairwise.copy()
    if horizon is not None:
        subset = subset.loc[subset["horizon"] == horizon].copy()
    rows = []
    for asset in PAPER_ASSETS:
        part = subset.loc[subset["asset_alias"] == asset]
        if part.empty:
            continue
        rows.append(
            {
                "资产": formal_asset_label(asset),
                "组合数": int(len(part)),
                "AUC提升数": int((part["auc_gain"] > 1e-6).sum()),
                "AUC提升占比": formal_round(float((part["auc_gain"] > 1e-6).mean()), 6),
                "平均AUC提升": formal_round(float(part["auc_gain"].mean()), 6),
                "平均F1提升": formal_round(float(part["f1_gain"].mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["AUC提升占比", "平均AUC提升", "平均F1提升"], ascending=[False, False, False])


def write_formal_auc_share(pairwise: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        formal_auc_share_rows(pairwise).to_excel(writer, index=False, sheet_name="总体")
        formal_auc_share_rows(pairwise, 1).to_excel(writer, index=False, sheet_name="1日")
        formal_auc_share_rows(pairwise, 22).to_excel(writer, index=False, sheet_name="22日")
    return path


def formal_comparison_model_order(metrics: pd.DataFrame) -> list[str]:
    rows = []
    base_order = ["ols_lpm"] + MODEL_IDS
    for model_id in base_order:
        model_rows = metrics.loc[metrics["model_id"] == model_id]
        auc_gains = []
        f1_gains = []
        auc_values = []
        f1_values = []
        group_cols = ["asset_alias", "horizon"]
        for _, part in model_rows.groupby(group_cols, sort=True):
            base = part.loc[part["scheme"] == "no_tradeflow4"]
            expanded = part.loc[part["scheme"] == "with_tradeflow4"]
            if not expanded.empty:
                auc_values.extend(pd.to_numeric(expanded["test_auc"], errors="coerce").dropna().astype(float).tolist())
                f1_values.extend(pd.to_numeric(expanded["test_f1"], errors="coerce").dropna().astype(float).tolist())
            if not base.empty and not expanded.empty:
                auc_gains.append(float(expanded.iloc[0]["test_auc"] - base.iloc[0]["test_auc"]))
                f1_gains.append(float(expanded.iloc[0]["test_f1"] - base.iloc[0]["test_f1"]))
        rows.append(
            {
                "model_id": model_id,
                "auc_gain": float(np.mean(auc_gains)) if auc_gains else np.nan,
                "f1_gain": float(np.mean(f1_gains)) if f1_gains else np.nan,
                "auc": float(np.mean(auc_values)) if auc_values else np.nan,
                "f1": float(np.mean(f1_values)) if f1_values else np.nan,
                "base_order": base_order.index(model_id),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["auc_gain", "f1_gain", "auc", "f1", "base_order"], ascending=[False, False, False, False, True])
        ["model_id"]
        .tolist()
    )


def formal_comparison_metric_table(metrics: pd.DataFrame, scheme: str, model_order: list[str]) -> pd.DataFrame:
    labels = {"ols_lpm": "OLS", **MODEL_LABELS}
    rows = []
    for horizon, horizon_label in [(1, "向前1日"), (22, "向前1月")]:
        for metric, metric_label in FORMAL_PAPER_METRICS:
            row: dict[str, Any] = {"期限": horizon_label, "指标": metric_label}
            for model_id in model_order:
                values = metrics.loc[
                    (metrics["model_id"] == model_id)
                    & (metrics["horizon"] == horizon)
                    & (metrics["scheme"] == scheme)
                ]
                if values.empty:
                    row[labels[model_id]] = None
                else:
                    row[labels[model_id]] = formal_round(pd.to_numeric(values[f"test_{metric}"], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def formal_comparison_gain_table(metrics: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    labels = {"ols_lpm": "OLS", **MODEL_LABELS}
    rows = []
    for horizon, horizon_label in [(1, "向前1日"), (22, "向前1月")]:
        for metric, metric_label in FORMAL_PAPER_METRICS:
            row: dict[str, Any] = {"期限": horizon_label, "指标": metric_label}
            for model_id in model_order:
                diffs = []
                model_horizon = metrics.loc[(metrics["model_id"] == model_id) & (metrics["horizon"] == horizon)]
                for _, part in model_horizon.groupby("asset_alias", sort=True):
                    base = part.loc[part["scheme"] == "no_tradeflow4"]
                    expanded = part.loc[part["scheme"] == "with_tradeflow4"]
                    if base.empty or expanded.empty:
                        continue
                    diffs.append(float(expanded.iloc[0][f"test_{metric}"] - base.iloc[0][f"test_{metric}"]))
                if not diffs:
                    row[labels[model_id]] = None
                else:
                    row[labels[model_id]] = formal_round(float(np.mean(diffs)))
            rows.append(row)
    return pd.DataFrame(rows)


def write_blocked_comparison_sheet(writer: pd.ExcelWriter, sheet_name: str, metrics: pd.DataFrame) -> None:
    model_order = formal_comparison_model_order(metrics)
    blocks = [
        ("扩展模型指标", formal_comparison_metric_table(metrics, "with_tradeflow4", model_order)),
        ("交易流提升", formal_comparison_gain_table(metrics, model_order)),
        ("基准模型指标", formal_comparison_metric_table(metrics, "no_tradeflow4", model_order)),
    ]
    startrow = 0
    for title, table in blocks:
        pd.DataFrame([[title]]).to_excel(writer, index=False, header=False, sheet_name=sheet_name, startrow=startrow)
        table.to_excel(writer, index=False, sheet_name=sheet_name, startrow=startrow + 1)
        startrow += len(table) + 4


def write_all_asset_ols_ml_comparison(ml_metrics: pd.DataFrame, ols_metrics: pd.DataFrame, output_dir: Path) -> Path:
    combined = pd.concat([ols_metrics.copy(), ml_metrics.copy()], ignore_index=True)
    excel_path = output_dir / "各资产_OLS与算法指标对比表.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for asset in PAPER_ASSETS:
            asset_metrics = combined.loc[combined["asset_alias"] == asset].copy()
            write_blocked_comparison_sheet(writer, excel_sheet_name(formal_asset_label(asset)), asset_metrics)
        write_blocked_comparison_sheet(writer, "总体", combined)
    return excel_path


def write_algorithm_details(output_dir: Path) -> Path:
    path = output_dir / "算法流程设计细节.txt"
    lines = [
        "正式算法流程设计细节",
        "",
        "一、机器学习分类模型",
        "1. 训练范围为5个资产、2个期限、2种方案、5类模型，共100组结果。",
        "2. 超参数搜索使用 search space v2：扩大决策树、随机森林、SVM、XGBoost、LightGBM 的粗搜、细搜和 fallback 空间。",
        "3. 候选模型只使用训练集和验证集筛选；测试集只用于最终评估。",
        "4. 候选硬过滤要求训练集和验证集 y_score 非常数，验证集阈值后两类标签均出现，预测正类比例位于非退化区间，每类预测样本不少于5个。",
        "5. 树模型额外排除单叶节点、无有效分裂和特征重要性全0的候选。",
        "6. 候选排序以验证集 AUC 为第一标准，其次为验证集 BA、KS、F1，训练验证 AUC 差距只作为后置辅助项。",
        "7. 正式硬分类阈值只在验证集上选择，候选来自验证集 y_score 的0.1%到99.9%分位点、验证集唯一分数和0.5。",
        "8. 指数 index 使用 valid_f1_prev015：优先最大化验证集F1，同时约束验证集预测正类比例在0.10到0.90，且与验证集真实正类比例差距不超过0.15。",
        "9. 非指数资产 eg、bu、jm、pp 使用 valid_f1_cap075：优先最大化验证集F1，同时约束验证集预测正类比例在0.25到0.75。",
        "",
        "二、OLS线性基准",
        "1. OLS采用线性概率模型，y_score 是连续预测分数，不解释为严格概率。",
        "2. OLS阈值策略为 valid_target_pos_040_nondegenerate：验证集预测正类比例在0.20到0.80，两类预测样本均不少于5个。",
        "3. OLS阈值优先选择验证集预测正类比例最接近0.40的候选，其次比较验证集BA和F1。",
        "4. OLS测试集不参与阈值选择；正式审计要求指数OLS结果不出现概率常数、全1、全0或近退化。",
        "",
        "三、论文展示表",
        "1. 方案对比均定义为扩展模型减去基准模型。",
        "2. 所有提升汇总按AUC提升降序、F1提升降序排序；横向模型列也按对应范围内的平均AUC提升、平均F1提升排序。",
        "3. 展示表保留准确率、精确率、召回率、F1、AUC和必要BA，不展示阈值和预测正类比例。",
        "4. 诊断字段保留在训练、汇总和OLS审计文件中，不进入论文主展示表。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_formal_paper_figures(records: list[dict[str, str]], output_dir: Path) -> list[Path]:
    """把章节图片集中复制到正式论文结果目录，方便替换论文图片。"""
    figure_dir = output_dir / "图片"
    clear_generated_files(figure_dir, ("*.png",))
    copied: list[Path] = []
    seen_names: set[str] = set()
    for record in records:
        if record.get("item_type") != "figure":
            continue
        source_path = Path(str(record.get("path", "")))
        if not source_path.exists():
            continue
        title = str(record.get("title", source_path.stem))
        filename = f"{excel_sheet_name(title)}.png"
        if filename in seen_names:
            filename = f"{len(seen_names) + 1:02d}_{filename}"
        seen_names.add(filename)
        target_path = figure_dir / filename
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
    return copied


def write_formal_paper_outputs(records: list[dict[str, str]], output_dir: Path, register_records: bool = True) -> list[Path]:
    clear_output_dir(output_dir)
    ml_metrics = load_formal_ml_metrics()
    ols_metrics = load_formal_ols_metrics()
    ml_pairwise = formal_pairwise_from_metrics(ml_metrics)
    ols_pairwise = formal_pairwise_from_metrics(ols_metrics)
    paths = [
        write_formal_improvement_summary(ml_pairwise, output_dir, "指标提升汇总_按期限.xlsx"),
        write_formal_auc_share(ml_pairwise, output_dir, "不同资产AUC提升占比统计.xlsx"),
        write_formal_improvement_summary(ols_pairwise, output_dir, "OLS指标提升汇总_按期限.xlsx"),
        write_formal_auc_share(ols_pairwise, output_dir, "OLS不同资产AUC提升占比统计.xlsx"),
    ]
    paths.extend(write_formal_asset_paper_table(ml_metrics, asset, output_dir) for asset in PAPER_ASSETS)
    paths.extend([write_all_asset_ols_ml_comparison(ml_metrics, ols_metrics, output_dir), write_algorithm_details(output_dir)])
    if register_records:
        for path in paths:
            item_type = "figure" if path.suffix.lower() == ".png" else ("text" if path.suffix.lower() == ".txt" else "table_xlsx")
            add_record(
                records,
                "5.5.2 基准模型对比",
                item_type,
                path.stem,
                path,
                str(MAINLINE_SUMMARY_DIR),
                "现有结果整理",
            )
    return paths


def build_classification_param_index() -> pd.DataFrame:
    """汇总全部分类 run 的特征数、阈值和最佳参数。"""
    rows: list[dict[str, object]] = []
    for horizon in PAPER_HORIZONS:
        experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
        for asset in BEST8_ASSETS:
            for model_id in MODEL_IDS:
                for scheme in ["with_tradeflow4", "no_tradeflow4"]:
                    run_name = tagged_run_name(asset, horizon, scheme, experiment_tag)
                    metrics_path = MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name / "metrics.csv"
                    params_path = MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name / "best_params.csv"
                    if not metrics_path.exists() or not params_path.exists():
                        continue
                    metrics_row = pd.read_csv(metrics_path).iloc[0]
                    params_row = pd.read_csv(params_path).iloc[0]
                    rows.append(
                        {
                            "资产": asset_label(asset),
                            "资产代码": asset,
                            "期限": f"{horizon}日",
                            "方案": SCHEME_LABELS[scheme],
                            "模型": model_label(model_id),
                            "模型代码": model_id,
                            "特征数": int(metrics_row["feature_count"]),
                            "参数JSON": params_row["params_json"],
                        }
                    )
    return pd.DataFrame(rows).sort_values(["期限", "资产代码", "模型代码", "方案"]).reset_index(drop=True)


def build_best_model_summary() -> pd.DataFrame:
    """整理各资产各期限在 with_tradeflow4 下的最佳模型汇总表。"""
    frames = []
    for horizon in PAPER_HORIZONS:
        best_df = load_best_valid(horizon, "with_tradeflow4").copy()
        best_df["期限"] = f"{horizon}日"
        frames.append(best_df)
    merged = pd.concat(frames, ignore_index=True)
    merged["资产"] = merged["asset_alias"].map(asset_label)
    merged["模型"] = merged["model_id"].map(model_label)
    return merged[
        [
            "资产",
            "asset_alias",
            "期限",
            "模型",
            "model_id",
            "valid_auc",
            "test_auc",
            "valid_f1",
            "test_f1",
            "decision_threshold",
            "feature_count",
        ]
    ].rename(
        columns={
            "asset_alias": "资产代码",
            "model_id": "模型代码",
            "valid_auc": "验证集AUC",
            "test_auc": "测试集AUC",
            "valid_f1": "验证集F1",
            "test_f1": "测试集F1",
            "decision_threshold": "决策阈值",
            "feature_count": "特征数",
        }
    )


# ===== 第四组：绘图函数 =====
# 这一组函数只负责把前面整理好的表格数据画成结果图片。
# 数据准备和图片渲染分开写，是为了在修改图表样式时不需要反推数据口径。

def plot_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    value_fmt: str = "{:.3f}",
    rotation: int = 0,
    zero_line: bool = False,
    orientation: str = "auto",
) -> Path:
    """绘制单序列柱状图。"""
    labels = df[label_col].tolist()
    values = df[value_col].tolist()
    bar_count = len(values)

    if orientation not in {"auto", "vertical", "horizontal"}:
        raise ValueError(f"orientation must be auto/vertical/horizontal, got: {orientation}")
    if orientation == "auto":
        # 标签数量较多时，竖向柱状图会出现“坐标文字重叠/数值标注挤在一起”，自动切换为水平柱状图。
        orientation = "horizontal" if bar_count >= 14 else "vertical"

    if orientation == "horizontal":
        fig_height = max(4.2, 0.32 * bar_count + 1.4)
        fig, ax = plt.subplots(figsize=(9.6, fig_height))
        bars = ax.barh(labels, values, color=gray_colors(bar_count), edgecolor="black", linewidth=0.8)
        # 水平柱状图：横轴是数值，纵轴是变量名称。
        style_axis(ax, title=title, xlabel=ylabel)
        if zero_line:
            ax.axvline(0, color="0.35", linewidth=0.9)
        ax.tick_params(axis="y", labelsize=9)
        ax.invert_yaxis()
        annotate_bars_horizontal(ax, bars, fmt=value_fmt)
        return save_figure(fig, output_path)

    # vertical
    fig_width = max(8.2, 0.45 * bar_count + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(labels, values, color=gray_colors(bar_count), edgecolor="black", linewidth=0.8)
    style_axis(ax, title=title, ylabel=ylabel)
    if zero_line:
        ax.axhline(0, color="0.35", linewidth=0.9)
    ax.tick_params(axis="x", rotation=rotation)
    annotate_bars(ax, bars, fmt=value_fmt)
    return save_figure(fig, output_path)


def plot_grouped_bar(
    df: pd.DataFrame,
    category_col: str,
    series_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    series_order: list[str] | None = None,
    value_fmt: str = "{:.3f}",
) -> Path:
    """绘制分组柱状图。"""
    plot_df = df.copy()
    categories = plot_df[category_col].drop_duplicates().tolist()
    if series_order is None:
        series_order = plot_df[series_col].drop_duplicates().tolist()
    x = np.arange(len(categories))
    width = 0.8 / max(len(series_order), 1)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for idx, series_name in enumerate(series_order):
        series_df = plot_df.loc[plot_df[series_col] == series_name].set_index(category_col).reindex(categories).reset_index()
        offsets = x - 0.4 + width / 2 + idx * width
        bars = ax.bar(
            offsets,
            series_df[value_col].tolist(),
            width=width,
            color=gray_colors(len(series_order))[idx],
            edgecolor="black",
            linewidth=0.8,
            label=series_name,
        )
        annotate_bars(ax, bars, fmt=value_fmt, fontsize=8, rotation=0)
    ax.set_xticks(x, categories)
    # 分组柱状图保留标题，便于单独打开图片时直接识别图意。
    style_axis(ax, title=title, ylabel=ylabel)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, output_path)


def plot_heatmap(heatmap_df: pd.DataFrame, output_path: Path, title: str) -> Path:
    """绘制带数值标注的灰阶热力图。"""
    pivot = heatmap_df.set_index("交易流变量")
    n_rows, n_cols = pivot.shape
    fig_width = max(14.0, n_cols * 0.52 + 2.6)
    fig_height = max(4.4, n_rows * 0.52 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap_values = pivot.to_numpy(dtype=float)
    heatmap = ax.imshow(heatmap_values, cmap=LIGHT_GRAYS_CMAP, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=9)
    style_axis(ax, title=title, enable_y_grid=False)
    annotate_heatmap(ax, heatmap_values, fmt="{:.2f}", fontsize=7)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("皮尔逊相关系数", rotation=90, labelpad=18, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return save_figure(fig, output_path)


def plot_state_sample_distribution(plot_df: pd.DataFrame, output_path: Path) -> Path:
    """绘制不同市场状态下的测试样本量分布图。"""
    return plot_grouped_bar(
        plot_df,
        category_col="状态",
        series_col="期限",
        value_col="样本量",
        output_path=output_path,
        title="各状态测试样本量分布",
        ylabel="样本量",
        series_order=["1日", "22日"],
        value_fmt="{:.0f}",
    )


def plot_target_return_trend(plot_df: pd.DataFrame, output_path: Path, title: str) -> Path:
    """绘制目标变量累计收益走势，每条线对应一个预测对象。"""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for asset_name, asset_df in plot_df.groupby("资产", sort=False):
        ordered = asset_df.sort_values("日期")
        ax.plot(ordered["日期"], ordered["累计目标收益"], linewidth=1.6, label=asset_name)
    ax.legend(frameon=False, ncol=3, loc="best")
    style_axis(ax, title=title, xlabel="日期", ylabel="累计目标收益")
    return save_figure(fig, output_path)


def plot_target_direction_distribution(plot_df: pd.DataFrame, output_path: Path, title: str) -> Path:
    """绘制目标方向标签中正收益样本占比，用于检查类别不平衡。"""
    return plot_bar(
        plot_df,
        label_col="资产",
        value_col="正收益占比",
        output_path=output_path,
        title=title,
        ylabel="正收益占比",
        value_fmt="{:.2f}",
        zero_line=False,
    )


def choose_representative_tree_model(horizon: int) -> str:
    """按测试集平均 AUC 选出指定期限的代表性树模型。"""
    with_df = load_summary_metrics(horizon, "with_tradeflow4")
    summary = with_df.loc[with_df["model_id"].isin(TREE_MODEL_IDS)].groupby("model_id", as_index=False)["test_auc"].mean()
    return summary.sort_values(["test_auc", "model_id"], ascending=[False, True]).iloc[0]["model_id"]


def aggregate_feature_importance(model_id: str, horizon: int, scheme: str) -> pd.DataFrame:
    """把主线资产的特征重要性按模型和方案做平均。"""
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    frames = []
    for asset in BEST8_ASSETS:
        run_name = tagged_run_name(asset, horizon, scheme, experiment_tag)
        path = MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name / "feature_importance.csv"
        df = pd.read_csv(path)
        df["asset_alias"] = asset
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    agg = (
        merged.groupby("feature_name", as_index=False)
        .agg(平均重要性=("importance_gain", "mean"))
        .sort_values(["平均重要性", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    agg["变量"] = agg["feature_name"].map(feature_label)
    agg["方案"] = SCHEME_LABELS[scheme]
    return agg


def build_representative_importance_table(horizon: int, model_id: str) -> pd.DataFrame:
    """生成 5.6 章节代表性树模型的重要性对照表。"""
    with_df = aggregate_feature_importance(model_id, horizon, "with_tradeflow4")
    no_df = aggregate_feature_importance(model_id, horizon, "no_tradeflow4")
    merged = with_df.merge(no_df, on="feature_name", how="outer", suffixes=("_with", "_no")).fillna(0.0)
    merged["变量"] = merged["feature_name"].map(feature_label)
    merged["最大平均重要性"] = merged[["平均重要性_with", "平均重要性_no"]].max(axis=1)
    out = (
        merged.sort_values(["最大平均重要性", "feature_name"], ascending=[False, True])
        .head(12)
        .reset_index(drop=True)[["变量", "feature_name", "平均重要性_with", "平均重要性_no"]]
        .rename(
            columns={
                "feature_name": "变量代码",
                "平均重要性_with": "加入交易流平均重要性",
                "平均重要性_no": "不加入交易流平均重要性",
            }
        )
    )
    return out


def plot_importance_comparison(df: pd.DataFrame, output_path: Path, title: str) -> Path:
    """绘制加入交易流前后的特征重要性对照图。"""
    plot_df = df.copy()
    categories = plot_df["变量"].tolist()
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    bars_with = ax.bar(
        x - width / 2,
        plot_df["加入交易流平均重要性"],
        width=width,
        color="0.35",
        edgecolor="black",
        linewidth=0.8,
        label="加入交易流",
    )
    bars_no = ax.bar(
        x + width / 2,
        plot_df["不加入交易流平均重要性"],
        width=width,
        color="0.75",
        edgecolor="black",
        linewidth=0.8,
        label="不加入交易流",
    )
    style_axis(ax, title=title, ylabel="平均重要性")
    ax.set_xticks(x, categories, rotation=50, ha="right")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    annotate_bars(ax, bars_with, "{:.3f}", fontsize=8)
    annotate_bars(ax, bars_no, "{:.3f}", fontsize=8)
    return save_figure(fig, output_path)


def select_nonlinear_features(model_id: str, horizon: int) -> list[str]:
    """选出代表性模型中用于非线性响应分析的特征。"""
    with_df = aggregate_feature_importance(model_id, horizon, "with_tradeflow4")
    non_tradeflow = [name for name in with_df["feature_name"].tolist() if name not in TRADEFLOW4_COLUMNS]
    tradeflow = [name for name in with_df["feature_name"].tolist() if name in TRADEFLOW4_COLUMNS]
    chosen = non_tradeflow[:2]
    if tradeflow:
        chosen.append(tradeflow[0])
    return chosen


def load_representative_prediction_features(model_id: str, horizon: int) -> pd.DataFrame:
    """把代表性模型测试预测与 prepared 特征拼起来，供响应分析复用。"""
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    frames = []
    for asset in BEST8_ASSETS:
        run_name = tagged_run_name(asset, horizon, "with_tradeflow4", experiment_tag)
        pred_path = MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name / "pred_test.csv"
        prepared_path = PROCESSED_DIR / asset / "horizons" / f"{horizon}d" / f"with_tradeflow4__{experiment_tag}" / "test_prepared.csv"
        pred_df = pd.read_csv(pred_path, parse_dates=["Date"])
        prepared_df = pd.read_csv(prepared_path, parse_dates=["Date"])
        merged = pred_df.merge(prepared_df, on="Date", how="inner")
        merged["asset_alias"] = asset
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def plot_nonlinear_feature_response(
    plot_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    """绘制特征分位组下的预测分数与实际正收益率对照图。"""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(plot_df["分位组"], plot_df["平均预测分数"], color="0.25", marker="o", linewidth=1.8, label="平均预测分数")
    ax.plot(plot_df["分位组"], plot_df["实际正收益率"], color="0.65", marker="s", linewidth=1.8, label="实际正收益率")
    style_axis(ax, title=title, ylabel="比率/分数", xlabel="分位组")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.tick_params(axis="x", rotation=0)
    return save_figure(fig, output_path)


# ===== 第五组：状态划分与稳健性统计 =====
# 这一组函数围绕 5.7 章节工作：先识别市场状态，再计算状态内样本量、AUC 和交易流增益。

def build_state_lookup(horizon: int) -> pd.DataFrame:
    """按动量和波动率把测试样本划成四类市场状态。"""
    rows = []
    target_col = f"target_label_{horizon}d"
    for asset in BEST8_ASSETS:
        feature_panel = pd.read_csv(PROCESSED_DIR / asset / "feature_panel.csv", parse_dates=["Date"])
        panel = feature_panel.loc[(feature_panel["split"] == "test") & feature_panel[target_col].notna(), ["Date", "asset_alias", "mom_h22d", "vol_h22d"]].copy()
        median_vol = panel["vol_h22d"].median()
        panel["趋势状态"] = np.where(panel["mom_h22d"] > 0, "上涨", "下跌")
        panel["波动状态"] = np.where(panel["vol_h22d"] > median_vol, "高波动", "低波动")
        panel["state"] = panel["趋势状态"] + panel["波动状态"]
        rows.append(panel)
    return pd.concat(rows, ignore_index=True)


def build_state_sample_counts() -> pd.DataFrame:
    """统计四类市场状态在不同期限下的测试样本量。"""
    rows = []
    for horizon in PAPER_HORIZONS:
        state_lookup = build_state_lookup(horizon)
        grouped = state_lookup.groupby("state", as_index=False).agg(样本量=("Date", "count"))
        for _, row in grouped.iterrows():
            rows.append({"horizon": horizon, "state": row["state"], "样本量": int(row["样本量"])})
    out = pd.DataFrame(rows)
    out["state"] = pd.Categorical(out["state"], categories=STATE_ORDER, ordered=True)
    return out.sort_values(["state", "horizon"]).reset_index(drop=True)


def build_state_sample_distribution_table(sample_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert state sample counts to the figure table used in 5.7.1."""
    plot_df = sample_counts.rename(columns={"state": "状态"}).copy()
    plot_df["期限"] = plot_df["horizon"].map(lambda value: f"{int(value)}日")
    plot_df["状态"] = pd.Categorical(plot_df["状态"], categories=STATE_ORDER, ordered=True)
    return plot_df[["状态", "期限", "样本量"]].sort_values(["状态", "期限"]).reset_index(drop=True)


def build_state_gain_plot_table(robustness: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Convert the robustness summary to the figure table used in 5.7.2."""
    plot_df = robustness.loc[robustness["horizon"] == horizon, ["state", "平均AUC增益"]].rename(
        columns={"state": "状态", "平均AUC增益": "状态平均AUC增益"}
    )
    plot_df["状态"] = pd.Categorical(plot_df["状态"], categories=STATE_ORDER, ordered=True)
    return plot_df.sort_values("状态").reset_index(drop=True)


def build_tradeflow_corr_heatmap_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the long correlation file to the heatmap table used in 5.3.2."""
    pivot = corr_df.pivot(index="tradeflow_feature", columns="candidate_feature", values="pearson_corr").reset_index()
    pivot = pivot.rename(columns={"tradeflow_feature": "交易流变量"})
    pivot["交易流变量"] = pivot["交易流变量"].map(feature_label)
    rename_map = {column: feature_label(column) for column in pivot.columns if column != "交易流变量"}
    return pivot.rename(columns=rename_map)


def _safe_pearson_corr(left: pd.Series, right: pd.Series) -> float:
    """在变量近似常数时返回0，保证第11步可从主线processed数据重建相关性诊断。"""
    left_std = float(left.std())
    right_std = float(right.std())
    if np.isclose(left_std, 0.0) or np.isclose(right_std, 0.0):
        return 0.0
    corr_value = left.corr(right)
    if pd.isna(corr_value):
        return 0.0
    return float(corr_value)


def load_or_build_tradeflow_corr() -> tuple[pd.DataFrame, str]:
    """读取第04步相关性表；若被清理，则用processed正式数据即时重建。"""
    corr_path = SCREENING_OUTPUT_DIR / "diagnostics" / "screening_trade_flow_corr.csv"
    if corr_path.exists():
        return pd.read_csv(corr_path), str(corr_path)

    pooled_ready_parts = []
    for asset in SCREENING_ASSETS:
        ready_path = asset_processed_dir(asset) / "screening_ready_panel.csv"
        if not ready_path.exists():
            continue
        pooled_ready_parts.append(pd.read_csv(ready_path, parse_dates=["Date"]))
    if not pooled_ready_parts:
        raise FileNotFoundError(corr_path)

    pooled_ready = pd.concat(pooled_ready_parts, ignore_index=True)
    pooled_train = pooled_ready.loc[pooled_ready["split"] == "train"].copy()
    rows = []
    for fixed_feature in FIXED_TRADEFLOW_COLUMNS:
        for candidate in SCREENING_CANDIDATE_COLUMNS:
            corr_value = _safe_pearson_corr(pooled_train[fixed_feature], pooled_train[candidate])
            rows.append(
                {
                    "tradeflow_feature": fixed_feature,
                    "candidate_feature": candidate,
                    "pearson_corr": corr_value,
                    "abs_pearson_corr": abs(corr_value),
                }
            )
    corr_df = pd.DataFrame(rows)
    return corr_df, "data/processed/*/screening_ready_panel.csv"


def compute_state_metrics(horizon: int) -> pd.DataFrame:
    """计算各 模型×资产×方案×状态 的状态内测试 AUC。"""
    state_lookup = build_state_lookup(horizon)
    rows = []
    experiment_tag = HORIZON_TO_EXPERIMENT_TAG[horizon]
    for asset in BEST8_ASSETS:
        asset_states = state_lookup.loc[state_lookup["asset_alias"] == asset]
        for model_id in MODEL_IDS:
            for scheme in ["with_tradeflow4", "no_tradeflow4"]:
                run_name = tagged_run_name(asset, horizon, scheme, experiment_tag)
                pred_path = MAINLINE_CLASSIFICATION_OUTPUT_DIR / model_id / run_name / "pred_test.csv"
                pred_df = pd.read_csv(pred_path, parse_dates=["Date"])
                merged = pred_df.merge(asset_states[["Date", "state"]], on="Date", how="left")
                for state in STATE_ORDER:
                    state_df = merged.loc[merged["state"] == state].copy()
                    sample_count = len(state_df)
                    positive_count = int(state_df["y_true"].sum()) if sample_count else 0
                    negative_count = int(sample_count - positive_count)
                    auc_value = np.nan
                    if sample_count > 0 and state_df["y_true"].nunique() == 2:
                        auc_value = float(roc_auc_score(state_df["y_true"], state_df["y_score"]))
                    rows.append(
                        {
                            "asset_alias": asset,
                            "model_id": model_id,
                            "horizon": horizon,
                            "scheme": scheme,
                            "state": state,
                            "sample_count": sample_count,
                            "positive_count": positive_count,
                            "negative_count": negative_count,
                            "state_test_auc": auc_value,
                        }
                    )
    return pd.DataFrame(rows)


def build_state_robustness_summary() -> pd.DataFrame:
    """汇总不同市场状态下的可计算组合数、正增益组合数和平均 AUC 增益。"""
    pair_frames = []
    for horizon in PAPER_HORIZONS:
        metrics_df = compute_state_metrics(horizon)
        with_df = metrics_df.loc[metrics_df["scheme"] == "with_tradeflow4"].rename(
            columns={
                "sample_count": "sample_count_with",
                "positive_count": "positive_count_with",
                "negative_count": "negative_count_with",
                "state_test_auc": "state_test_auc_with",
            }
        )
        no_df = metrics_df.loc[metrics_df["scheme"] == "no_tradeflow4"].rename(
            columns={
                "sample_count": "sample_count_no",
                "positive_count": "positive_count_no",
                "negative_count": "negative_count_no",
                "state_test_auc": "state_test_auc_no",
            }
        )
        paired = with_df.merge(
            no_df,
            on=["asset_alias", "model_id", "horizon", "state"],
            how="outer",
        )
        paired["状态内AUC可计算"] = paired["state_test_auc_with"].notna() & paired["state_test_auc_no"].notna()
        paired["状态内AUC增益"] = paired["state_test_auc_with"] - paired["state_test_auc_no"]
        paired["正增益组合"] = paired["状态内AUC增益"] > 0
        pair_frames.append(paired)
    all_pairs = pd.concat(pair_frames, ignore_index=True)
    summary = (
        all_pairs.groupby(["horizon", "state"], as_index=False)
        .agg(
            total_combo_count=("model_id", "count"),
            可计算组合数=("状态内AUC可计算", "sum"),
            正增益组合数=("正增益组合", "sum"),
            平均AUC增益=("状态内AUC增益", "mean"),
        )
        .rename(columns={"total_combo_count": "总组合数"})
    )
    summary["不可计算组合数"] = summary["总组合数"] - summary["可计算组合数"]
    summary["期限"] = summary["horizon"].map(lambda value: f"{int(value)}日")
    summary["状态"] = pd.Categorical(summary["state"], categories=STATE_ORDER, ordered=True)
    return summary.sort_values(["horizon", "状态"]).reset_index(drop=True)


# ===== 第六组：结果索引与章节装配 =====
# 这一组函数是第 11 步的“收口层”。
# 它们把各章图片、Excel 和静态正文统一写入 outputs/visualizations，
# 并生成后续第 12 步会读取的章节结果目录表和结果索引。


def write_result_index(records: list[dict[str, str]]) -> dict[str, Path]:
    """Write the machine-readable csv and a navigation-only markdown index."""
    result_index_dir = VISUALIZATION_DIR / RESULT_INDEX_DIRNAME
    result_index_dir.mkdir(parents=True, exist_ok=True)
    result_index_df = pd.DataFrame(records)
    for column in ["related_title", "related_path", "excel_sheet"]:
        if column not in result_index_df.columns:
            result_index_df[column] = ""
        result_index_df[column] = result_index_df[column].fillna("")
    result_index_df = result_index_df[["chapter", "item_type", "title", "path", "related_title", "related_path", "excel_sheet"]]
    csv_path = result_index_dir / "章节结果目录表.csv"
    write_csv(result_index_df, csv_path)

    def chapter_folder(chapter_title: str) -> str:
        return CHAPTER_SPECS.get(chapter_title, "")

    def unique_names(rows: pd.DataFrame, item_types: set[str]) -> list[str]:
        subset = rows.loc[rows["item_type"].isin(item_types), "path"].dropna().astype(str).tolist()
        return sorted(dict.fromkeys(Path(path_text).name for path_text in subset))

    def join_names(names: list[str]) -> str:
        return "`" + "`、`".join(names) + "`" if names else "无"

    lines = ["# 章节结果索引", "", "完整目录见 `章节结果目录表.csv`。", "本目录按当前论文第五章结构组织。", ""]
    for chapter_title in CHAPTER_SPECS.keys():
        lines.append(f"## {chapter_title}")
        chapter_rows = result_index_df.loc[result_index_df["chapter"] == chapter_title]
        if chapter_rows.empty:
            lines.append("- 章节目录：无")
            lines.append("")
            continue
        folder_name = chapter_folder(chapter_title)
        if folder_name:
            lines.append(f"- 章节目录：`outputs/visualizations/{Path(folder_name).as_posix()}/`")
        lines.append("- 图片目录：`figures/`")
        lines.append("- 表格目录：`tables_excel/`")
        lines.append(f"- 图片文件：{join_names(unique_names(chapter_rows, {'figure'}))}")
        lines.append(f"- 表格文件：{join_names(unique_names(chapter_rows, {'table_xlsx', 'figure_result_excel_sheet'}))}")
        lines.append(f"- 正文文件：{join_names(unique_names(chapter_rows, {'figure_analysis_txt', 'table_analysis_txt'}))}")
        if chapter_title.startswith("5.5.2"):
            key_result_count = len(unique_names(chapter_rows, {KEY_RESULT_TABLE_ITEM_TYPE}))
            lines.append(f"- 重点结果表：`tables_excel/重点结果表/`，当前共 {key_result_count} 张 Excel。")
        lines.append("")
    md_path = result_index_dir / "章节结果索引.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return {"csv": csv_path, "markdown": md_path}


def audit_png_file(path: Path) -> dict[str, Any]:
    """检查单张 PNG 的基本质量，重点排查空白图和边缘裁切风险。"""
    try:
        from PIL import Image, ImageStat
    except Exception:
        return {
            "宽度": None,
            "高度": None,
            "文件KB": formal_round(path.stat().st_size / 1024, 1) if path.exists() else None,
            "非空白": None,
            "边缘裁切风险": None,
            "检查状态": "未检查：Pillow不可用",
        }

    if not path.exists():
        return {
            "宽度": None,
            "高度": None,
            "文件KB": None,
            "非空白": False,
            "边缘裁切风险": True,
            "检查状态": "失败：文件不存在",
        }

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        width, height = rgb.size
        stat = ImageStat.Stat(gray)
        non_blank = bool(stat.stddev and stat.stddev[0] > 0.5)
        pixels = np.asarray(rgb)
        non_white = np.any(pixels < 245, axis=2)
        border = max(4, min(width, height) // 120)
        edge_mask = np.zeros(non_white.shape, dtype=bool)
        edge_mask[:border, :] = True
        edge_mask[-border:, :] = True
        edge_mask[:, :border] = True
        edge_mask[:, -border:] = True
        edge_content_ratio = float(non_white[edge_mask].mean()) if edge_mask.any() else 0.0
        edge_risk = edge_content_ratio > 0.025
    if not non_blank:
        status = "失败：疑似空白图"
    elif edge_risk:
        status = "需复核：边缘存在内容"
    else:
        status = "通过"
    return {
        "宽度": int(width),
        "高度": int(height),
        "文件KB": formal_round(path.stat().st_size / 1024, 1),
        "非空白": non_blank,
        "边缘裁切风险": edge_risk,
        "边缘内容占比": formal_round(edge_content_ratio, 6),
        "检查状态": status,
    }


def write_figure_quality_report(records: list[dict[str, str]], extra_figure_dirs: list[Path] | None = None) -> Path:
    """输出图片质量检查表，用于确认数值标注和版式没有明显生成错误。"""
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for record in records:
        if record.get("item_type") != "figure":
            continue
        path = Path(str(record.get("path", ""))).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        rows.append(
            {
                "来源": "章节目录",
                "章节": record.get("chapter", ""),
                "图名": record.get("title", path.stem),
                "路径": str(path),
                **audit_png_file(path),
            }
        )
    for figure_dir in extra_figure_dirs or []:
        if not figure_dir.exists():
            continue
        for path in sorted(figure_dir.glob("*.png")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            rows.append(
                {
                    "来源": "正式论文结果副本",
                    "章节": FORMAL_PAPER_RESULT_DIRNAME,
                    "图名": path.stem,
                    "路径": str(resolved),
                    **audit_png_file(resolved),
                }
            )
    report_path = VISUALIZATION_DIR / RESULT_INDEX_DIRNAME / "图片质量检查表.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    return report_path


def build_chapter_531(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.3.1 目标变量小节的目标收益走势和方向分布图。"""
    feature_panel_source = best8_feature_panel_sources()
    for horizon in PAPER_HORIZONS:
        # 第一类图：直接从 feature_panel.csv 提取目标变量走势，用来说明“要预测的 y 到底长什么样”。
        trend_df = build_target_return_trend_table(horizon)
        trend_title = f"{horizon}日目标变量累计收益走势"
        trend_path = plot_target_return_trend(
            trend_df,
            paths.figures_dir / f"{trend_title}.png",
            trend_title,
        )
        add_record(records, paths.chapter_title, "figure", trend_title, trend_path, feature_panel_source, "派生统计")
        write_figure_result_excel_sheet(records, paths, trend_title, trend_path, trend_df, feature_panel_source, "派生统计")

        # 第二类图：统计未来收益方向标签的正负样本分布，用来衔接 5.3.5 的类别不平衡说明。
        direction_df = build_target_direction_distribution_table(horizon)
        direction_title = f"{horizon}日目标收益方向分布图"
        direction_path = plot_target_direction_distribution(
            direction_df,
            paths.figures_dir / f"{direction_title}.png",
            direction_title,
        )
        add_record(records, paths.chapter_title, "figure", direction_title, direction_path, feature_panel_source, "派生统计")
        write_figure_result_excel_sheet(records, paths, direction_title, direction_path, direction_df, feature_panel_source, "派生统计")


def build_chapter_532(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.3.2 特征集小节的筛选诊断图和入选控制变量表。"""
    # 优先读取第 4 步诊断表；如果输出目录被清理，则用 processed 正式数据即时重建同口径相关性表。
    corr_df, corr_source = load_or_build_tradeflow_corr()
    heatmap_table = build_tradeflow_corr_heatmap_table(corr_df)
    heatmap_path = plot_heatmap(
        heatmap_table,
        paths.figures_dir / "交易流变量与候选变量相关性热力图.png",
        "交易流变量与候选变量相关性热力图",
    )
    add_record(records, paths.chapter_title, "figure", "交易流变量与候选变量相关性热力图", heatmap_path, corr_source, "现有结果整理")
    write_figure_result_excel_sheet(records, paths, "交易流变量与候选变量相关性热力图", heatmap_path, heatmap_table, corr_source, "现有结果整理")

    rank_df = (
        corr_df.groupby("candidate_feature", as_index=False)["abs_pearson_corr"]
        .max()
        .sort_values("abs_pearson_corr", ascending=False)
        .reset_index(drop=True)
    )
    rank_df["变量"] = rank_df["candidate_feature"].map(feature_label)
    rank_plot_df = rank_df.rename(columns={"abs_pearson_corr": "最大绝对相关系数"})[["变量", "最大绝对相关系数"]].copy()
    rank_plot_path = plot_bar(
        rank_plot_df,
        label_col="变量",
        value_col="最大绝对相关系数",
        output_path=paths.figures_dir / "候选变量相关性强度排序图.png",
        title="候选变量相关性强度排序图",
        ylabel="最大绝对相关系数",
        value_fmt="{:.2f}",
        rotation=55,
        orientation="horizontal",
    )
    add_record(records, paths.chapter_title, "figure", "候选变量相关性强度排序图", rank_plot_path, corr_source, "现有结果整理")
    write_figure_result_excel_sheet(records, paths, "候选变量相关性强度排序图", rank_plot_path, rank_plot_df, corr_source, "现有结果整理")

    selected_features = build_selected_feature_table()
    selected_only = (
        selected_features.loc[selected_features["是否入选"] == "是", ["变量", "统一筛选重要性"]]
        .sort_values("统一筛选重要性", ascending=False)
        .reset_index(drop=True)
    )
    importance_plot_path = plot_bar(
        selected_only,
        label_col="变量",
        value_col="统一筛选重要性",
        output_path=paths.figures_dir / "八个入选控制变量重要性排序图.png",
        title="八个入选控制变量重要性排序图",
        ylabel="统一筛选重要性",
        value_fmt="{:.0f}",
        rotation=40,
    )
    add_record(records, paths.chapter_title, "figure", "八个入选控制变量重要性排序图", importance_plot_path, str(PROCESSED_DIR / "screening" / "screening_reference_mainline_controls.csv"), "现有结果整理")
    write_figure_result_excel_sheet(
        records,
        paths,
        "八个入选控制变量重要性排序图",
        importance_plot_path,
        selected_only,
        str(PROCESSED_DIR / "screening" / "screening_reference_mainline_controls.csv"),
        "现有结果整理",
    )

    selected_csv_path = write_csv(selected_features, paths.tables_excel_dir / "八个入选控制变量筛选总表.xlsx")
    add_record(records, paths.chapter_title, "table_xlsx", "八个入选控制变量筛选总表", selected_csv_path, str(PROCESSED_DIR / "screening" / "screening_reference_mainline_controls.csv"), "现有结果整理")


def build_chapter_533(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.3.3 特征标准化小节的缺失填补和标准化统计表。"""
    preprocess_df = build_preprocessing_standardization_table()
    preprocess_source = best8_preprocess_stats_sources()
    preprocess_path = write_csv(preprocess_df, paths.tables_excel_dir / "缺失填补与标准化统计表.xlsx")
    add_record(records, paths.chapter_title, "table_xlsx", "缺失填补与标准化统计表", preprocess_path, preprocess_source, "派生统计")


def build_chapter_534(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.3.4 数据划分小节的训练、验证和测试样本表。"""
    split_df = build_sample_split_table()
    feature_panel_source = best8_feature_panel_sources()
    split_table = split_df[["资产", "资产代码", "期限", "划分", "样本量"]].copy()
    split_csv_path = write_csv(split_table, paths.tables_excel_dir / "样本划分表.xlsx")
    add_record(records, paths.chapter_title, "table_xlsx", "样本划分表", split_csv_path, feature_panel_source, "派生统计")


def build_chapter_535(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.3.5 处理类别不平衡小节的正收益占比表。"""
    split_df = build_sample_split_table()
    feature_panel_source = best8_feature_panel_sources()
    balance_table = split_df[["资产", "资产代码", "期限", "划分", "样本量", "正收益占比"]].copy()
    balance_path = write_csv(balance_table, paths.tables_excel_dir / "类别分布与正收益占比表.xlsx")
    add_record(records, paths.chapter_title, "table_xlsx", "类别分布与正收益占比表", balance_path, feature_panel_source, "派生统计")


def build_chapter_54(
    records: list[dict[str, str]],
    search_paths: ChapterPaths,
    best_model_paths: ChapterPaths,
    framework_paths: ChapterPaths,
) -> None:
    """生成 5.4 各小节的参数、最佳模型和方案对比框架表。"""
    # 5.4 这一章以表格为主，不额外生成图片。
    # 5.4.1 放超参数搜索和参数索引；5.4.2 放最佳模型分布；5.4.3 放对比口径说明表。
    screening_required = [
        SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_coarse_search.csv",
        SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_fine_search.csv",
        SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_metrics.csv",
        SCREENING_OUTPUT_DIR / "unified_lgbm" / "screening_lgbm_best_params.csv",
    ]
    if all(path.exists() for path in screening_required):
        screening_summary = build_screening_training_summary()
        screening_csv_path = write_csv(screening_summary, search_paths.tables_excel_dir / "统一筛选模型调参摘要表.xlsx")
        add_record(
            records,
            search_paths.chapter_title,
            "table_xlsx",
            "统一筛选模型调参摘要表",
            screening_csv_path,
            str(SCREENING_OUTPUT_DIR / "unified_lgbm"),
            "现有结果整理",
        )

    param_index = build_classification_param_index()
    param_index_csv_path = write_csv(param_index, search_paths.tables_excel_dir / "主线分类模型阈值特征数最佳参数索引表.xlsx")
    add_record(records, search_paths.chapter_title, "table_xlsx", "主线分类模型阈值特征数最佳参数索引表", param_index_csv_path, str(MAINLINE_CLASSIFICATION_OUTPUT_DIR), "现有结果整理")

    best_model_df = build_best_model_summary()
    best_model_csv_path = write_csv(best_model_df, best_model_paths.tables_excel_dir / "各资产各期限最佳模型汇总表.xlsx")
    add_record(records, best_model_paths.chapter_title, "table_xlsx", "各资产各期限最佳模型汇总表", best_model_csv_path, str(MAINLINE_SUMMARY_DIR), "现有结果整理")

    framework_df = pd.DataFrame(
        [
            {"项目": "基准方案", "口径": "使用正式控制变量，不加入四个交易流指标。"},
            {"项目": "扩展方案", "口径": "在基准方案基础上加入行业交易流、机构交易流、交易分歧、交易分歧×多空分歧。"},
            {"项目": "机器学习阈值", "口径": "只用验证集选择阈值；index采用valid_f1_prev015，非指数资产采用valid_f1_cap075。"},
            {"项目": "OLS阈值", "口径": "线性概率模型连续分数，采用valid_target_pos_040_nondegenerate。"},
            {"项目": "性能提升", "口径": "扩展方案减去基准方案，展示表按AUC提升第一、F1提升第二排序。"},
            {"项目": "测试集使用", "口径": "测试集只用于最终评估，不参与特征、参数或阈值选择。"},
        ]
    )
    framework_path = write_csv(framework_df, framework_paths.tables_excel_dir / "基准方案与扩展方案对比框架表.xlsx")
    add_record(records, framework_paths.chapter_title, "table_xlsx", "基准方案与扩展方案对比框架表", framework_path, str(MAINLINE_SUMMARY_DIR), "现有结果整理")


def build_chapter_55(records: list[dict[str, str]], task_paths: ChapterPaths, baseline_paths: ChapterPaths) -> None:
    """生成 5.5.1 分类任务图和 5.5.2 基准模型对比表。"""
    comparison_workbook_path = baseline_paths.tables_excel_dir / "模型与基线评估指标对比.xlsx"
    comparison_frames: dict[str, pd.DataFrame] = {}
    for horizon in PAPER_HORIZONS:
        # 先把 with_tradeflow4 和 no_tradeflow4 两种方案按资产、模型、期限配对，
        # 后面的图片和 Excel 对比表都从这张 merged 表继续整理。
        merged = merge_with_no_metrics(horizon)
        by_asset = (
            merged.groupby(["资产", "asset_alias"], as_index=False)
            .agg(平均测试AUC增益=("test_auc_gain", "mean"), 平均测试F1增益=("test_f1_gain", "mean"))
            .rename(columns={"test_auc_gain": "平均测试AUC增益"})
            .sort_values(["平均测试AUC增益", "平均测试F1增益", "asset_alias"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        by_model = (
            merged.groupby(["模型", "model_id"], as_index=False)
            .agg(平均测试AUC增益=("test_auc_gain", "mean"), 平均测试F1增益=("test_f1_gain", "mean"))
            .rename(columns={"test_auc_gain": "平均测试AUC增益"})
            .sort_values(["平均测试AUC增益", "平均测试F1增益", "model_id"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        asset_plot = plot_bar(
            by_asset,
            label_col="资产",
            value_col="平均测试AUC增益",
            output_path=task_paths.figures_dir / f"{horizon}日交易流增益按资产.png",
            title=f"{horizon}日交易流增益按资产",
            ylabel="平均测试AUC增益",
            value_fmt="{:.3f}",
            zero_line=True,
        )
        add_record(records, task_paths.chapter_title, "figure", f"{horizon}日交易流增益按资产", asset_plot, str(MAINLINE_SUMMARY_DIR), "现有结果整理")
        write_figure_result_excel_sheet(records, task_paths, f"{horizon}日交易流增益按资产", asset_plot, by_asset, str(MAINLINE_SUMMARY_DIR), "现有结果整理")
        model_plot = plot_bar(
            by_model,
            label_col="模型",
            value_col="平均测试AUC增益",
            output_path=task_paths.figures_dir / f"{horizon}日交易流增益按模型.png",
            title=f"{horizon}日交易流增益按模型",
            ylabel="平均测试AUC增益",
            value_fmt="{:.3f}",
            zero_line=True,
            rotation=20,
        )
        add_record(records, task_paths.chapter_title, "figure", f"{horizon}日交易流增益按模型", model_plot, str(MAINLINE_SUMMARY_DIR), "现有结果整理")
        write_figure_result_excel_sheet(records, task_paths, f"{horizon}日交易流增益按模型", model_plot, by_model, str(MAINLINE_SUMMARY_DIR), "现有结果整理")

        eval_table = merged[
            [
                "资产",
                "asset_alias",
                "模型",
                "model_id",
                "期限",
                "test_auc_with",
                "test_auc_no",
                "test_auc_gain",
                "test_f1_with",
                "test_f1_no",
                "test_f1_gain",
                "valid_auc_with",
                "valid_auc_no",
                "valid_auc_gain",
                "valid_f1_with",
                "valid_f1_no",
                "valid_f1_gain",
            ]
        ].rename(
            columns={
                "asset_alias": "资产代码",
                "model_id": "模型代码",
                "test_auc_with": "加入交易流测试AUC",
                "test_auc_no": "不加入交易流测试AUC",
                "test_auc_gain": "测试AUC增益",
                "test_f1_with": "加入交易流测试F1",
                "test_f1_no": "不加入交易流测试F1",
                "test_f1_gain": "测试F1增益",
                "valid_auc_with": "加入交易流验证AUC",
                "valid_auc_no": "不加入交易流验证AUC",
                "valid_auc_gain": "验证AUC增益",
                "valid_f1_with": "加入交易流验证F1",
                "valid_f1_no": "不加入交易流验证F1",
                "valid_f1_gain": "验证F1增益",
            }
        ).sort_values(["测试AUC增益", "测试F1增益"], ascending=[False, False])
        eval_sheet_name = f"{horizon}日模型基线对比"
        write_excel_sheet(eval_table, comparison_workbook_path, eval_sheet_name)
        comparison_frames[eval_sheet_name] = eval_table
        add_record(
            records,
            baseline_paths.chapter_title,
            "table_xlsx",
            f"{horizon}日加入与不加入交易流分类评估总表",
            comparison_workbook_path,
            str(MAINLINE_SUMMARY_DIR),
            "现有结果整理",
            note="写入模型与基线评估指标对比工作簿",
            excel_sheet=eval_sheet_name,
        )

    ablation_frames = []
    for horizon in PAPER_HORIZONS:
        ablation_path = MAINLINE_SUMMARY_DIR / classification_summary_filename("ablation_overall", HORIZON_TO_EXPERIMENT_TAG[horizon])
        ablation_df = pd.read_csv(ablation_path)
        ablation_df["期限"] = f"{horizon}日"
        ablation_df["模型"] = ablation_df["model_id"].map(model_label)
        ablation_frames.append(ablation_df)
    ablation_all = pd.concat(ablation_frames, ignore_index=True)[
        [
            "期限",
            "模型",
            "model_id",
            "avg_valid_auc_with_tradeflow4",
            "avg_valid_auc_no_tradeflow4",
            "avg_valid_auc_diff",
            "avg_test_auc_diff",
            "avg_test_f1_diff",
            "avg_test_accuracy_diff",
            "avg_test_precision_diff",
            "avg_test_recall_diff",
            "avg_test_balanced_accuracy_diff",
        ]
    ].rename(
        columns={
            "model_id": "模型代码",
            "avg_valid_auc_with_tradeflow4": "加入交易流平均验证AUC",
            "avg_valid_auc_no_tradeflow4": "不加入交易流平均验证AUC",
            "avg_valid_auc_diff": "平均验证AUC增益",
            "avg_test_auc_diff": "平均测试AUC增益",
            "avg_test_f1_diff": "平均测试F1增益",
            "avg_test_accuracy_diff": "平均测试准确率增益",
            "avg_test_precision_diff": "平均测试精确率增益",
            "avg_test_recall_diff": "平均测试召回率增益",
            "avg_test_balanced_accuracy_diff": "平均测试BA增益",
        }
    ).sort_values(["平均测试AUC增益", "平均测试F1增益"], ascending=[False, False])
    write_excel_sheet(ablation_all, comparison_workbook_path, "总体消融对比")
    comparison_frames["总体消融对比"] = ablation_all
    add_record(
        records,
        baseline_paths.chapter_title,
        "table_xlsx",
        "总体消融对比总表",
        comparison_workbook_path,
        str(MAINLINE_SUMMARY_DIR),
        "现有结果整理",
        note="写入模型与基线评估指标对比工作簿",
        excel_sheet="总体消融对比",
    )

    best_model_df = build_best_model_summary()
    write_excel_sheet(best_model_df, comparison_workbook_path, "最佳模型表现")
    comparison_frames["最佳模型表现"] = best_model_df
    add_record(
        records,
        baseline_paths.chapter_title,
        "table_xlsx",
        "最佳模型测试验证表现表",
        comparison_workbook_path,
        str(MAINLINE_SUMMARY_DIR),
        "现有结果整理",
        note="写入模型与基线评估指标对比工作簿",
        excel_sheet="最佳模型表现",
    )

    build_key_result_tables(records, baseline_paths)


def build_chapter_56(records: list[dict[str, str]], paths: ChapterPaths) -> None:
    """生成 5.6 章节的代表性模型重要性表和重要性对照图。"""
    for horizon in PAPER_HORIZONS:
        representative_model = choose_representative_tree_model(horizon)
        importance_table = build_representative_importance_table(horizon, representative_model)
        importance_plot_df = importance_table.copy()
        importance_plot_df.insert(0, "代表模型", model_label(representative_model))
        importance_plot_df.insert(1, "代表模型代码", representative_model)
        importance_plot_df.insert(2, "期限", f"{horizon}日")
        table_xlsx_path = write_csv(importance_table, paths.tables_excel_dir / f"{horizon}日代表性模型重点特征重要性表.xlsx")
        add_record(records, paths.chapter_title, "table_xlsx", f"{horizon}日代表性模型重点特征重要性表", table_xlsx_path, str(MAINLINE_CLASSIFICATION_OUTPUT_DIR), "现有结果整理")
        importance_plot = plot_importance_comparison(
            importance_plot_df,
            paths.figures_dir / f"{horizon}日代表性树模型加入交易流前后重要性对照.png",
            f"{horizon}日代表性树模型加入交易流前后重要性对照",
        )
        add_record(records, paths.chapter_title, "figure", f"{horizon}日代表性树模型加入交易流前后重要性对照", importance_plot, str(table_xlsx_path), "现有结果整理", "重要性前后对照")
        write_figure_result_excel_sheet(
            records,
            paths,
            f"{horizon}日代表性树模型加入交易流前后重要性对照",
            importance_plot,
            importance_plot_df,
            str(table_xlsx_path),
            "现有结果整理",
            note="重要性前后对照",
        )



def build_chapter_57(
    records: list[dict[str, str]],
    state_paths: ChapterPaths,
    group_paths: ChapterPaths,
    robustness_paths: ChapterPaths,
) -> None:
    """生成 5.7.1 状态划分、5.7.2 分组预测和 5.7.3 稳健性检验结果。"""
    # 5.7 章节分三层：
    # 1）先说明不同状态下样本量是否足够；
    # 2）再比较不同状态下交易流增益；
    # 3）最后用稳健性总表汇总可计算组合数与正增益组合数。
    sample_counts = build_state_sample_counts()
    feature_panel_source = best8_feature_panel_sources()
    sample_csv_path = write_csv(sample_counts.rename(columns={"state": "状态", "horizon": "期限代码"}), state_paths.tables_excel_dir / "各状态样本量分布表.xlsx")
    add_record(records, state_paths.chapter_title, "table_xlsx", "各状态样本量分布表", sample_csv_path, feature_panel_source, "派生统计")
    sample_plot_df = build_state_sample_distribution_table(sample_counts)
    sample_plot = plot_state_sample_distribution(sample_plot_df, state_paths.figures_dir / "各状态样本量分布.png")
    add_record(records, state_paths.chapter_title, "figure", "各状态样本量分布", sample_plot, str(sample_csv_path), "派生统计")
    write_figure_result_excel_sheet(records, state_paths, "各状态样本量分布", sample_plot, sample_plot_df, feature_panel_source, "派生统计")

    robustness = build_state_robustness_summary()
    robustness_csv_path = write_csv(
        robustness[["期限", "state", "总组合数", "可计算组合数", "正增益组合数", "平均AUC增益", "不可计算组合数"]].rename(columns={"state": "状态"}),
        robustness_paths.tables_excel_dir / "各状态稳健性总表.xlsx",
    )
    add_record(records, robustness_paths.chapter_title, "table_xlsx", "各状态稳健性总表", robustness_csv_path, str(MAINLINE_CLASSIFICATION_OUTPUT_DIR), "派生统计")

    for horizon in PAPER_HORIZONS:
        horizon_df = build_state_gain_plot_table(robustness, horizon)
        gain_plot = plot_bar(
            horizon_df,
            label_col="状态",
            value_col="状态平均AUC增益",
            output_path=group_paths.figures_dir / f"{horizon}日各状态下交易流增益.png",
            title=f"{horizon}日各状态下交易流增益",
            ylabel="状态平均AUC增益",
            value_fmt="{:.3f}",
            zero_line=True,
        )
        add_record(records, group_paths.chapter_title, "figure", f"{horizon}日各状态下交易流增益", gain_plot, str(robustness_csv_path), "派生统计")
        write_figure_result_excel_sheet(records, group_paths, f"{horizon}日各状态下交易流增益", gain_plot, horizon_df, str(robustness_csv_path), "派生统计")


def run() -> dict[str, Path]:
    """执行主线结果图片与结果索引的全量重建。"""
    # 第 11 步不会重新训练任何模型。
    # 它只读取已经存在的 screening、classification、summary 和 processed 结果，然后整理成结果目录。
    ensure_best8_dirs()
    configure_matplotlib()
    # 重建 outputs/visualizations 下的生成型产物。
    # 这里清理范围只限当前结果目录，不会触碰筛选、训练和汇总阶段的上游文件。
    prepare_visualization_root()
    records: list[dict[str, str]] = []
    paths_map = chapter_paths()
    # 下面严格按结果目录顺序逐章写出图片、章节 Excel 和章节结果目录记录。
    # 如果需要追踪“某一章的图和表到底由哪段代码生成”，就沿着对应的 build_chapter_* 函数往上看。
    build_chapter_531(records, paths_map["5.3.1 目标变量"])
    build_chapter_532(records, paths_map["5.3.2 特征集"])
    build_chapter_533(records, paths_map["5.3.3 特征标准化"])
    build_chapter_534(records, paths_map["5.3.4 数据划分"])
    build_chapter_535(records, paths_map["5.3.5 处理类别不平衡"])
    build_chapter_54(
        records,
        paths_map["5.4.1 超参数搜索过程"],
        paths_map["5.4.2 最优模型分布与选择"],
        paths_map["5.4.3 基准方案与扩展方案对比框架"],
    )
    build_chapter_55(records, paths_map["5.5.1 分类任务"], paths_map["5.5.2 基准模型对比"])
    formal_output_dir = VISUALIZATION_DIR / FORMAL_PAPER_RESULT_DIRNAME
    write_formal_paper_outputs(records, formal_output_dir, register_records=True)
    try:
        write_formal_paper_outputs(records, ROOT_PAPER_RESULT_DIR, register_records=False)
    except PermissionError:
        # 人工查看目录可能正被 WPS 打开；正式结果仍以 outputs/visualizations/正式论文结果 为准。
        pass
    build_chapter_56(records, paths_map["5.6 基于变量重要性的可解释性分析"])
    build_chapter_57(records, paths_map["5.7.1 状态划分"], paths_map["5.7.2 分组预测"], paths_map["5.7.3 稳健性检验"])
    write_formal_paper_figures(records, formal_output_dir)
    try:
        write_formal_paper_figures(records, ROOT_PAPER_RESULT_DIR)
    except PermissionError:
        pass
    register_static_text_records(records, paths_map)
    result_index_paths = write_result_index(records)
    figure_quality_report = write_figure_quality_report(records, [formal_output_dir / "图片"])
    return {
        "result_index_csv": result_index_paths["csv"],
        "result_index_markdown": result_index_paths["markdown"],
        "figure_quality_report": figure_quality_report,
        "root": VISUALIZATION_DIR,
    }


def main() -> None:
    """命令行入口；业务逻辑在 run() 中。"""
    parser = argparse.ArgumentParser(description="第 11 步：在已有训练结果基础上重建结果图片、Excel 结果表和结果索引。")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
