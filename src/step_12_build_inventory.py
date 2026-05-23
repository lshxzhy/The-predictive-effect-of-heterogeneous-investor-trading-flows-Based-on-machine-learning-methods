from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 第 12 步：生成代码和结果清单
# 当前主线位置：章节结果与模型结果 -> 项目清单。
# 本文件负责：扫描项目白名单代码、结果和模型，生成便于查找的代码文件清单和结果文件清单。
# 主函数：`run()`。
# 命令行入口：`main()`，负责触发 `run()`。
# 直接输入：`src/` 白名单代码；`outputs/` 白名单结果；`models/` 白名单模型；`outputs/visualizations/结果索引/`。
# 直接输出：`outputs/项目清单/代码文件清单.csv`；`outputs/项目清单/结果文件清单.csv`。
# 下游读取：无，属于主线最后一步。
# 关键修改位置：代码白名单、结果白名单、清单字段、结果分类规则和章节结果索引路径映射。
# 变更后重跑起点：第 12 步。该步骤只负责重新扫描当前项目状态并更新清单。
# 对应文档：`README.md` 的“主线结构总览”“结果目录与阅读顺序”和“出错时优先检查”。
import argparse
import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MAINLINE_CODE_ROOT = ROOT_DIR / "src"
MAINLINE_OUTPUT_ROOT = ROOT_DIR / "outputs"
MAINLINE_MODEL_ROOT = ROOT_DIR / "models"
PROJECT_INDEX_DIR = MAINLINE_OUTPUT_ROOT / "项目清单"
LEGACY_RESULT_PREFIXES = (
    "outputs/report/",
    "outputs/exhibits/",
    "outputs/best8_1d/",
    "outputs/best8_1d_22d/",
)


CODE_METADATA = {
    # 运行入口：按 step_01 到 step_12 的编号顺序执行。
    "src/step_01_prepare_panel.py": ("step", "第01步：合并两份原始数据，生成单资产原始面板", "运行入口"),
    "src/step_02_prepare_features.py": ("step", "第02步：构造衍生特征、目标变量和时间切分", "运行入口"),
    "src/step_03_build_screening_long_panel.py": ("step", "第03步：拼接多资产统一筛选长面板", "运行入口"),
    "src/step_04_check_screening_missing.py": ("step", "第04步：生成筛选缺失与相关性诊断", "运行入口"),
    "src/step_05_screen_lgbm.py": ("step", "第05步：训练统一筛选模型并导出重要性", "运行入口"),
    "src/step_06_screen_tradeflow_lgbm.py": ("step", "第06步：训练交易流辅助诊断模型", "运行入口"),
    "src/step_07_build_selected_features_from_controls.py": ("step", "第07步：写出正式控制变量文件和筛选参考总表", "运行入口"),
    "src/step_08_prepare_model_data.py": ("step", "第08步：按资产、期限和方案生成建模输入", "运行入口"),
    "src/step_09_train_batch_cls.py": ("step", "第09步：批量训练分类模型", "运行入口"),
    "src/step_10_summarize_cls_results.py": ("step", "第10步：汇总分类训练结果", "运行入口"),
    "src/step_11_build_visualizations.py": ("step", "第11步：生成结果图片、配套数据表和结果索引", "运行入口"),
    "src/step_12_build_inventory.py": ("step", "第12步：生成代码文件清单和结果文件清单", "运行入口"),
    # 支撑文件：由上面的入口自动导入，不需要单独运行。
    # __init__.py 不是运行入口，但属于当前代码结构的一部分，需要进入代码文件清单。
    "src/__init__.py": ("support", "src 包识别文件", "支撑文件"),
    "src/config.py": ("support", "资产、期限、模型、方案、变量和路径规则", "支撑文件"),
    "src/common/__init__.py": ("support", "公共模块目录识别文件", "支撑文件"),
    "src/common/excel_utils.py": ("support", "读取并校验原始 Excel", "支撑文件"),
    "src/common/feature_registry.py": ("support", "统一维护衍生特征、目标变量和时间切分", "支撑文件"),
    "src/common/metrics_utils.py": ("support", "分类和回归评价指标", "支撑文件"),
    "src/common/model_train_utils.py": ("support", "分类模型训练、调参、阈值和重要性导出", "支撑文件"),
    "src/common/paths.py": ("support", "目录创建与路径辅助函数", "支撑文件"),
    "src/common/preprocessing.py": ("support", "仅用训练集统计量做缺失填补和标准化", "支撑文件"),
    "src/common/search_specs.py": ("support", "筛选模型和分类模型的参数搜索空间", "支撑文件"),
    "src/common/paper_common.py": ("support", "结果图片与主线汇总共用常量", "支撑文件"),
    "src/common/visual_style.py": ("support", "结果图片统一样式、中文字体和数值标注", "支撑文件"),
}

CODE_WHITELIST = tuple(CODE_METADATA.keys())
RESULT_WHITELIST_PREFIXES = (
    "outputs/screening/",
    "outputs/classification/",
    "outputs/summary/",
    "outputs/visualizations/",
    "outputs/visualizations/结果索引/",
    "outputs/项目清单/",
    "models/classification/",
    "models/screen_lgbm/",
    "models/screen_tradeflow_lgbm/",
)
RESULT_BLACKLIST_PREFIXES = LEGACY_RESULT_PREFIXES + (
    "outputs/visualizations/资产收盘价走势图/",
)
RESULT_BLACKLIST_SUFFIXES: tuple[str, ...] = ()


def ensure_project_index_dir() -> None:
    """确保项目清单目录存在。"""
    PROJECT_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def iter_code_rows() -> list[dict[str, str]]:
    """遍历当前项目代码白名单，生成代码文件清单记录。"""
    rows: list[dict[str, str]] = []
    for relative_path in CODE_WHITELIST:
        path = ROOT_DIR / Path(relative_path)
        if not path.exists():
            continue
        subsystem, role, entry_type = CODE_METADATA.get(relative_path, (path.parent.name, "主线代码文件", "支撑文件"))
        # entry_type 用来区分“主线运行入口”和“只需阅读、不需要单独运行的支撑文件”。
        rows.append(
            {
                "relative_path": relative_path,
                "subsystem": subsystem,
                "role": role,
                "entry_type": entry_type,
            }
        )
    return rows


def describe_result_path(relative_path: str) -> tuple[str, str, str]:
    """根据结果路径判断产出来源、范围和说明文字。"""
    path = Path(relative_path)
    name = path.name
    if relative_path.startswith("outputs/screening/diagnostics/"):
        return ("src.step_04_check_screening_missing", "screening", f"筛选诊断产物：{name}")
    if relative_path.startswith("outputs/screening/unified_lgbm/"):
        return ("src.step_05_screen_lgbm", "screening", f"统一筛选模型产物：{name}")
    if relative_path.startswith("outputs/screening/tradeflow_summary/") or relative_path.startswith("outputs/screening/screen_tradeflow_lgbm/"):
        return ("src.step_06_screen_tradeflow_lgbm", "screening", f"交易流辅助诊断产物：{name}")
    if relative_path.startswith("outputs/screening/selected_features") or relative_path.startswith("outputs/screening/screening_reference"):
        return ("src.step_07_build_selected_features_from_controls", "screening", f"正式控制变量与筛选参考产物：{name}")
    if relative_path.startswith("outputs/screening/"):
        return ("src.step_03_to_step_07", "screening", f"筛选阶段产物：{name}")
    if relative_path.startswith("outputs/classification/"):
        return ("src.step_09_train_batch_cls", "classification", f"分类模型单次训练产物：{name}")
    if relative_path.startswith("outputs/summary/"):
        return ("src.step_10_summarize_cls_results", "summary", f"分类汇总产物：{name}")
    if relative_path.startswith("outputs/visualizations/"):
        return ("src.step_11_build_visualizations", "visualizations", f"论文章节图表产物：{name}")
    if relative_path.startswith("outputs/项目清单/"):
        return ("src.step_12_build_inventory", "project_index", f"项目清单文件：{name}")
    if relative_path.startswith("models/classification/"):
        return ("src.step_09_train_batch_cls", "classification", "已训练分类模型文件")
    if relative_path.startswith("models/screen_lgbm/"):
        return ("src.step_05_screen_lgbm", "screening", "统一筛选模型文件")
    if relative_path.startswith("models/screen_tradeflow_lgbm/"):
        return ("src.step_06_screen_tradeflow_lgbm", "screening", "交易流辅助诊断模型文件")
    return ("unknown", "unknown", "主线产物")


def include_result_path(relative_path: str) -> bool:
    """判断某个结果路径是否属于项目结果清单的白名单。"""
    if any(relative_path.startswith(prefix) for prefix in RESULT_BLACKLIST_PREFIXES):
        return False
    if relative_path.startswith("outputs/summary/") and relative_path.endswith(".xlsx"):
        return False
    if relative_path.endswith(RESULT_BLACKLIST_SUFFIXES):
        return False
    # 只有通过白名单的结果才会进入结果文件清单，避免无关目录或技术副产物混进当前结果目录。
    return any(relative_path.startswith(prefix) for prefix in RESULT_WHITELIST_PREFIXES)


def iter_result_rows() -> list[dict[str, str]]:
    """遍历当前项目结果目录和模型目录，生成结果文件清单记录。"""
    rows: list[dict[str, str]] = []
    for root in [MAINLINE_OUTPUT_ROOT, MAINLINE_MODEL_ROOT]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            relative_path = path.relative_to(ROOT_DIR).as_posix()
            if not include_result_path(relative_path):
                continue
            producer, scope, description = describe_result_path(relative_path)
            rows.append(
                {
                    "relative_path": relative_path,
                    "producer": producer,
                    "scope": scope,
                    "description": description,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """把清单记录写成 UTF-8 BOM CSV。"""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Path]:
    """重建当前项目的代码文件清单和结果文件清单。"""
    ensure_project_index_dir()
    code_rows = iter_code_rows()
    result_rows = iter_result_rows()
    code_path = PROJECT_INDEX_DIR / "代码文件清单.csv"
    result_path = PROJECT_INDEX_DIR / "结果文件清单.csv"
    # 代码文件清单负责说明“哪些文件需要按编号运行，哪些文件只是支撑文件”。
    write_csv(code_path, code_rows, ["relative_path", "subsystem", "role", "entry_type"])
    # 结果文件清单负责说明“每类结果由谁生成、属于哪一层结果”。
    write_csv(result_path, result_rows, ["relative_path", "producer", "scope", "description"])
    return {"code_inventory": code_path, "result_inventory": result_path}


def main() -> None:
    """命令行入口；业务逻辑在 run() 中。"""
    parser = argparse.ArgumentParser(description="第 12 步：按项目白名单重建代码文件清单和结果文件清单。")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
