# 交易流预测效应项目说明

本项目主线围绕“异质投资者交易流是否提升收益方向预测效果”展开。正式结果以当前主线目录为准，不再依赖历史试验目录。

## 公开仓库说明

本仓库只发布可复现的代码、依赖列表和运行说明，不包含原始数据、处理后数据、训练好的模型、预测明细、论文图表成品或 Word 文档。

本地运行时需要自行准备授权数据文件，并放入 `data/` 目录。运行生成的 `models/` 和 `outputs/` 文件只保留在本地，不上传到 GitHub。

## 环境准备

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 主线目录

- `data/raw/`：原始 Excel 数据。
- `data/processed/`：清洗、特征工程、筛选参考和建模输入。
- `outputs/screening/`：变量筛选和交易流辅助诊断结果。
- `outputs/classification/`：100 组机器学习分类模型正式结果。
- `outputs/summary/`：分类模型汇总、交易流增益对比和 OLS 正式结果。
- `outputs/visualizations/`：按当前论文第五章结构整理的图片、Excel 表和索引。
- `outputs/visualizations/正式论文结果/`：论文替换用的成品表格与图片集中目录。
- `论文结果_混合阈值F1提升/`：人工查看用成品副本。
- `models/classification/`：100 组分类模型文件。
- `outputs/项目清单/`：代码文件清单和结果文件清单。

## 运行顺序

1. `src/step_01_prepare_panel.py`
2. `src/step_02_prepare_features.py`
3. `src/step_03_build_screening_long_panel.py`
4. `src/step_04_check_screening_missing.py`
5. `src/step_05_screen_lgbm.py`
6. `src/step_06_screen_tradeflow_lgbm.py`
7. `src/step_07_build_selected_features_from_controls.py`
8. `src/step_08_prepare_model_data.py`
9. `src/step_09_train_batch_cls.py`
10. `src/step_10_summarize_cls_results.py`
11. `src/run_all_ols_and_select_candidates.py`
12. `src/step_11_build_visualizations.py`
13. `src/step_12_build_inventory.py`

第 11 步只读取已经生成的筛选、训练、汇总和 OLS 结果，不重新训练模型。

## 当前正式算法口径

机器学习分类模型：

- 资产范围：`index`、`eg`、`bu`、`jm`、`pp`。
- 期限：1 日、22 日。
- 方案：`no_tradeflow4` 与 `with_tradeflow4`。
- 模型：决策树、随机森林、支持向量机、XGBoost、LightGBM。
- 超参数搜索使用当前主线扩展搜索空间和候选非退化过滤。
- 参数和阈值只使用训练集、验证集选择；测试集只用于最终评估。
- 阈值口径为 `mixed_f1_by_asset`：指数使用 `valid_f1_prev015`，非指数资产使用 `valid_f1_cap075`。

OLS：

- 使用线性概率模型连续分数，不解释为严格概率。
- 阈值口径为 `valid_target_pos_040_nondegenerate`。
- OLS 结果保存在 `outputs/summary/ols/`。

## 第 11 步结果结构

`outputs/visualizations/` 当前按论文第五章结构重建：

- `5.3 特征工程与数据准备/5.3.1 目标变量/`
- `5.3 特征工程与数据准备/5.3.2 特征集/`
- `5.3 特征工程与数据准备/5.3.3 特征标准化/`
- `5.3 特征工程与数据准备/5.3.4 数据划分/`
- `5.3 特征工程与数据准备/5.3.5 处理类别不平衡/`
- `5.4 模型训练与超参数优化/5.4.1 超参数搜索过程/`
- `5.4 模型训练与超参数优化/5.4.2 最优模型分布与选择/`
- `5.4 模型训练与超参数优化/5.4.3 基准方案与扩展方案对比框架/`
- `5.5 预测效果评估/5.5.1 分类任务/`
- `5.5 预测效果评估/5.5.2 基准模型对比/`
- `5.6 基于变量重要性的可解释性分析/`
- `5.7 交易流指标在不同市场状态下的预测表现/5.7.1 状态划分/`
- `5.7 交易流指标在不同市场状态下的预测表现/5.7.2 分组预测/`
- `5.7 交易流指标在不同市场状态下的预测表现/5.7.3 稳健性检验/`
- `结果索引/`
- `正式论文结果/`

每次运行第 11 步会清空并重建 `outputs/visualizations/` 下的生成结果，防止旧图表和旧章节编号混用。

## 结果检查

优先查看：

- `outputs/visualizations/结果索引/章节结果索引.md`
- `outputs/visualizations/结果索引/章节结果目录表.csv`
- `outputs/visualizations/结果索引/图片质量检查表.csv`
- `outputs/visualizations/正式论文结果/`

图片质量检查表会记录每张图片的宽度、高度、文件大小、是否空白和边缘裁切风险。论文展示型表格不放入阈值、预测正类比例和历史版本使用过的旧命名字段。

## 常用修改入口

- 资产、期限、方案、变量清单：`src/config.py`
- 搜索空间：`src/common/search_specs.py`
- 模型训练、候选过滤、阈值选择：`src/common/model_train_utils.py`
- 指标计算：`src/common/metrics_utils.py`
- 图表样式和数值标注：`src/common/visual_style.py`
- 论文章节目录、图片、Excel 和正式结果表：`src/step_11_build_visualizations.py`
- 项目文件清单：`src/step_12_build_inventory.py`
