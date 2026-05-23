from __future__ import annotations

# 本模块负责：统一控制主线图表的字体、灰度风格、边框、数值标签和表格渲染方式。
# 直接服务的步骤：第 04 步和第 11 步。
# 关键修改位置：字体候选、灰度配色、坐标轴样式、数字标注和图片保存逻辑。
# 变更后重跑起点：第 04 步或第 11 步，取决于修改影响的是筛选诊断图还是章节图表。
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import font_manager


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# 主线图表默认使用常见 Windows 中文字体和灰度配色。
# 统一修改字体、灰度深浅或边框风格时，优先改这里。
CHINESE_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun"]
GRAYSCALE_PALETTE = ["0.20", "0.35", "0.50", "0.65", "0.80", "0.90"]


def pick_chinese_font() -> str:
    """按候选顺序选择本机可用的中文字体。"""

    # 如果项目迁移到另一台电脑后出现中文显示异常，
    # 先在这里补一项本机已经安装的中文字体名称。
    for font_name in CHINESE_FONT_CANDIDATES:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            return font_name
        except Exception:
            continue
    return "sans-serif"


def configure_matplotlib() -> str:
    """统一设置主线图表使用的字体、边框和灰度样式。"""

    font_name = pick_chinese_font()
    # 所有主线图片都会复用这里的 rcParams。
    # 统一修改字体、线宽、背景色或是否显示网格线时，优先改这里。
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        }
    )
    return font_name


def gray_colors(n: int) -> list[str]:
    """按需要数量返回循环复用的灰度配色。"""

    if n <= 0:
        return []
    repeats = (n // len(GRAYSCALE_PALETTE)) + 1
    return (GRAYSCALE_PALETTE * repeats)[:n]


def style_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str | None = None,
    xlabel: str | None = None,
    enable_y_grid: bool = False,
) -> None:
    """统一设置坐标轴标题、标签和边框样式，默认不显示网格线。"""

    # 标题允许为空字符串。
    # 如果某类图只想保留坐标轴和图例，而不在图内额外写标题，就传空字符串。
    normalized_title = str(title).strip()
    if normalized_title:
        ax.set_title(normalized_title, fontsize=13, pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if enable_y_grid:
        ax.grid(axis="y", color="0.92", linewidth=0.45)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    # 默认隐藏上、右边框，避免出现多余横线和竖线。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, bars: Any, fmt: str = "{:.3f}", fontsize: int = 9, rotation: int = 0) -> None:
    """给柱状图添加数值标签，并自动扩展纵轴留白。"""

    heights = [bar.get_height() for bar in bars if not pd.isna(bar.get_height())]
    if not heights:
        return

    ymin, ymax = ax.get_ylim()
    data_min = min(heights + [0.0])
    data_max = max(heights + [0.0])
    span = max(ymax - ymin, abs(data_max - data_min), abs(data_max), abs(data_min), 1.0)
    pad = span * 0.08
    new_ymin = min(ymin, data_min - pad if data_min < 0 else ymin)
    new_ymax = max(ymax, data_max + pad * 1.6)
    ax.set_ylim(new_ymin, new_ymax)

    # 先扩展坐标轴范围，再计算数字标签偏移量，避免数值压在柱顶或越出边界。
    offset = max((new_ymax - new_ymin) * 0.022, span * 0.012)
    for bar in bars:
        height = bar.get_height()
        x_center = bar.get_x() + bar.get_width() / 2
        if pd.isna(height):
            continue
        va = "bottom" if height >= 0 else "top"
        y_pos = height + offset if height >= 0 else height - offset
        ax.text(
            x_center,
            y_pos,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
        )


def annotate_bars_horizontal(ax: plt.Axes, bars: Any, fmt: str = "{:.3f}") -> None:
    """给水平柱状图添加数值标签，并自动扩展横轴留白。"""

    widths = [bar.get_width() for bar in bars if not pd.isna(bar.get_width())]
    if not widths:
        return

    xmin, xmax = ax.get_xlim()
    data_min = min(widths + [0.0])
    data_max = max(widths + [0.0])
    span = max(xmax - xmin, abs(data_max - data_min), abs(data_max), abs(data_min), 1.0)
    pad = span * 0.08
    new_xmin = min(xmin, data_min - pad if data_min < 0 else xmin)
    new_xmax = max(xmax, data_max + pad * 1.6)
    ax.set_xlim(new_xmin, new_xmax)

    # 水平柱状图也先扩展横轴范围，再写数字，避免标签与最长柱重叠。
    offset = max((new_xmax - new_xmin) * 0.018, span * 0.010)
    for bar in bars:
        width = bar.get_width()
        if pd.isna(width):
            continue
        y_center = bar.get_y() + bar.get_height() / 2
        ha = "left" if width >= 0 else "right"
        x_pos = width + offset if width >= 0 else width - offset
        ax.text(
            x_pos,
            y_center,
            fmt.format(width),
            ha=ha,
            va="center",
            fontsize=9,
            clip_on=False,
        )


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    fmt: str = "{:.2f}",
    fontsize: int = 8,
) -> None:
    """在热力图每个有效网格中写入数值。"""

    # 热力图里的数字直接写在格子中心，不额外加边框，避免出现多余线框。
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
            )


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """保存图片文件并关闭 figure。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.25)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return output_path


def dataframe_for_display(df: pd.DataFrame, formatters: dict[str, Any] | None = None) -> pd.DataFrame:
    """把表格内容转成适合渲染到 PNG 的字符串格式。"""

    if formatters is None:
        return df.astype(str)
    out = df.copy()
    for column in out.columns:
        formatter = formatters.get(column) if formatters else None
        if formatter is None:
            out[column] = out[column].astype(str)
            continue
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else formatter(value))
    return out


def save_table_png(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    formatters: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> Path:
    """把 DataFrame 渲染成表格 PNG。"""

    render_df = df.copy()
    if max_rows is not None:
        render_df = render_df.head(max_rows).copy()
    display_df = dataframe_for_display(render_df, formatters)
    n_rows, n_cols = display_df.shape
    fig_width = max(8.0, 1.2 * n_cols + 2.0)
    fig_height = max(2.8, 0.38 * (n_rows + 2) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("0.65")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("0.55")
            cell.set_text_props(weight="bold", color="white", ha="center")
        else:
            cell.set_facecolor("0.94" if row_idx % 2 == 1 else "0.86")
            if col_idx == 0:
                cell.set_text_props(ha="left")
            else:
                cell.set_text_props(ha="right")
    return save_figure(fig, output_path)
