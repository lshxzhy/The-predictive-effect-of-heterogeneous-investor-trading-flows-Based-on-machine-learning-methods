import matplotlib.pyplot as plt


MONO_BAR_COLOR = "#6b6b6b"
MONO_EDGE_COLOR = "#000000"
MONO_LIGHT_COLOR = "#d9d9d9"


# 全局统一使用黑白灰绘图风格，避免不同脚本各自保留默认彩色主题。
def configure_monochrome_matplotlib() -> None:
    """设置统一的黑白灰绘图参数。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.titlecolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["grid.color"] = MONO_LIGHT_COLOR
    plt.rcParams["image.cmap"] = "Greys"
