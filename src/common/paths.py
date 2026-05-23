from __future__ import annotations

# 本模块负责：统一封装“写入文件前先确保目录存在”的固定动作。
# 直接服务的步骤：第 01 到第 12 步中所有需要写目录或写文件的步骤。
# 关键修改位置：`ensure_dir()` 和 `ensure_parent()` 的目录创建规则。
# 变更后重跑起点：不单独决定重跑起点；仅在目录创建策略变化时随调用它的步骤一起验证。

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回目录路径本身。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    """确保目标文件的父目录存在，并返回原始文件路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
