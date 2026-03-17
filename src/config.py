from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetSpec:
    """单个资产的基础配置。"""

    alias: str
    code: str
    sheet_name: str


@dataclass(frozen=True)
class PipelineConfig:
    """项目级配置。"""

    label_horizons: tuple[int, ...] = (1, 22, 33, 44, 55, 66, 77, 88, 99)
    screening_horizon: int = 1
    market_lag_horizons: tuple[int, ...] = (1, 5, 22)
    market_stat_horizons: tuple[int, ...] = (5, 22)
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    index_sheet_name: str = "中证能源行业指数932077"
    common_sample_horizon_value: int = 99
    normalized_cols: tuple[str, ...] = (
        "IND_SECTOR_TV_ene_norm",
        "INS_SECTOR_TV_ene_norm",
    )
    fixed_core_feature_cols: tuple[str, ...] = (
        "IND_SECTOR_TV_ene_norm",
        "INS_SECTOR_TV_ene_norm",
        "ITVvar",
    )
    supplementary_old_feature_cols: tuple[str, ...] = (
        "sigpre",
        "sigpre30",
        "dolsha30",
        "dolsha",
        "ITVvar_x_dolsha",
    )
    shared_control_cols: tuple[str, ...] = (
        "iVX",
        "cred",
        "liqu",
    )
    # 这里只保留真正进入最终特征层和统一筛选层的通用行情变量。
    # pct_chg 与自算的一日对数收益高度同质，当前主线不再保留它；
    # turn 需要一直保留到人工 --drop-cols 这一步，由用户决定是否删除。
    market_base_feature_cols: tuple[str, ...] = (
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
    )
    core_source_cols: tuple[str, ...] = (
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
    )
    # close 只作为原始价格输入，用于构造标签、单日收益、滚动波动和动量；
    # pct_chg 不再进入当前主线；turn 需要保留到统一筛选和人工删列阶段。
    market_source_cols: tuple[str, ...] = (
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
    )
    rate_source_cols: tuple[str, ...] = (
        "FirmBondAA10Y",
        "ChBond10Y",
        "ChBond3M",
        "R6M",
    )
    intermediate_only_source_cols: tuple[str, ...] = (
        "close",
        "FirmBondAA10Y",
        "ChBond10Y",
        "ChBond3M",
        "R6M",
    )
    raw_diagnostic_cols: tuple[str, ...] = ("turn",)
    interaction_feature_name: str = "ITVvar_x_dolsha"
    screening_id_cols: tuple[str, ...] = (
        "Date",
        "asset_alias",
        "asset_code",
    )
    search_metric: str = "roc_auc"

    # 在配置层先校验主线比例、期限集合和统一筛选期限，避免脚本运行到中途才暴露配置矛盾。
    def __post_init__(self) -> None:
        """校验配置本身是否完整。"""
        ratio_sum = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError("训练集、验证集、测试集比例之和必须等于 1。")
        if not self.label_horizons:
            raise ValueError("至少需要配置一个预测期限。")
        if any(h <= 0 for h in self.label_horizons):
            raise ValueError("label_horizons 里的期限必须全部为正整数。")
        if any(h <= 0 for h in self.market_lag_horizons):
            raise ValueError("market_lag_horizons 里的期限必须全部为正整数。")
        if any(h <= 0 for h in self.market_stat_horizons):
            raise ValueError("market_stat_horizons 里的期限必须全部为正整数。")
        if self.screening_horizon not in self.label_horizons:
            raise ValueError("screening_horizon 必须包含在 label_horizons 中。")
        if self.common_sample_horizon_value not in self.label_horizons:
            raise ValueError("common_sample_horizon_value 必须包含在 label_horizons 中。")

    # 统一维护“资产简称 -> 代码/原始 sheet 名”的映射，让指数和期货都走同一接口。
    @property
    def asset_specs(self) -> dict[str, AssetSpec]:
        """返回当前项目支持的资产映射。"""
        return {
            "index": AssetSpec(
                alias="index",
                code="932077.CSI",
                sheet_name=self.index_sheet_name,
            ),
            "sc": AssetSpec(alias="sc", code="SC00.INE", sheet_name="原油期货近月连续"),
            "lu": AssetSpec(alias="lu", code="LU00.INE", sheet_name="低硫燃料油近月连续"),
            "pp": AssetSpec(alias="pp", code="PP00.DCE", sheet_name="聚丙烯近月连续"),
            "eg": AssetSpec(alias="eg", code="EG00.DCE", sheet_name="乙二醇近月连续"),
            "bu": AssetSpec(alias="bu", code="BU00.SHF", sheet_name="沥青近月连续"),
            "jm": AssetSpec(alias="jm", code="JM00.DCE", sheet_name="焦煤近月连续"),
        }

    # 所有资产脚本都通过简称取配置，缺失简称时立即报错而不是静默回退。
    def get_asset_spec(self, asset_alias: str) -> AssetSpec:
        """按简称获取资产配置。"""
        if asset_alias not in self.asset_specs:
            raise KeyError(f"未配置资产简称：{asset_alias}")
        return self.asset_specs[asset_alias]

    # 期限在配置层统一校验，避免不同脚本各自接受不一致的 horizon 集合。
    def validate_horizon(self, horizon: int) -> int:
        """校验单个期限。"""
        if horizon not in self.label_horizons:
            raise ValueError(f"当前未配置 {horizon}d，请先在 label_horizons 中补充。")
        return horizon

    def horizon_dir_name(self, horizon: int) -> str:
        """返回期限目录名。"""
        self.validate_horizon(horizon)
        return f"{horizon}d"

    def scheme_name(self, asset_alias: str, horizon: int) -> str:
        """返回单资产单期限训练方案名。"""
        self.validate_horizon(horizon)
        return f"{asset_alias}_{horizon}d"

    def target_return_col(self, horizon: int) -> str:
        """返回未来累计收益列名。"""
        self.validate_horizon(horizon)
        return f"target_return_{horizon}d"

    def target_label_col(self, horizon: int) -> str:
        """返回未来收益方向标签列名。"""
        self.validate_horizon(horizon)
        return f"target_label_{horizon}d"

    def common_sample_horizon(self) -> int:
        """返回统一裁样本所用的固定期限。"""
        return self.common_sample_horizon_value

    def selected_feature_source_horizon(self) -> int:
        """返回统一变量筛选所用的固定期限。"""
        return self.screening_horizon

    def screening_scheme_name(self, horizon: int) -> str:
        """返回统一筛选模型方案名。"""
        self.validate_horizon(horizon)
        return f"screening_{horizon}d"

    def market_log_return_col(self, horizon: int) -> str:
        """返回单日对数收益滞后变量列名。"""
        if horizon not in self.market_lag_horizons:
            raise ValueError(f"未配置 market_lag_horizons={self.market_lag_horizons} 之外的期限：{horizon}")
        return f"log_return_lag{horizon}"

    def market_log_return_abs_col(self, horizon: int) -> str:
        """返回单日绝对对数收益滞后变量列名。"""
        if horizon not in self.market_lag_horizons:
            raise ValueError(f"未配置 market_lag_horizons={self.market_lag_horizons} 之外的期限：{horizon}")
        return f"log_return_abs_lag{horizon}"

    def market_vol_col(self, horizon: int) -> str:
        """返回滚动波动率列名。"""
        if horizon not in self.market_stat_horizons:
            raise ValueError(f"未配置 market_stat_horizons={self.market_stat_horizons} 之外的期限：{horizon}")
        return f"vol_h{horizon}d"

    def market_mom_col(self, horizon: int) -> str:
        """返回滚动动量列名。"""
        if horizon not in self.market_stat_horizons:
            raise ValueError(f"未配置 market_stat_horizons={self.market_stat_horizons} 之外的期限：{horizon}")
        return f"mom_h{horizon}d"

    # log_return_lag* 和 log_return_abs_lag* 是“过去某一日的一日收益”，
    # 在 Date=t 上以 t 收盘后已知信息对齐，因此 lag1 不再额外平移。
    def market_lag_shift_steps(self, horizon: int) -> int:
        """返回单日收益滞后变量在 Date=t 上需要平移的行数。"""
        if horizon not in self.market_lag_horizons:
            raise ValueError(f"未配置 market_lag_horizons={self.market_lag_horizons} 之外的期限：{horizon}")
        return horizon - 1

    def all_label_columns(self) -> list[str]:
        """返回全部目标列。"""
        label_cols: list[str] = []
        for horizon in self.label_horizons:
            label_cols.extend(
                [
                    self.target_return_col(horizon),
                    self.target_label_col(horizon),
                ]
            )
        return label_cols

    # merged_raw_panel 进入 feature_panel 前必须具备这些源字段。
    def feature_input_required_cols(self) -> list[str]:
        """返回特征层必需具备的输入列。"""
        cols = [
            *self.screening_id_cols,
            *self.fixed_core_feature_cols,
            "sigpre",
            "sigpre30",
            "dolsha30",
            "dolsha",
            "iVX",
            *self.rate_source_cols,
            *self.market_source_cols,
        ]
        return list(dict.fromkeys(cols))

    # close 和利率原始列只用于派生特征和标签，进入 feature_panel 前必须删除。
    # turn 不在这里删除，它要继续保留到统一筛选和手动删列阶段。
    def feature_intermediate_drop_cols(self) -> list[str]:
        """返回派生完成后必须从最终特征层删除的中间列。"""
        return list(dict.fromkeys(self.intermediate_only_source_cols))

    # 这组是无需按资产展开的公共补充候选变量。
    def shared_screening_supplementary_cols(self) -> list[str]:
        """返回不需要按资产展开的补充控制变量。"""
        return list(
            dict.fromkeys(
                [
                    *self.supplementary_old_feature_cols,
                    *self.shared_control_cols,
                ]
            )
        )

    # 这组是进入最终 feature_panel 的行情变量与派生变量，不再包含 close 原始列。
    def generic_market_feature_cols(self) -> list[str]:
        """返回统一筛选使用的行情与派生特征列。"""
        cols: list[str] = [*self.market_base_feature_cols]
        for horizon in self.market_lag_horizons:
            cols.extend(
                [
                    self.market_log_return_col(horizon),
                    self.market_log_return_abs_col(horizon),
                ]
            )
        for horizon in self.market_stat_horizons:
            cols.extend(
                [
                    self.market_vol_col(horizon),
                    self.market_mom_col(horizon),
                ]
            )
        return list(dict.fromkeys(cols))

    # screening_ready_panel 里的候选变量由“公共补充变量 + 行情最终特征列”组成。
    def screening_ready_supplementary_cols(self) -> list[str]:
        """返回资产级筛选输入层使用的补充候选变量。"""
        return list(
            dict.fromkeys(
                [
                    *self.shared_screening_supplementary_cols(),
                    *self.generic_market_feature_cols(),
                ]
            )
        )

    # screening_long_panel 的候选列集合与单资产 screening_ready_panel 保持完全一致。
    def all_screening_supplementary_cols(self) -> list[str]:
        """返回统一筛选长面板使用的候选列。"""
        return self.screening_ready_supplementary_cols()

    # 资产级筛选输入会把固定保留的 3 列和统一筛选候选变量一起带上，
    # 其中前 3 列只保留、不参与统一筛选淘汰。
    def screening_feature_cols(self) -> list[str]:
        """返回资产级筛选输入层使用的全部特征列。"""
        return list(
            dict.fromkeys(
                [
                    *self.fixed_core_feature_cols,
                    *self.screening_ready_supplementary_cols(),
                ]
            )
        )


@dataclass(frozen=True)
class ProjectPaths:
    """项目路径集合。"""

    project_root: Path
    raw_dir: Path
    processed_root: Path
    asset_processed_dir: Path | None
    horizon_processed_dir: Path | None
    screening_root_dir: Path
    core_raw_file: Path
    extra_raw_file: Path
    models_root: Path
    outputs_root: Path

    def _require_asset_dir(self) -> Path:
        """返回资产级目录。"""
        if self.asset_processed_dir is None:
            raise ValueError("当前上下文未配置资产目录。")
        return self.asset_processed_dir

    def _require_horizon_dir(self) -> Path:
        """返回单期限目录。"""
        if self.horizon_processed_dir is None:
            raise ValueError("当前上下文未配置单期限目录。")
        return self.horizon_processed_dir

    def ensure_base_dirs(self) -> None:
        """创建项目级基础目录。"""
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.screening_root_dir.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def ensure_asset_dirs(self) -> None:
        """创建资产级目录。"""
        self.ensure_base_dirs()
        self._require_asset_dir().mkdir(parents=True, exist_ok=True)

    def ensure_screening_data_dir(self) -> None:
        """创建统一筛选数据目录。"""
        self.ensure_base_dirs()
        self.screening_root_dir.mkdir(parents=True, exist_ok=True)

    def ensure_horizon_dirs(self) -> None:
        """创建单期限目录。"""
        self.ensure_asset_dirs()
        self._require_horizon_dir().mkdir(parents=True, exist_ok=True)

    def merged_raw_panel_file(self) -> Path:
        """返回原始合并层文件。"""
        return self._require_asset_dir() / "merged_raw_panel.csv"

    def raw_missing_stats_file(self) -> Path:
        """返回原始缺失统计文件。"""
        return self._require_asset_dir() / "raw_missing_stats.csv"

    def feature_panel_file(self) -> Path:
        """返回特征层文件。"""
        return self._require_asset_dir() / "feature_panel.csv"

    def feature_desc_stats_file(self) -> Path:
        """返回特征层描述统计文件。"""
        return self._require_asset_dir() / "feature_desc_stats.csv"

    def screening_ready_panel_file(self) -> Path:
        """返回资产级筛选输入文件。"""
        return self._require_asset_dir() / "screening_ready_panel.csv"

    def screening_long_panel_file(self) -> Path:
        """返回统一筛选长面板文件。"""
        return self.screening_root_dir / "screening_long_panel.csv"

    def selected_features_file(self) -> Path:
        """返回统一筛选结果文件。"""
        return self.screening_root_dir / "selected_features.csv"

    def screening_output_dir(self) -> Path:
        """返回统一筛选分析结果目录。"""
        return self.outputs_root / "screening"

    def screening_output_file(self, filename: str) -> Path:
        """返回统一筛选分析结果文件。"""
        return self.screening_output_dir() / filename

    def prepared_file(self, split_name: str) -> Path:
        """返回单期限准备层文件。"""
        return self._require_horizon_dir() / f"{split_name}_prepared.csv"

    def model_input_file(self, split_name: str) -> Path:
        """返回单期限模型输入文件。"""
        return self._require_horizon_dir() / f"{split_name}_model_input.csv"

    def model_dir(self, model_id: str, scheme_name: str) -> Path:
        """返回模型目录。"""
        return self.models_root / model_id / scheme_name

    def model_file(self, model_id: str, scheme_name: str) -> Path:
        """返回模型文件。"""
        return self.model_dir(model_id, scheme_name) / f"{model_id}.joblib"

    def output_dir(self, model_id: str, scheme_name: str) -> Path:
        """返回结果目录。"""
        return self.outputs_root / model_id / scheme_name

    def output_file(self, model_id: str, scheme_name: str, filename: str) -> Path:
        """返回结果文件。"""
        return self.output_dir(model_id, scheme_name) / filename

    def ensure_model_dirs(self, model_id: str, scheme_name: str) -> None:
        """创建模型目录和结果目录。"""
        self.model_dir(model_id, scheme_name).mkdir(parents=True, exist_ok=True)
        self.output_dir(model_id, scheme_name).mkdir(parents=True, exist_ok=True)

    def ensure_screening_output_dir(self) -> None:
        """创建统一筛选分析结果目录。"""
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.screening_output_dir().mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AssetRuntimeContext:
    """资产级运行时上下文。"""

    config: PipelineConfig
    asset_spec: AssetSpec
    asset_alias: str
    paths: ProjectPaths


@dataclass(frozen=True)
class HorizonRuntimeContext:
    """期限级运行时上下文。"""

    config: PipelineConfig
    horizon: int
    paths: ProjectPaths


@dataclass(frozen=True)
class RuntimeContext:
    """单资产单期限建模上下文。"""

    config: PipelineConfig
    asset_spec: AssetSpec
    asset_alias: str
    horizon: int
    scheme_name: str
    paths: ProjectPaths


# 如果脚本未显式传入 project_root，就默认以 src/ 的上一级作为项目根目录。
def resolve_project_root(project_root: Path | None = None) -> Path:
    """解析项目根目录。"""
    if project_root is None:
        return Path(__file__).resolve().parent.parent
    return project_root


# 所有路径对象都在这里一次构造好，让上游脚本只处理“当前资产/期限对应的标准文件位置”。
def build_project_paths(
    project_root: Path,
    asset_alias: str | None = None,
    horizon: int | None = None,
) -> ProjectPaths:
    """构造项目路径集合。"""
    asset_processed_dir = None
    horizon_processed_dir = None

    if asset_alias is not None:
        asset_processed_dir = project_root / "data" / "processed" / asset_alias

    if asset_processed_dir is not None and horizon is not None:
        horizon_processed_dir = asset_processed_dir / "horizons" / f"{horizon}d"

    return ProjectPaths(
        project_root=project_root,
        raw_dir=project_root / "data" / "raw",
        processed_root=project_root / "data" / "processed",
        asset_processed_dir=asset_processed_dir,
        horizon_processed_dir=horizon_processed_dir,
        screening_root_dir=project_root / "data" / "processed" / "screening",
        core_raw_file=project_root / "data" / "raw" / "核心变量时间序列日度.xlsx",
        extra_raw_file=project_root / "data" / "raw" / "指标数据补充.xlsx",
        models_root=project_root / "models",
        outputs_root=project_root / "outputs",
    )


# 资产级上下文负责把配置、资产规格和资产目录打包起来，供单资产主线脚本复用。
def get_asset_context(
    project_root: Path | None = None,
    asset_alias: str | None = None,
) -> AssetRuntimeContext:
    """返回资产级上下文。"""
    if asset_alias is None:
        raise ValueError("必须显式提供 --asset。")

    resolved_root = resolve_project_root(project_root)
    config = PipelineConfig()
    asset_spec = config.get_asset_spec(asset_alias)
    paths = build_project_paths(resolved_root, asset_alias=asset_alias)

    return AssetRuntimeContext(
        config=config,
        asset_spec=asset_spec,
        asset_alias=asset_alias,
        paths=paths,
    )


# 期限级上下文服务于统一筛选这类“只认 horizon、不认单个资产”的脚本。
def get_horizon_context(
    project_root: Path | None = None,
    horizon: int | None = None,
) -> HorizonRuntimeContext:
    """返回期限级上下文。"""
    if horizon is None:
        raise ValueError("必须显式提供 --horizon。")

    resolved_root = resolve_project_root(project_root)
    config = PipelineConfig()
    config.validate_horizon(horizon)
    paths = build_project_paths(resolved_root)

    return HorizonRuntimeContext(
        config=config,
        horizon=horizon,
        paths=paths,
    )


# 运行时上下文面向单资产单期限建模，把 asset 和 horizon 两套约束同时绑定起来。
def get_runtime_context(
    project_root: Path | None = None,
    asset_alias: str | None = None,
    horizon: int | None = None,
) -> RuntimeContext:
    """返回单资产单期限建模上下文。"""
    if asset_alias is None:
        raise ValueError("必须显式提供 --asset。")
    if horizon is None:
        raise ValueError("必须显式提供 --horizon。")

    resolved_root = resolve_project_root(project_root)
    config = PipelineConfig()
    asset_spec = config.get_asset_spec(asset_alias)
    config.validate_horizon(horizon)
    scheme_name = config.scheme_name(asset_alias, horizon)
    paths = build_project_paths(
        resolved_root,
        asset_alias=asset_alias,
        horizon=horizon,
    )

    return RuntimeContext(
        config=config,
        asset_spec=asset_spec,
        asset_alias=asset_alias,
        horizon=horizon,
        scheme_name=scheme_name,
        paths=paths,
    )
