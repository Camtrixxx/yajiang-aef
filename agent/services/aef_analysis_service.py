from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener

from agent.config import AEFConfig, ReportConfig
from agent.schemas.report import AnalysisResult, ChartAsset, MetricCard, ReportRequest


TASK_TO_AEF = {
    "地物分类": "landcover",
    "水体分布": "water",
    "水体分类": "water",
    "高程地形": "dem",
    "高程重建": "dem",
    "地形分析": "dem",
}

TASK_DISPLAY = {
    "landcover": "地物分类",
    "water": "水体分类",
    "dem": "高程地形",
}

LANDCOVER_PATCH_POOL = [0, 141, 300, 400, 431, 612, 766, 960, 1230, 1289, 1438, 1607]
DEM_PATCH_POOL = [0, 141, 300, 400, 431, 612, 766, 960, 1230, 1289, 1438, 1607]
WATER_PATCH_POOL = [1438, 431, 141, 1289, 62, 300, 22, 101, 1448, 339, 220, 961]


def _finite_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "暂无"
    return f"{value * 100:.{digits}f}%"


def _finite_number(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "暂无"
    return f"{value:.{digits}f}{suffix}"


def _quarter_from_month(time_range: str) -> str:
    try:
        month = int(time_range.split("-", 1)[1])
    except (IndexError, ValueError):
        return "latest"
    quarter = (month - 1) // 3 + 1
    return f"{time_range[:4]}Q{quarter}"


def _stable_pick(pool: list[int], key: str, count: int) -> list[int]:
    count = max(1, min(count, len(pool)))
    ranked = sorted(
        pool,
        key=lambda item: hashlib.sha1(f"{key}:{item}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


class MockPatchSelector:
    """Temporary deterministic region/time -> patch mapping.

    The selector is intentionally isolated so real AOI-to-patch lookup can
    replace it without touching the agent graph or report writer.
    """

    def select(self, request: ReportRequest, aef_task: str, count: int) -> list[int]:
        if aef_task == "water":
            pool = WATER_PATCH_POOL
        elif aef_task == "dem":
            pool = DEM_PATCH_POOL
        else:
            pool = LANDCOVER_PATCH_POOL
        return _stable_pick(pool, f"{request.region}-{request.time_range}-{aef_task}", count)


class AEFAnalysisService:
    """Analysis service backed by the external AEF inference API."""

    def __init__(
        self,
        config: AEFConfig | None = None,
        report_config: ReportConfig | None = None,
        patch_selector: MockPatchSelector | None = None,
    ) -> None:
        self.config = config or AEFConfig()
        self.report_config = report_config or ReportConfig()
        self.asset_dir = self.report_config.asset_dir
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.patch_selector = patch_selector or MockPatchSelector()
        self.opener = build_opener(ProxyHandler({}))

    def analyze(self, request: ReportRequest) -> AnalysisResult:
        aef_task = self._normalize_task(request.task)
        rgb_period = _quarter_from_month(request.time_range)
        sample_indices = self.patch_selector.select(
            request,
            aef_task=aef_task,
            count=self.config.sample_count,
        )
        payload = self._infer(
            sample_indices=sample_indices,
            task=aef_task,
            rgb_period=rgb_period,
        )
        model = payload.get("model") or {}
        model_path = str(model.get("deploy_model_path") or "")
        model_name = Path(model_path).stem or "aef_model"
        summary = payload.get("summary") or {}
        items = payload.get("items") or []
        charts = self._build_charts(aef_task, payload)
        metrics = self._build_metrics(request, aef_task, model_name, summary, items)
        findings = self._build_findings(request, aef_task, summary, items, model_name)
        recommendations = self._build_recommendations(aef_task)
        narrative_blocks = self._build_narratives(request, aef_task, summary)
        risks = self._build_risks(aef_task)
        method_notes = [
            f"Agent 已将用户需求标准化为 task={aef_task}、region={request.region}、time_range={request.time_range}。",
            f"区域到 patch 的映射当前由临时选择器完成，本次选中样本为 {sample_indices}；后续可替换为真实 AOI 检索服务。",
            f"模型推理来自外部 AEF 服务 {self.config.base_url}，模型文件为 {model_path}。",
        ]
        confidence_notes = self._build_confidence_notes(aef_task, summary, items)
        limitations = [
            "当前已经接入真实 AEF 推理结果，但区域到 patch 的选择仍为临时映射，尚不代表完整 AOI 覆盖。",
            "当前报告以 patch 级样本验证端到端闭环，正式区域报告需要补充 AOI 边界、patch 覆盖率和多样本汇总策略。",
        ]
        task_display = TASK_DISPLAY.get(aef_task, request.task)
        return AnalysisResult(
            task=task_display,
            region=request.region,
            time_range=request.time_range,
            headline=f"{request.region}{request.time_range}{task_display}遥感分析",
            summary=self._summary_text(request, aef_task, summary),
            metrics=metrics,
            findings=findings,
            recommendations=recommendations,
            narrative_blocks=narrative_blocks,
            risks=risks,
            method_notes=method_notes,
            limitations=limitations,
            confidence_notes=confidence_notes,
            data_source="aef_inference",
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            aef_payload={
                "service": self.config.base_url,
                "model": model,
                "task": aef_task,
                "region": request.region,
                "time_range": request.time_range,
                "rgb_source": self.config.rgb_source,
                "rgb_period": rgb_period,
                "sample_indices": sample_indices,
                "selector": "temporary_deterministic_patch_selector",
                "summary": summary,
                "fingerprint": self._fingerprint(payload),
            },
            charts=charts,
        )

    def _normalize_task(self, task: str) -> str:
        return TASK_TO_AEF.get(task, "landcover")

    def _infer(self, *, sample_indices: list[int], task: str, rgb_period: str) -> dict[str, Any]:
        url = urljoin(self.config.base_url.rstrip("/") + "/", "api/infer")
        payload = {
            "sample_indices": sample_indices,
            "task": task,
            "use_cache": True,
            "water_threshold": 0.5,
            "rgb_source": self.config.rgb_source,
            "rgb_period": rgb_period,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"AEF 推理服务调用失败：{url}，原因：{exc}") from exc

    def _build_charts(self, aef_task: str, payload: dict[str, Any]) -> list[ChartAsset]:
        items = payload.get("items") or []
        if not items:
            return []
        first = items[0]
        artifacts = first.get("artifacts") or {}
        if aef_task == "landcover":
            specs = [
                ("landcover_compare_png", "地物分类真值与推理对比", "展示地物分类真值、模型预测、正确/错误区域和置信度。"),
                ("landcover_overlay_png", "地物分类叠加图", "将地物分类结果叠加到原始遥感 patch 上，用于观察空间分布。"),
            ]
        elif aef_task == "water":
            specs = [
                ("water_compare_png", "水体分类真值与推理对比", "展示水体真值、模型预测、正确/错误区域和水体概率。"),
                ("water_overlay_png", "水体叠加图", "将水体概率和阈值结果叠加到原始遥感 patch 上。"),
            ]
        else:
            specs = [
                ("dem_terrain_overview_png", "高程地形分析总览", "展示原始影像、地形阴影、高程分区、坡度强度和剖面曲线。"),
                ("dem_compare_png", "高程重建验证图", "展示 DEM 真值、预测重建结果和绝对误差。"),
            ]
        charts = []
        for key, title, caption in specs:
            url = artifacts.get(key)
            if not url:
                continue
            local_url = self._copy_artifact(url, payload, key)
            charts.append(ChartAsset(title=title, kind="image", url=local_url, caption=caption))
        return charts

    def _copy_artifact(self, artifact_url: str, payload: dict[str, Any], key: str) -> str:
        source_url = urljoin(self.config.base_url.rstrip("/") + "/", artifact_url.lstrip("/"))
        model_path = str((payload.get("model") or {}).get("deploy_model_path") or "")
        digest = hashlib.sha1(f"{model_path}-{source_url}-{key}".encode("utf-8")).hexdigest()[:12]
        suffix = Path(artifact_url).suffix or ".png"
        out_path = self.asset_dir / f"aef_{key}_{digest}{suffix}"
        if not out_path.exists():
            try:
                with self.opener.open(source_url, timeout=self.config.timeout) as response, out_path.open("wb") as fh:
                    shutil.copyfileobj(response, fh)
            except OSError as exc:
                raise RuntimeError(f"AEF 图像下载失败：{source_url}，原因：{exc}") from exc
        return f"/reports/assets/{out_path.name}"

    def _build_metrics(
        self,
        request: ReportRequest,
        aef_task: str,
        model_name: str,
        summary: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> list[MetricCard]:
        cards = [
            MetricCard("任务", TASK_DISPLAY.get(aef_task, request.task), "Agent 识别后的标准任务"),
            MetricCard("地区", request.region, "前端选择或意图解析得到的目标区域"),
            MetricCard("时间", request.time_range, "用户指定的分析月份"),
            MetricCard("模型", model_name, "实际调用的 AEF 模型版本"),
            MetricCard("样本数", str(len(items)), "本次临时 patch 选择器返回的样本数量"),
        ]
        if aef_task == "landcover":
            dominant = summary.get("dominant_landcover") or {}
            cards.extend(
                [
                    MetricCard("总体精度", _finite_percent(summary.get("landcover_overall_accuracy_mean")), "与 WorldCover 真值对比得到的平均精度"),
                    MetricCard("平均置信度", _finite_percent(summary.get("landcover_mean_confidence")), "模型 top-1 分类概率均值"),
                    MetricCard("低置信比例", _finite_percent(summary.get("landcover_low_confidence_ratio_mean")), "top-1 置信度低于阈值的像元比例"),
                    MetricCard("主导类别", str(dominant.get("label_zh") or dominant.get("label") or "暂无"), "预测占比最高的地物类型"),
                ]
            )
        elif aef_task == "water":
            cards.extend(
                [
                    MetricCard("水体 F1", _finite_percent(summary.get("water_f1_mean")), "水体分类综合精度指标"),
                    MetricCard("水体 IoU", _finite_percent(summary.get("water_iou_mean")), "预测水体与真值水体的交并比"),
                    MetricCard("召回率", _finite_percent(summary.get("water_accuracy_mean")), "当前水体真值区域内被识别出的比例"),
                    MetricCard("预测水体占比", _finite_percent(summary.get("water_ratio_mean")), "模型预测为水体的 patch 像元比例"),
                ]
            )
        else:
            cards.extend(
                [
                    MetricCard("平均高程", _finite_number(summary.get("dem_pred_mean"), 1, " m"), "模型重建 DEM 的平均高程"),
                    MetricCard("MAE", _finite_number(summary.get("dem_mae_mean"), 1, " m"), "预测高程与真值高程的平均绝对误差"),
                    MetricCard("R²", _finite_number(summary.get("dem_r2_mean"), 3), "高程重建拟合程度"),
                    MetricCard("地形起伏", _finite_number(summary.get("dem_terrain_relief_mean"), 1, " m"), "patch 内预测最高与最低高程差"),
                ]
            )
        return cards

    def _summary_text(self, request: ReportRequest, aef_task: str, summary: dict[str, Any]) -> str:
        if aef_task == "landcover":
            return (
                f"本次调用真实 AEF 模型完成 {request.region} {request.time_range} 的地物分类分析，"
                f"平均精度为 {_finite_percent(summary.get('landcover_overall_accuracy_mean'))}，"
                f"平均置信度为 {_finite_percent(summary.get('landcover_mean_confidence'))}。"
            )
        if aef_task == "water":
            return (
                f"本次调用真实 AEF 模型完成 {request.region} {request.time_range} 的水体分类分析，"
                f"水体 F1 为 {_finite_percent(summary.get('water_f1_mean'))}，"
                f"IoU 为 {_finite_percent(summary.get('water_iou_mean'))}。"
            )
        return (
            f"本次调用真实 AEF 模型完成 {request.region} {request.time_range} 的高程地形分析，"
            f"平均高程约 {_finite_number(summary.get('dem_pred_mean'), 1, ' m')}，"
            f"MAE 为 {_finite_number(summary.get('dem_mae_mean'), 1, ' m')}。"
        )

    def _build_findings(
        self,
        request: ReportRequest,
        aef_task: str,
        summary: dict[str, Any],
        items: list[dict[str, Any]],
        model_name: str,
    ) -> list[str]:
        if aef_task == "landcover":
            dominant = summary.get("dominant_landcover") or {}
            label = dominant.get("label_zh") or dominant.get("label") or "主要类别"
            return [
                f"本次分析由 {model_name} 生成地物分类结果，主导地物类型为 {label}，可作为当前区域专题解读的核心对象。",
                f"平均分类精度为 {_finite_percent(summary.get('landcover_overall_accuracy_mean'))}，平均置信度为 {_finite_percent(summary.get('landcover_mean_confidence'))}，说明当前 patch 结果具备初步解释价值。",
                "分类对比图同时展示真值、预测、正确/错误区域和置信度，可用于快速定位模型误差集中区域。",
            ]
        if aef_task == "water":
            return [
                f"本次水体分类由 {model_name} 完成，平均 F1 为 {_finite_percent(summary.get('water_f1_mean'))}，IoU 为 {_finite_percent(summary.get('water_iou_mean'))}。",
                f"模型预测水体占比约为 {_finite_percent(summary.get('water_ratio_mean'))}，可用于观察河谷、湖面或湿地区域的空间连通性。",
                "水体概率图保留了阈值前的连续概率信息，比单一 mask 更适合解释边界不确定性。",
            ]
        return [
            f"本次高程地形分析由 {model_name} 完成，平均高程约 {_finite_number(summary.get('dem_pred_mean'), 1, ' m')}。",
            f"高程重建 MAE 为 {_finite_number(summary.get('dem_mae_mean'), 1, ' m')}，R² 为 {_finite_number(summary.get('dem_r2_mean'), 3)}，可用于判断当前 patch 的重建可靠性。",
            "地形总览图将阴影、高程分区、坡度强度和剖面曲线组合展示，比直接查看 DEM 色带更适合面向用户解释。",
        ]

    def _build_recommendations(self, aef_task: str) -> list[str]:
        common = [
            "后续应将临时 patch 选择器替换为真实 AOI 到 patch 的空间检索服务，并输出覆盖率。",
            "建议在正式报告中保留模型版本、样本编号、时相和图像产物，便于复核与追溯。",
        ]
        if aef_task == "landcover":
            return ["对低置信度区域进行人工抽样复核，优先检查类型交界带和阴影区域。", *common]
        if aef_task == "water":
            return ["对水体边界区建议结合多时相影像复核，避免季节性水位变化造成误判。", *common]
        return ["高程产品建议重点关注误差图和坡度强度区，避免把局部重建误差直接解释为真实地形突变。", *common]

    def _build_narratives(self, request: ReportRequest, aef_task: str, summary: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "title": "真实 AEF 调用",
                "text": f"Agent 已把用户输入转换为标准化任务 {aef_task}，并调用 AEF 推理服务生成指标和图像产物。",
            },
            {
                "title": "空间样本说明",
                "text": f"当前 {request.region} 的 patch 选择仍为临时映射，用于验证从自然语言到模型推理再到报告生成的完整链路。",
            },
        ]

    def _build_risks(self, aef_task: str) -> list[str]:
        if aef_task == "landcover":
            return ["地物分类结果在类别边界和混合像元区域更容易出现误差，应结合置信度图一起解读。"]
        if aef_task == "water":
            return ["当前水体真值有效区域有限，水体分类结果更适合做闭环验证和局部观察，正式使用前需要更完整的负样本评估。"]
        return ["高程重建误差可能在陡坡、阴影和纹理复杂区域放大，业务解释时应同时查看误差图。"]

    def _build_confidence_notes(self, aef_task: str, summary: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
        if aef_task == "landcover":
            return [
                f"平均置信度为 {_finite_percent(summary.get('landcover_mean_confidence'))}，低置信度比例为 {_finite_percent(summary.get('landcover_low_confidence_ratio_mean'))}。",
                "绿色/红色正确性图用于直观判断预测与真值的一致性。",
            ]
        if aef_task == "water":
            return [
                f"水体 F1 为 {_finite_percent(summary.get('water_f1_mean'))}，IoU 为 {_finite_percent(summary.get('water_iou_mean'))}。",
                "水体概率图展示连续概率，阈值 mask 展示最终二分类结果。",
            ]
        return [
            f"高程重建 MAE 为 {_finite_number(summary.get('dem_mae_mean'), 1, ' m')}，RMSE 为 {_finite_number(summary.get('dem_rmse_mean'), 1, ' m')}。",
            "地形展示图经过轻微平滑以便用户阅读，模型验证仍以真值/预测/误差图为准。",
        ]

    def _fingerprint(self, payload: dict[str, Any]) -> str:
        compact = json.dumps(
            {
                "task": payload.get("task"),
                "sample_indices": payload.get("sample_indices"),
                "model": payload.get("model"),
                "summary": payload.get("summary"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:16]
