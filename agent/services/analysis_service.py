from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from agent.schemas.report import AnalysisResult, ChartAsset, MetricCard, ReportRequest


def _configure_fonts() -> bool:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        font_path = Path(path)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


HAS_CJK_FONT = _configure_fonts()


REGION_PROFILES = {
    "雅江区域": {
        "terrain": "山地河谷",
        "main": "林地/草地",
        "summary": "区域内林地、草地和裸地占比较高，水体沿河谷呈线状分布。",
        "detail": "高海拔地形和河谷切割使得林地、草地、裸地形成明显的垂向分异，水体沿谷底与支流延展。",
        "values": [42, 28, 14, 9, 7],
        "labels": ["林地", "草地", "裸地", "水体", "建设用地"],
        "chart_labels": ["Forest", "Grass", "Bare", "Water", "Built-up"],
    },
    "哈尔滨区域": {
        "terrain": "平原城市与农田",
        "main": "耕地/水体",
        "summary": "耕地和建设用地分布明显，水体沿松花江及支流展开。",
        "detail": "农田网格与城市扩张带构成较强的人类活动特征，水体与道路/堤岸关系清晰。",
        "values": [36, 24, 18, 15, 7],
        "labels": ["耕地", "建设用地", "林草地", "水体", "裸地"],
        "chart_labels": ["Cropland", "Built-up", "Vegetation", "Water", "Bare"],
    },
    "北京市海淀区": {
        "terrain": "城市建设区",
        "main": "建设用地",
        "summary": "建设用地占比较高，绿地分布在西部山前地带和公园片区。",
        "detail": "建设用地与绿地交织，水体多呈零散斑块，适合对城市边缘区进行精细识别。",
        "values": [48, 21, 16, 9, 6],
        "labels": ["建设用地", "绿地", "林地", "水体", "裸地"],
        "chart_labels": ["Built-up", "Green", "Forest", "Water", "Bare"],
    },
}


class MockAnalysisService:
    """Deterministic placeholder for the future AEF analysis service."""

    def __init__(self, asset_dir: str | Path = "agent/reports/assets") -> None:
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, request: ReportRequest) -> AnalysisResult:
        profile = REGION_PROFILES.get(request.region, REGION_PROFILES["雅江区域"])
        chart = self._build_landcover_chart(request, profile)
        aef_payload = self._build_aef_payload(request, profile)
        confidence = self._score(request.region, request.task, low=0.78, high=0.91)
        valid_pixels = self._score(request.prompt, request.region, low=94.2, high=98.7)
        coverage = self._score(request.time_range, request.region, low=0.81, high=0.96)
        stability = self._score(request.prompt, request.time_range, low=0.72, high=0.93)

        metrics = [
            MetricCard("任务", request.task, "本次报告的主分析方向"),
            MetricCard("地区", request.region, "前端选择或意图解析得到的目标区域"),
            MetricCard("时间", request.time_range, "用户指定的分析月份"),
            MetricCard("主导类型", profile["main"], "当前区域中最突出的地表类型"),
            MetricCard("平均置信度", f"{confidence:.2f}", "AEF 推理后可替换为模型统计"),
            MetricCard("有效像元", f"{valid_pixels:.1f}%", "覆盖有效像元比例"),
            MetricCard("时相覆盖", f"{coverage:.2f}", "时间窗与可用影像匹配程度"),
            MetricCard("结果稳定性", f"{stability:.2f}", "跨候选时间窗的结构稳定性"),
        ]

        findings = [
            f"{request.region}整体呈现{profile['terrain']}特征，{profile['summary']}",
            f"围绕“{request.task}”任务，当前结果显示{profile['main']}是最需要优先解释的空间类型。",
            f"空间格局上，{profile['detail']}",
            "指标层面显示有效像元、覆盖程度和稳定性处于可解释区间，适合继续做专题深挖。",
        ]
        recommendations = [
            "正式报告应增加 AOI 边界、时间窗口、数据源清单和模型版本说明。",
            "若用于业务交付，建议输出 HTML 与 PDF 两种格式，并保存结构化 JSON 便于复核。",
            "若面向业务决策，建议在结论页后追加一页“行动建议”，把发现转成具体处置项。",
        ]

        narrative_blocks = [
            {
                "title": "执行摘要",
                "text": f"{request.region}{request.time_range}的{request.task}分析显示，{profile['main']}为主要对象，"
                f"区域空间格局与地貌/人类活动特征耦合明显。{profile['summary']}",
            },
            {
                "title": "空间解读",
                "text": f"{profile['detail']} 这意味着在后续接入真实 AEF 模型时，可重点观察水体边界、"
                "植被覆盖层次和建设用地外扩边界的变化。",
            },
            {
                "title": "交付建议",
                "text": "正式报告建议使用统一版式：标题页、执行摘要、关键指标、图表页、专题发现、方法说明和附录，"
                "这样更适合交付和复核。",
            },
        ]

        return AnalysisResult(
            task=request.task,
            region=request.region,
            time_range=request.time_range,
            headline=f"{request.region}{request.time_range}{request.task}遥感分析",
            summary=f"本次分析聚焦{request.region}在{request.time_range}期间的{request.task}情况。{profile['summary']}",
            metrics=metrics,
            findings=findings,
            recommendations=recommendations,
            narrative_blocks=narrative_blocks,
            aef_payload=aef_payload,
            charts=[chart],
        )

    def _build_landcover_chart(self, request: ReportRequest, profile: dict) -> ChartAsset:
        slug = hashlib.sha1(
            f"{request.region}-{request.task}-{request.time_range}-chart-v2".encode("utf-8")
        ).hexdigest()[:12]
        out_path = self.asset_dir / f"landcover_{slug}.png"
        if not out_path.exists():
            plt.figure(figsize=(7.2, 4.2), dpi=160)
            colors = ["#2563eb", "#16a34a", "#eab308", "#0ea5e9", "#64748b"]
            labels = profile["labels"] if HAS_CJK_FONT else profile["chart_labels"]
            title = f"{request.region} {request.task}类型占比" if HAS_CJK_FONT else "Land-cover Composition"
            ylabel = "占比（%）" if HAS_CJK_FONT else "Share"
            plt.bar(labels, [v / 100 for v in profile["values"]], color=colors)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.ylim(0, (max(profile["values"]) + 14) / 100)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.grid(axis="y", alpha=0.22)
            for idx, val in enumerate(profile["values"]):
                plt.text(idx, (val + 1) / 100, f"{val}%", ha="center", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        return ChartAsset(
            title="地表类型占比",
            kind="bar",
            url=f"/reports/assets/{out_path.name}",
            caption="地表类型占比图，显示当前区域各类地表的相对构成。",
        )

    def _score(self, *parts: str, low: float, high: float) -> float:
        raw = "|".join(parts).encode("utf-8")
        value = int(hashlib.sha1(raw).hexdigest()[:8], 16) / 0xFFFFFFFF
        return low + (high - low) * value

    def _build_aef_payload(self, request: ReportRequest, profile: dict) -> dict:
        return {
            "region": request.region,
            "task": request.task,
            "time_range": request.time_range,
            "aoi": {
                "name": request.region,
                "main_terrain": profile["terrain"],
            },
            "metrics": {
                "coverage": self._score(request.region, request.time_range, low=0.81, high=0.96),
                "stability": self._score(request.prompt, request.time_range, low=0.72, high=0.93),
            },
            "outputs": [
                "embedding_map",
                "landcover_distribution",
                "confidence_summary",
            ],
        }
