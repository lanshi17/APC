"""LangGraph 管线冒烟：mock 客户端全节点贯通（PRD 4.1）。"""
from __future__ import annotations

from pathlib import Path

from apc.graph.pipeline import build_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_graph_end_to_end_mock(tmp_path):
    graph = build_graph()
    state = graph.invoke({
        "task_spec_path": str(REPO_ROOT / "configs" / "tasks" / "financial_analysis.yaml"),
        "genome_path": str(REPO_ROOT / "configs" / "genomes" / "base.json"),
        "model_id": "glm",
        "prefer_mock": True,
        "artifacts_dir": str(tmp_path),
        "budget": 8,
    })
    assert state.get("trial_result") and state["trial_result"]["trial_id"]
    assert state.get("optimization_report") and state["optimization_report"]["champion_genome_id"]
    # 画像 / 报告均落在指定 artifacts 目录（测试隔离生效）
    assert (tmp_path / "profiles" / "glm.json").exists()
    assert (tmp_path / "optimizations" / "glm_report.json").exists()


def test_visualize_mermaid():
    graph = build_graph()
    text = graph.get_graph().draw_mermaid()
    assert "probe" in text and "compile" in text and "evaluate" in text
