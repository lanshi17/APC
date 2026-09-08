"""python -m apc.graph --visualize 输出 LangGraph 流程图（mermaid）。"""
from __future__ import annotations

import argparse


def main() -> None:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
    parser = argparse.ArgumentParser(description="APC LangGraph 管线")

    if args.visualize:
        print(build_graph().get_graph().draw_mermaid())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
