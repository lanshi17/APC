"""SQLAlchemy 记录模型（PRD 7.4：Task/Model/Genome/CompiledPrompt/Trial 五类记录）。

数据库路径：环境变量 APC_DB_URL 优先；否则仓库根 data/apc.db（避免旧的 ../ 相对路径坑）。
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_URL = f"sqlite:///{_REPO_ROOT / 'data' / 'apc.db'}"

Base = declarative_base()


class TaskRecord(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(64), nullable=False, index=True)
    version = Column(String(16), nullable=False, default="1.0")
    objective = Column(Text, nullable=False)
    spec_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(64), nullable=False, index=True)
    provider = Column(String(32), nullable=False)
    capability_json = Column(JSON, nullable=False)
    profile_version = Column(String(32), nullable=False)
    created_at = Column(DateTime, nullable=False)


class GenomeRecord(Base):
    __tablename__ = "genomes"

    id = Column(Integer, primary_key=True)
    genome_id = Column(String(64), nullable=False, unique=True, index=True)
    parent_genome_id = Column(String(64), nullable=True, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    mutation_note = Column(Text, nullable=True)
    is_baseline = Column(Boolean, nullable=False, default=False)
    genome_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)


class CompiledPromptRecord(Base):
    __tablename__ = "compiled_prompts"

    id = Column(Integer, primary_key=True)
    prompt_id = Column(String(64), nullable=False, unique=True, index=True)
    genome_id = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    token_estimate = Column(Integer, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)


class TrialRecord(Base):
    __tablename__ = "trials"

    id = Column(Integer, primary_key=True)
    trial_id = Column(String(64), nullable=False, unique=True, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    genome_id = Column(String(64), nullable=False, index=True)
    prompt_id = Column(String(64), nullable=False, index=True)
    dataset_id = Column(String(64), nullable=False)
    dataset_version = Column(String(16), nullable=False)
    judge_id = Column(String(64), nullable=False)
    score = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    instruction_following = Column(Float, nullable=False)
    format_score = Column(Float, nullable=False)
    constraint_score = Column(Float, nullable=False)
    robustness = Column(Float, nullable=False)
    efficiency_score = Column(Float, nullable=False)
    format_error_rate = Column(Float, nullable=False)
    parent_genome_id = Column(String(64), nullable=True, index=True)
    mutation_note = Column(Text, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    total_input_tokens = Column(Integer, nullable=True)
    total_output_tokens = Column(Integer, nullable=True)
    phase = Column(String(32), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)


def get_engine(url: str | None = None):
    url = url or os.getenv("APC_DB_URL", DEFAULT_DB_URL)
    return create_engine(url, echo=False, future=True)


def get_session(engine: "Engine | str | None" = None):
    """engine 可传 SQLAlchemy Engine、连接 URL 字符串或 None（默认库）。"""
    if engine is None or isinstance(engine, str):
        engine = get_engine(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
