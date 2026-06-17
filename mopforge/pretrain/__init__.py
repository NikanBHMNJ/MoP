"""Continued-pretraining corpus API for MoP-Forge."""

from mopforge.pretrain.config import ContinuedPretrainConfig, ContinuedPretrainResult
from mopforge.pretrain.corpus import (
    TextCorpusRecord,
    TextCorpusStore,
    build_corpus_from_lessons,
    build_demo_code_corpus,
)
from mopforge.pretrain.dataset import CorpusCausalLMCollator, CorpusCausalLMDataset
from mopforge.pretrain.runner import run_continued_pretraining

__all__ = [
    "ContinuedPretrainConfig",
    "ContinuedPretrainResult",
    "CorpusCausalLMCollator",
    "CorpusCausalLMDataset",
    "TextCorpusRecord",
    "TextCorpusStore",
    "build_corpus_from_lessons",
    "build_demo_code_corpus",
    "run_continued_pretraining",
]
