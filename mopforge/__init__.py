"""MoP-Forge package.

MoP-Forge is an experimental framework for structured training data and,
eventually, Mixture-of-Parameters research prototypes.

The broad top-level namespace is retained for backwards compatibility with older
examples and tests. New integrations should prefer
``mopforge.public_api`` for the curated stable/experimental policy.
"""

from mopforge.kts import (
    IndexedLessonStore,
    KnowledgeLesson,
    LessonDataset,
    LessonIndex,
    LessonStore,
)

__all__ = [
    "CurriculumConfig",
    "EXPERIMENTAL_PUBLIC_API",
    "PUBLIC_API_POLICY",
    "STABLE_PUBLIC_API",
    "PublicAPIPolicy",
    "CurriculumPlan",
    "CurriculumScheduler",
    "ArtifactManager",
    "ArtifactRecord",
    "AnalysisConfig",
    "AnalysisRecord",
    "AnalysisRegistry",
    "AnalysisResult",
    "BenchmarkConfig",
    "BenchmarkRecord",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "ByteTokenizer",
    "CheckpointManager",
    "DatasetConfig",
    "DatasetManifest",
    "DatasetRecord",
    "DatasetRegistry",
    "DatasetSplit",
    "DatasetStats",
    "FileFingerprint",
    "ContinuedPretrainConfig",
    "ContinuedPretrainResult",
    "CorpusCausalLMCollator",
    "CorpusCausalLMDataset",
    "FeedbackRetrainingConfig",
    "FeedbackRetrainingResult",
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentRunResult",
    "ConditionEmbedding",
    "FastAdapter",
    "FastAdapterBank",
    "FastAdapterConfig",
    "FinetuneConfig",
    "FinetuneResult",
    "GeneratedAdapter",
    "GeneratedParameterConfig",
    "GPUDataConfig",
    "GPUTrainer",
    "GPUTrainingConfig",
    "GPUTrainingResult",
    "GPUTrainingState",
    "GPURunRecord",
    "GPURunRegistry",
    "HFTokenizerWrapper",
    "LessonFeedbackRecord",
    "LessonFeedbackStore",
    "IndexedLessonStore",
    "KnowledgeLesson",
    "LessonDataset",
    "LessonIndex",
    "LessonStore",
    "MoPForgeConfig",
    "ParameterGroupSummary",
    "RunRegistry",
    "SUPPORTED_POLICY_MODES",
    "TinyTrainingRunConfig",
    "TinyTrainer",
    "TextCorpusRecord",
    "TextCorpusStore",
    "TokenizerProtocol",
    "TokenizerSpec",
    "TrainableParameterPolicy",
    "TrainingQueueItem",
    "TrainingQueueStore",
    "TrainingCheckpointRecord",
    "TrainerConfig",
    "TrainerResult",
    "TrainerState",
    "AmpScaler",
    "DistributedConfig",
    "ModelMemoryEstimate",
    "ModuleRoutingPlan",
    "build_torchrun_command",
    "estimate_training_memory",
    "model_profile_100m_dense",
    "model_profile_100m_mop",
    "model_profile_1b_mop",
    "model_profile_2b_mop",
    "model_profile_500m_dense",
    "model_profile_500m_mop",
    "model_profile_7b_mop",
    "TrainingRunRecord",
    "TrainingModeSpec",
    "adapter_names_from_target_modules",
    "apply_trainable_policy",
    "build_module_queue_from_indexed_store",
    "build_finetune_lesson_filter",
    "build_corpus_from_lessons",
    "build_demo_code_corpus",
    "build_optimizer_for_trainable_parameters",
    "build_tokenizer",
    "build_queue_items_from_curriculum",
    "analysis_config_from_envelope",
    "benchmark_config_from_envelope",
    "build_markdown_report",
    "combined_fingerprint",
    "compute_dataset_stats",
    "consume_queue_once",
    "condition_names_from_target_modules",
    "count_parameters",
    "count_by_key",
    "compare_results",
    "create_dataset_split",
    "default_analysis_adapter_vs_generated_config",
    "default_analysis_composite_report_config",
    "default_analysis_dense_vs_mop_config",
    "default_benchmark_code_correctness_config",
    "default_benchmark_composite_config",
    "default_benchmark_loss_config",
    "default_benchmark_parameter_efficiency_config",
    "default_benchmark_router_config",
    "default_dataset_register_corpus_config",
    "default_dataset_register_lessons_config",
    "default_dataset_split_lessons_config",
    "default_fast_adapter_sft_config",
    "default_experiment_adapter_vs_generated_config",
    "default_experiment_dense_vs_mop_config",
    "default_generated_sft_config",
    "default_pretrain_config",
    "default_sft_config",
    "default_trainer_config",
    "dry_run_config",
    "evaluate_code_correctness",
    "evaluate_composite",
    "evaluate_loss",
    "evaluate_parameter_efficiency",
    "evaluate_router",
    "experiment_config_from_envelope",
    "expand_experiment_matrix",
    "filter_rows",
    "flatten_metrics",
    "feedback_records_from_generation_eval",
    "fingerprint_file",
    "fingerprint_files",
    "finetune_config_from_envelope",
    "infer_parameter_group",
    "get_training_mode_spec",
    "get_tokenizer_pad_token_id",
    "get_tokenizer_special_token_id",
    "get_tokenizer_vocab_size",
    "get_default_config",
    "group_rows",
    "list_default_config_names",
    "list_training_modes",
    "load_benchmark_metrics",
    "load_config_file",
    "load_dataset_split",
    "load_experiment_summary",
    "load_full_training_checkpoint",
    "load_records_for_split",
    "load_run_result",
    "markdown_table",
    "normalize_benchmark_metrics",
    "normalize_experiment_rows",
    "normalize_run_result",
    "numeric_delta",
    "policy_from_queue_item",
    "pretrain_config_from_envelope",
    "rank_lesson_ids_by_feedback",
    "register_tokenizer_type",
    "rank_rows",
    "dataset_config_from_envelope",
    "run_benchmark",
    "run_analysis",
    "run_finetune",
    "run_feedback_retraining_loop",
    "run_continued_pretraining",
    "run_experiment",
    "run_tiny_training_from_curriculum",
    "score_lesson",
    "save_config_file",
    "save_full_training_checkpoint",
    "safe_mean",
    "safe_rate",
    "slugify_dataset_id",
    "summarize_feedback_delta",
    "summarize_group",
    "summarize_parameter_groups",
    "normalize_adapter_names",
    "normalize_condition_names",
    "trainer_config_from_envelope",
    "trainer_config_from_finetune_config",
    "tokenizer_spec_from_config",
    "validate_config_envelope",
    "write_split_jsonl",
    "capture_rng_state",
    "restore_rng_state",
]

__all__.extend(
    [
        "AblationConfig",
        "AblationRecord",
        "AblationRegistry",
        "AblationResult",
        "AblationVariant",
        "BaselineConfig",
        "BaselineSpec",
        "ManifestConfig",
        "ManifestRegistry",
        "ModelArchitectureConfig",
        "ModelConfig",
        "ModelManifest",
        "ModelRecord",
        "ModelRegistry",
        "PaperReportConfig",
        "PaperReportRecord",
        "PaperReportRegistry",
        "ResearchRunManifest",
        "ResourceSpec",
        "ResultImportConfig",
        "ResultImportRecord",
        "ResultImportRegistry",
        "ablation_config_from_envelope",
        "baseline_config_from_envelope",
        "build_baseline_experiment_config",
        "build_paper_report",
        "build_tiny_model_from_architecture",
        "command_text",
        "compare_groups_simple",
        "config_from_path_or_payload",
        "default_ablation_adapter_vs_generated_config",
        "default_ablation_dense_vs_mop_config",
        "default_ablation_fastparam_policy_config",
        "default_baseline_dense_adapter_mop_config",
        "default_import_results_config",
        "default_manifest_a100_2b_plan_config",
        "default_manifest_b300_multi_gpu_plan_config",
        "default_manifest_cpu_smoke_config",
        "default_manifest_h100_mop_plan_config",
        "default_model_future_2b_mop_config",
        "default_model_tiny_adapter_config",
        "default_model_tiny_dense_config",
        "default_model_tiny_generated_config",
        "default_model_tiny_mop_config",
        "default_paper_report_smoke_config",
        "default_stats_summary_config",
        "detect_artifacts",
        "dry_run_ablation",
        "dry_run_payload",
        "expand_ablation_variants",
        "get_baseline",
        "import_config_from_envelope",
        "import_results",
        "list_baselines",
        "make_metric_table",
        "manifest_config_from_envelope",
        "mean",
        "median",
        "model_config_from_envelope",
        "paper_report_config_from_envelope",
        "parameter_summary_for_architecture",
        "percent_change",
        "plan_run_manifest",
        "run_ablation",
        "stddev",
        "stderr",
        "summarize_by_group",
        "summarize_metric",
        "write_table_csv",
        "write_table_json",
        "write_table_markdown",
        "RuntimeConfig",
        "RuntimeContext",
        "DeviceInfo",
        "PrecisionPolicy",
        "apply_runtime_determinism",
        "apply_tf32_policy",
        "autocast_context",
        "build_runtime_context",
        "detect_devices",
        "move_batch_to_device",
        "move_model_to_runtime",
        "resolve_device",
        "resolve_precision",
        "runtime_config_from_envelope",
        "runtime_config_from_kwargs",
        "runtime_metadata",
        "default_runtime_cpu_config",
        "default_runtime_auto_config",
        "default_runtime_cuda_bf16_plan_config",
        "default_trainer_runtime_auto_config",
        "default_sft_runtime_auto_config",
        "default_benchmark_runtime_auto_config",
    ]
)

try:
    from mopforge.configs import (
        MoPForgeConfig,
        ablation_config_from_envelope,
        analysis_config_from_envelope,
        baseline_config_from_envelope,
        default_analysis_adapter_vs_generated_config,
        default_analysis_composite_report_config,
        default_analysis_dense_vs_mop_config,
        default_ablation_adapter_vs_generated_config,
        default_ablation_dense_vs_mop_config,
        default_ablation_fastparam_policy_config,
        benchmark_config_from_envelope,
        dataset_config_from_envelope,
        default_benchmark_code_correctness_config,
        default_benchmark_composite_config,
        default_benchmark_loss_config,
        default_benchmark_parameter_efficiency_config,
        default_benchmark_router_config,
        default_baseline_dense_adapter_mop_config,
        default_dataset_register_corpus_config,
        default_dataset_register_lessons_config,
        default_dataset_split_lessons_config,
        default_experiment_adapter_vs_generated_config,
        default_experiment_dense_vs_mop_config,
        default_fast_adapter_sft_config,
        default_generated_sft_config,
        default_import_results_config,
        default_manifest_a100_2b_plan_config,
        default_manifest_b300_multi_gpu_plan_config,
        default_manifest_cpu_smoke_config,
        default_manifest_h100_mop_plan_config,
        default_model_future_2b_mop_config,
        default_model_tiny_adapter_config,
        default_model_tiny_dense_config,
        default_model_tiny_generated_config,
        default_model_tiny_mop_config,
        default_paper_report_smoke_config,
        default_pretrain_config,
        default_sft_config,
        default_stats_summary_config,
        default_trainer_config,
        dry_run_config,
        experiment_config_from_envelope,
        finetune_config_from_envelope,
        get_default_config,
        import_config_from_envelope,
        list_default_config_names,
        load_config_file,
        manifest_config_from_envelope,
        model_config_from_envelope,
        paper_report_config_from_envelope,
        pretrain_config_from_envelope,
        save_config_file,
        trainer_config_from_envelope,
        validate_config_envelope,
    )
except Exception:
    pass

try:
    from mopforge.datasets import (
        DatasetConfig,
        DatasetManifest,
        DatasetRecord,
        DatasetRegistry,
        DatasetSplit,
        DatasetStats,
        FileFingerprint,
        combined_fingerprint,
        compute_dataset_stats,
        create_dataset_split,
        fingerprint_file,
        fingerprint_files,
        load_dataset_split,
        load_records_for_split,
        slugify_dataset_id,
        write_split_jsonl,
    )
except Exception:
    pass

try:
    from mopforge.analysis import (
        AnalysisConfig,
        AnalysisRecord,
        AnalysisRegistry,
        AnalysisResult,
        build_markdown_report,
        compare_results,
        filter_rows,
        group_rows,
        load_benchmark_metrics,
        load_experiment_summary,
        load_run_result,
        markdown_table,
        normalize_benchmark_metrics,
        normalize_experiment_rows,
        normalize_run_result,
        numeric_delta,
        rank_rows,
        run_analysis,
        summarize_group,
    )
except Exception:
    pass

try:
    from mopforge.ablations import (
        AblationConfig,
        AblationRecord,
        AblationRegistry,
        AblationResult,
        AblationVariant,
        dry_run_ablation,
        expand_ablation_variants,
        run_ablation,
    )
except Exception:
    pass

try:
    from mopforge.baselines import (
        BaselineConfig,
        BaselineSpec,
        build_baseline_experiment_config,
        get_baseline,
        list_baselines,
    )
except Exception:
    pass

try:
    from mopforge.benchmarks import (
        BenchmarkConfig,
        BenchmarkRecord,
        BenchmarkRegistry,
        BenchmarkResult,
        count_by_key,
        evaluate_code_correctness,
        evaluate_composite,
        evaluate_loss,
        evaluate_parameter_efficiency,
        evaluate_router,
        flatten_metrics,
        run_benchmark,
        safe_mean,
        safe_rate,
    )
except Exception:
    pass

try:
    from mopforge.tokenization import (
        ByteTokenizer,
        HFTokenizerWrapper,
        TokenizerProtocol,
        TokenizerSpec,
        build_tokenizer,
        get_tokenizer_pad_token_id,
        get_tokenizer_special_token_id,
        get_tokenizer_vocab_size,
        register_tokenizer_type,
        tokenizer_spec_from_config,
    )
except Exception:
    pass

try:
    from mopforge.artifacts import ArtifactManager, ArtifactRecord, CheckpointManager
except Exception:
    pass

try:
    from mopforge.lifecycle import (
        TrainingCheckpointRecord,
        capture_rng_state,
        load_full_training_checkpoint,
        restore_rng_state,
        save_full_training_checkpoint,
    )
except Exception:
    pass

try:
    from mopforge.curriculum import CurriculumConfig, CurriculumPlan, CurriculumScheduler
except Exception:
    pass

try:
    from mopforge.experiments import (
        ExperimentConfig,
        ExperimentRecord,
        ExperimentRegistry,
        ExperimentRunResult,
        expand_experiment_matrix,
        run_experiment,
    )
except Exception:
    pass

try:
    from mopforge.models import (
        ConditionEmbedding,
        FastAdapter,
        FastAdapterBank,
        FastAdapterConfig,
        GeneratedAdapter,
        GeneratedParameterConfig,
        ModelArchitectureConfig,
        ModelConfig,
        ModelManifest,
        ModelRecord,
        ModelRegistry,
        adapter_names_from_target_modules,
        build_tiny_model_from_architecture,
        condition_names_from_target_modules,
        normalize_adapter_names,
        normalize_condition_names,
        parameter_summary_for_architecture,
    )
except Exception:
    pass

try:
    from mopforge.manifests import (
        ManifestConfig,
        ManifestRegistry,
        ResearchRunManifest,
        ResourceSpec,
        command_text,
        config_from_path_or_payload,
        dry_run_payload,
        plan_run_manifest,
    )
except Exception:
    pass

try:
    from mopforge.importers import (
        ResultImportConfig,
        ResultImportRecord,
        ResultImportRegistry,
        detect_artifacts,
        import_results,
    )
except Exception:
    pass

try:
    from mopforge.statistics import (
        compare_groups_simple,
        make_metric_table,
        mean,
        median,
        percent_change,
        stddev,
        stderr,
        summarize_by_group,
        summarize_metric,
        write_table_csv,
        write_table_json,
        write_table_markdown,
    )
except Exception:
    pass

try:
    from mopforge.papers import (
        PaperReportConfig,
        PaperReportRecord,
        PaperReportRegistry,
        build_paper_report,
    )
except Exception:
    pass

try:
    from mopforge.pretrain import (
        ContinuedPretrainConfig,
        ContinuedPretrainResult,
        CorpusCausalLMCollator,
        CorpusCausalLMDataset,
        TextCorpusRecord,
        TextCorpusStore,
        build_corpus_from_lessons,
        build_demo_code_corpus,
        run_continued_pretraining,
    )
except Exception:
    pass

try:
    from mopforge.sft import (
        FinetuneConfig,
        FinetuneResult,
        TrainingModeSpec,
        build_finetune_lesson_filter,
        get_training_mode_spec,
        list_training_modes,
        run_finetune,
        trainer_config_from_finetune_config,
    )
except Exception:
    pass

try:
    from mopforge.runs import RunRegistry, TinyTrainingRunConfig, TrainingRunRecord
    from mopforge.training import (
        SUPPORTED_POLICY_MODES,
        ParameterGroupSummary,
        TinyTrainer,
        TrainableParameterPolicy,
        TrainerConfig,
        TrainerResult,
        TrainerState,
        apply_trainable_policy,
        build_optimizer_for_trainable_parameters,
        count_parameters,
        infer_parameter_group,
        policy_from_queue_item,
        run_tiny_training_from_curriculum,
        summarize_parameter_groups,
    )
except Exception:
    pass

try:
    from mopforge.feedback import (
        LessonFeedbackRecord,
        LessonFeedbackStore,
        feedback_records_from_generation_eval,
        rank_lesson_ids_by_feedback,
        score_lesson,
    )
except Exception:
    pass

try:
    from mopforge.loops import (
        FeedbackRetrainingConfig,
        FeedbackRetrainingResult,
        run_feedback_retraining_loop,
        summarize_feedback_delta,
    )
except Exception:
    pass

try:
    from mopforge.queues import (
        TrainingQueueItem,
        TrainingQueueStore,
        build_module_queue_from_indexed_store,
        build_queue_items_from_curriculum,
        consume_queue_once,
    )
except Exception:
    pass

try:
    from mopforge.configs import (
        ablation_config_from_envelope,
        baseline_config_from_envelope,
        default_ablation_adapter_vs_generated_config,
        default_ablation_dense_vs_mop_config,
        default_ablation_fastparam_policy_config,
        default_baseline_dense_adapter_mop_config,
        default_import_results_config,
        default_manifest_a100_2b_plan_config,
        default_manifest_b300_multi_gpu_plan_config,
        default_manifest_cpu_smoke_config,
        default_manifest_h100_mop_plan_config,
        default_model_future_2b_mop_config,
        default_model_tiny_adapter_config,
        default_model_tiny_dense_config,
        default_model_tiny_generated_config,
        default_model_tiny_mop_config,
        default_paper_report_smoke_config,
        default_runtime_auto_config,
        default_runtime_cpu_config,
        default_runtime_cuda_bf16_plan_config,
        default_sft_runtime_auto_config,
        default_stats_summary_config,
        default_benchmark_runtime_auto_config,
        default_trainer_runtime_auto_config,
        import_config_from_envelope,
        manifest_config_from_envelope,
        model_config_from_envelope,
        paper_report_config_from_envelope,
        runtime_config_from_envelope,
    )
    from mopforge.manifests import (
        ManifestConfig,
        ManifestRegistry,
        ResearchRunManifest,
        ResourceSpec,
        command_text,
        config_from_path_or_payload,
        dry_run_payload,
        plan_run_manifest,
    )
except Exception:
    pass

try:
    from mopforge.runtime import (
        DeviceInfo,
        PrecisionPolicy,
        RuntimeConfig,
        RuntimeContext,
        apply_runtime_determinism,
        apply_tf32_policy,
        autocast_context,
        build_runtime_context,
        detect_devices,
        move_batch_to_device,
        move_model_to_runtime,
        resolve_device,
        resolve_precision,
        runtime_config_from_kwargs,
        runtime_metadata,
    )
except Exception:
    pass

try:
    from mopforge.gpu import (
        AmpScaler,
        DistributedConfig,
        GPUDataConfig,
        GPUTrainer,
        GPUTrainingConfig,
        GPUTrainingResult,
        GPUTrainingState,
        GPURunRecord,
        GPURunRegistry,
        ModelMemoryEstimate,
        ModuleRoutingPlan,
        build_torchrun_command,
        estimate_training_memory,
        model_profile_100m_dense,
        model_profile_100m_mop,
        model_profile_1b_mop,
        model_profile_2b_mop,
        model_profile_500m_dense,
        model_profile_500m_mop,
        model_profile_7b_mop,
    )
except Exception:
    pass

__version__ = "0.46.0"

try:
    from mopforge.public_api import (
        EXPERIMENTAL_PUBLIC_API,
        PUBLIC_API_POLICY,
        STABLE_PUBLIC_API,
        PublicAPIPolicy,
    )
except Exception:
    pass
