from __future__ import annotations

import logging
from typing import Any

from lightguard.reporting_phase1 import (
    generate_class_specific_analysis,
    visualize_concept_drift,
    visualize_performance_decay,
)

logger = logging.getLogger(__name__)


def run_full_analysis_phase1(analyzer: Any) -> bool:
    """Run the complete Phase I analysis pipeline for a DurabilityAnalyzer-like object."""
    logger.info("=" * 80)
    logger.info("STARTING PHASE I: DIAGNOSING THE DURABILITY DECEPTION")
    logger.info("=" * 80)

    if not analyzer.load_and_preprocess_data():
        logger.error("Data loading failed. Exiting.")
        return False

    if not analyzer.create_time_windows(n_windows=8, min_samples_per_class=10):
        logger.error("Time window creation failed. Exiting.")
        return False

    analyzer.initialize_models()
    analyzer.train_on_first_window()
    analyzer.evaluate_longitudinal_performance()

    _significance_results = analyzer.calculate_statistical_significance()

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    visualize_performance_decay(analyzer)
    visualize_concept_drift(analyzer)
    generate_class_specific_analysis(analyzer)

    logger.info("\n" + "=" * 80)
    logger.info("PHASE I ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return True
