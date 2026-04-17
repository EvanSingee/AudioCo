"""
run_pipeline.py – Orchestrator: runs all 4 agents in sequence.

Usage:
    python run_pipeline.py              # full pipeline D0 → D3
    python run_pipeline.py --from 2     # resume from Agent 2
    python run_pipeline.py --only 1     # run just Agent 1
"""

import argparse
import logging
import time
import json
import os

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("Pipeline")

STAGES = {
    1: ("VAD Segmentation",     "agents.agent1_vad",     "run"),
    2: ("Speaker Clustering",   "agents.agent2_cluster", "run"),
    3: ("Data Quality Filter",  "agents.agent3_quality", "run"),
    4: ("Dataset Builder",      "agents.agent4_builder", "run"),
}


def run_stage(stage_num: int) -> float:
    name, module_str, func_str = STAGES[stage_num]
    logger.info(f"\n{'='*60}")
    logger.info(f"  STAGE {stage_num}: {name}")
    logger.info(f"{'='*60}")
    import importlib
    module = importlib.import_module(module_str)
    func   = getattr(module, func_str)
    t0 = time.time()
    func()
    elapsed = time.time() - t0
    logger.info(f"Stage {stage_num} done in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="AudioCo Pipeline Orchestrator")
    parser.add_argument("--from",  dest="from_stage", type=int, default=1,
                        help="Start from this stage (1-4)")
    parser.add_argument("--only",  dest="only_stage", type=int, default=None,
                        help="Run only this stage")
    args = parser.parse_args()

    timings = {}

    if args.only_stage:
        t = run_stage(args.only_stage)
        timings[STAGES[args.only_stage][0]] = t
    else:
        for stage_num in range(args.from_stage, 5):
            t = run_stage(stage_num)
            timings[STAGES[stage_num][0]] = t

    total = sum(timings.values())
    logger.info("\n=== Pipeline Complete ===")
    for name, t in timings.items():
        logger.info(f"  {name:<30} {t:6.1f}s")
    logger.info(f"  {'TOTAL':<30} {total:6.1f}s")

    # Save timing log
    log_path = os.path.join(config.LOGS_DIR, "pipeline_timings.json")
    with open(log_path, "w") as f:
        json.dump({"stages": timings, "total_seconds": total}, f, indent=2)
    logger.info(f"Timings → {log_path}")


if __name__ == "__main__":
    main()
