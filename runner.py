import argparse
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path

import yaml

from python import analyzer, buildinstance, solver

LOGGER = logging.getLogger("matcher")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
ALGORITHMS = {"RAMP", "Perturbed_Maximization", "Randomized", "Default"}
MATCHING_FILES_DIRNAME = "matching_files"
INPUT_FILENAMES = (
    "paper_info.csv",
    "reviewer_info.csv",
    "similarity_scores.csv",
    "bid_scores.csv",
    "constraints.csv",
)


class ApplicationLogFilter(logging.Filter):
    def filter(self, record):
        return record.name == "matcher" or record.name.startswith("python.")


def configure_logging(level):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.addFilter(ApplicationLogFilter())
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)


def add_run_log(path):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if getattr(handler, "is_matcher_run_log", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.is_matcher_run_log = True
    file_handler.addFilter(ApplicationLogFilter())
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("The config root must be a YAML mapping")

    required_keys = ("dataset_dir", "output_dir", "algo_name", "2stage")
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")

    dataset_dir = Path(config["dataset_dir"])
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    for filename in INPUT_FILENAMES:
        input_path = dataset_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    if config["algo_name"] not in ALGORITHMS:
        raise ValueError(
            f"Unknown algo_name {config['algo_name']!r}; choose one of "
            f"{', '.join(sorted(ALGORITHMS))}"
        )
    if not isinstance(config["2stage"], bool):
        raise ValueError("2stage must be true or false")
    if not isinstance(config.get("optimize_sampling", False), bool):
        raise ValueError("optimize_sampling must be true or false")

    for key in ("maxprob", "dynamic_maxprob"):
        if key not in config or config[key] is None:
            continue
        try:
            probability = float(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number in (0, 1]") from exc
        if not 0 < probability <= 1:
            raise ValueError(f"{key} must be in (0, 1]; got {config[key]}")

    for key in ("beta", "reward_region", "pen_coauthor", "pen_2cycle"):
        if key not in config or config[key] is None:
            continue
        try:
            value = float(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a non-negative number") from exc
        if value < 0:
            raise ValueError(f"{key} must be non-negative; got {config[key]}")


def load_config(config_path):
    LOGGER.info("Loading configuration from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def build_instance_from_config(config):
    dataset_dir = Path(config["dataset_dir"])
    return buildinstance.InputInstance(
        *(dataset_dir / filename for filename in INPUT_FILENAMES)
    )


def write_rows(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def solve_stage(instance, config, senior_only):
    return solver.solve(
        instance,
        config["algo_name"],
        config.get("maxprob"),
        config.get("dynamic_maxprob"),
        config.get("beta"),
        config.get("reward_region"),
        config.get("pen_coauthor"),
        config.get("pen_2cycle"),
        optimize_sampling=config.get("optimize_sampling", False),
        senior_only=senior_only,
    )


def write_stage_results(
    output_dir, prefix, instance, probability_pairs, matching_pairs
):
    filename_prefix = f"{prefix}_" if prefix else ""
    write_rows(
        output_dir / f"{filename_prefix}prob_assignment.csv",
        ("paper_id", "reviewer_id", "probability"),
        instance.external_probability_pairs(probability_pairs),
    )
    write_rows(
        output_dir / f"{filename_prefix}matching_pairs.csv",
        ("paper_id", "reviewer_id"),
        instance.external_matching_pairs(matching_pairs),
    )


def run_matching(config):
    output_dir = Path(config["output_dir"])
    algorithm = config["algo_name"]
    matching_output_dir = output_dir / MATCHING_FILES_DIRNAME
    matching_output_dir.mkdir(parents=True, exist_ok=True)
    add_run_log(matching_output_dir / "matching.log")

    start_time = time.perf_counter()
    LOGGER.info("Algorithm execution started at %s", dt.datetime.now().isoformat())

    instance_start = time.perf_counter()
    instance = build_instance_from_config(config)
    LOGGER.info(
        "Instance loading time: %.3f seconds", time.perf_counter() - instance_start
    )
    LOGGER.info("Algorithm: %s", algorithm)
    LOGGER.info("Two-stage matching: %s", config["2stage"])
    LOGGER.info("Sampling optimization: %s", config.get("optimize_sampling", False))
    LOGGER.info("Output directory: %s", output_dir)

    if config["2stage"]:
        LOGGER.info("Running stage 1 for senior reviewers")
        probability_matrix, probability_pairs, matching_pairs = solve_stage(
            instance, config, senior_only=True
        )
        write_stage_results(
            matching_output_dir,
            "stage1",
            instance,
            probability_pairs,
            matching_pairs,
        )
        LOGGER.info("Saved %d stage 1 matching pairs", len(matching_pairs))

        stage1_statistics = analyzer.analyze(
            instance, probability_matrix, matching_pairs, final=False
        )
        for paper, reviewer in matching_pairs:
            instance.constraint[paper][reviewer] = 1
            instance.matched_this_stage[paper].add(reviewer)
            instance.ellp[paper] -= 1

        LOGGER.info("Running stage 2 for junior reviewers")
        probability_matrix, probability_pairs, matching_pairs = solve_stage(
            instance, config, senior_only=False
        )
        write_stage_results(
            matching_output_dir,
            "stage2",
            instance,
            probability_pairs,
            matching_pairs,
        )
        LOGGER.info("Saved %d stage 2 matching pairs", len(matching_pairs))

        statistics = analyzer.analyze(
            instance, probability_matrix, matching_pairs, final=True
        )
        statistics.prob_quality += stage1_statistics.prob_quality
        statistics.matching_quality += stage1_statistics.matching_quality
        statistics.support_size += stage1_statistics.support_size
        statistics.entropy += stage1_statistics.entropy
    else:
        LOGGER.info("Running one-stage matching for all reviewers")
        probability_matrix, probability_pairs, matching_pairs = solve_stage(
            instance, config, senior_only=False
        )
        write_stage_results(
            matching_output_dir,
            "",
            instance,
            probability_pairs,
            matching_pairs,
        )
        LOGGER.info("Saved %d matching pairs", len(matching_pairs))
        statistics = analyzer.analyze(
            instance, probability_matrix, matching_pairs, final=True
        )

    statistics.running_time = time.perf_counter() - start_time
    LOGGER.info(
        "Matching computation completed in %.3f seconds", statistics.running_time
    )
    analyzer.output_statistics(instance, statistics, output_dir)
    LOGGER.info("All outputs written in %.3f seconds", time.perf_counter() - start_time)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run robust paper assignment")
    parser.add_argument("--config", default="configs/sample_config.yaml")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level))
    try:
        config = load_config(args.config)
        run_matching(config)
    except (FileNotFoundError, OSError, yaml.YAMLError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOGGER.error("Matching failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
