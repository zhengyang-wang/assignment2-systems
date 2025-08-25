import torch
import timeit
import logging
import numpy as np


logger = logging.getLogger(__name__)


class time_block:
    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        self.start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        self.elapsed = timeit.default_timer() - self.start


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using device cuda:0")
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda:0")
    if torch.mps.is_available():
        logger.info("Using device mps")
        return torch.device("mps")
    logger.info("Using device cpu")
    return torch.device("cpu")


def log_stats(name: str, perf_metrics: list[float]) -> dict[str, float | int]:
    perf_metrics_nd = np.array(perf_metrics)
    avg_metric = np.mean(perf_metrics_nd)
    std_metric = np.std(perf_metrics_nd)
    avg_metric = sum(perf_metrics_nd) / len(perf_metrics_nd)
    metrics = {"avg": avg_metric, "std": std_metric}
    print_str_list = []
    for agg_name, agg_val in metrics.items():
        print_str_list.append(f"{agg_name}: {agg_val:.6f}")
    print_str = f"[{name}] " + ", ".join(print_str_list)
    logger.info(print_str)
    return metrics