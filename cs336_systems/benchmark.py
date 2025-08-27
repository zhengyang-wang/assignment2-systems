import argparse
import logging
import torch
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int
from utils import time_block, get_device, log_stats

logger = logging.getLogger(__name__)

try:
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
    logger.info("Using new API: BasicsTransformerLM + nn_utils.cross_entropy")
except ImportError:
    from cs336_basics.model import TransformerLM as BasicsTransformerLM
    from cs336_basics.train import cross_entropy_loss as cross_entropy
    logger.info("Using own API: TransformerLM + train.cross_entropy_loss")


def parse_args():
    parser = argparse.ArgumentParser(
        description="basic end-to-end benchmarking of the forward and backward passes"
    )

    benchmarking_setup_group = parser.add_argument_group('Benchmarking', 'Benchmarking setup parameters')
    model_group = parser.add_argument_group('Model', 'Model architecture parameters')

    benchmarking_setup_group.add_argument(
        "--include-backward",
        action="store_true",
        help="Include the backward pass in benchmarking"
    )

    benchmarking_setup_group.add_argument(
        "--apply-torch-compile",
        action="store_true",
        help="Use torch.compile"
    )

    benchmarking_setup_group.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Warm-up steps before starting measuring time (default: 5)"
    )

    benchmarking_setup_group.add_argument(
        "--execution-steps",
        type=int,
        default=5,
        help="Execution steps for measuring time (default: 5)"
    )

    benchmarking_setup_group.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for benchmarking (default: 4)"
    )

    model_group.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size of the model"
    )
    
    model_group.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Context length for transformer (default: 256)"
    )
    
    model_group.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Hidden dimension of the model (default: 768)"
    )
    
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 12)"
    )
    
    model_group.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads (default: 12)"
    )
    
    model_group.add_argument(
        "--d-ff",
        type=int,
        default=1344,
        help="Feedforward dimension (default: 3072)"
    )
    
    model_group.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter (default: 10000.0)"
    )
    return parser.parse_args()


def get_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: torch.device,
    apply_torch_compile: bool
) -> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.to(device)
    if apply_torch_compile:
        # Device-specific model compilation
        if device.type == "cpu":
            model = torch.compile(model)  # Standard compilation for CPU
        elif device.type == "mps":
            model = torch.compile(model, backend="aot_eager")  # Optimize backward pass on MPS
        else:  # CUDA and other devices
            model = torch.compile(model)  # Standard compilation
    return model


def get_random_batch(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    device: torch.device
) -> Int[Tensor, " ..."]:
    return (
        torch.randint(
            low=0,
            high=vocab_size-1,
            size=(batch_size, context_length),
            dtype=torch.long,
            device=device
        ),
        torch.randint(
            low=0,
            high=vocab_size-1,
            size=(batch_size, context_length),
            dtype=torch.long,
            device=device
        )
    )


def main():
    args = parse_args()
    device = get_device()

    logger.info("Start benchmarking with the following config:")
    logger.info(args)

    logger.info("\nInitialize model")
    model = get_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        apply_torch_compile=args.apply_torch_compile
    )
    logger.info(f"model size: {sum(p.numel() for p in model.parameters())}")

    logger.info("\nInitizalize a random batch")
    x_batch, y_batch = get_random_batch(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device
    )
    logger.info(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

    logger.info("\nWarm up")
    for _ in tqdm(range(args.warmup_steps), desc="Warm up"):
        logits = model(x_batch)
        loss = cross_entropy(
            inputs=logits.view(-1, args.vocab_size),
            targets=y_batch.view(-1)
        )
        if args.include_backward:
            loss.backward()
            model.zero_grad()

    fwd_time = []
    loss_time = []
    if args.include_backward:
        bwd_time = []
    logger.info("\nMeasure")
    for _ in tqdm(range(args.execution_steps), desc="Execution"):
        with time_block(device) as tb:
            logits = model(x_batch)
        fwd_time.append(tb.elapsed)

        with time_block(device) as tb:
            loss = cross_entropy(
                inputs=logits.view(-1, args.vocab_size),
                targets=y_batch.view(-1)
            )
        loss_time.append(tb.elapsed)

        if args.include_backward:
            with time_block(device) as tb:
                loss.backward()
            bwd_time.append(tb.elapsed)
            
            model.zero_grad()

    log_stats("fwd time", fwd_time)
    log_stats("loss time", loss_time)
    if args.include_backward:
        log_stats("bwd time", bwd_time)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()