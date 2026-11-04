import time
import csv
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# SPARTA project path
# ============================================================

SPARTA_ROOT = "/home/nrashvan/sparta_test/Sparta"

if SPARTA_ROOT not in sys.path:
    sys.path.insert(
        0,
        SPARTA_ROOT,
    )

from models import TransformerPredictor


# ============================================================
# Configuration
# ============================================================

WARMUP_ITERS = 50
LATENCY_ITERS = 200
THROUGHPUT_ITERS = 200

BATCH_SIZE = 1

OUTPUT_CSV = "sparta_edge_benchmark"


# ============================================================
# Device synchronization
# ============================================================

def synchronize(device):
    """
    Synchronize CUDA only when running on GPU.
    On CPU, this function does nothing.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# Read checkpoint argument
# ============================================================

def get_checkpoint_arg(
    ckpt_args,
    name,
    default=None,
):

    if isinstance(
        ckpt_args,
        dict,
    ):

        return ckpt_args.get(
            name,
            default,
        )

    return getattr(
        ckpt_args,
        name,
        default,
    )


# ============================================================
# Load SPARTA model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    if "args" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "the original training arguments."
        )


    ckpt_args = checkpoint[
        "args"
    ]


    # --------------------------------------------------------
    # Original training configuration
    # --------------------------------------------------------

    seg_len = int(
        get_checkpoint_arg(
            ckpt_args,
            "seg_len",
            24,
        )
    )


    num_kp = int(
        get_checkpoint_arg(
            ckpt_args,
            "num_kp",
            18,
        )
    )


    token_config = (
        get_checkpoint_arg(
            ckpt_args,
            "token_config",
            "CTR",
        )
    )


    relative = bool(
        get_checkpoint_arg(
            ckpt_args,
            "relative",
            False,
        )
    )


    traj = bool(
        get_checkpoint_arg(
            ckpt_args,
            "traj",
            False,
        )
    )


    num_heads = int(
        get_checkpoint_arg(
            ckpt_args,
            "model_num_heads",
            get_checkpoint_arg(
                ckpt_args,
                "num_heads",
                2,
            ),
        )
    )


    latent_dim = int(
        get_checkpoint_arg(
            ckpt_args,
            "model_latent_dim",
            get_checkpoint_arg(
                ckpt_args,
                "latent_dim",
                64,
            ),
        )
    )


    num_layers = int(
        get_checkpoint_arg(
            ckpt_args,
            "model_num_layers",
            get_checkpoint_arg(
                ckpt_args,
                "num_layers",
                2,
            ),
        )
    )


    dropout = float(
        get_checkpoint_arg(
            ckpt_args,
            "dropout",
            0.1,
        )
    )


    # ========================================================
    # Reproduce original d_model logic
    # ========================================================

    expand_ratio = 1
    extra_dim = 0


    if relative:
        expand_ratio = 2


    token_config_lower = str(
        token_config
    ).lower()


    if token_config_lower == "kps":

        input_dim = (
            seg_len
            *
            2
        )


    elif token_config_lower == "2ds":

        input_dim = (
            seg_len
        )


    elif token_config_lower == "st":

        input_dim = 2


    else:

        if traj:

            if relative:
                extra_dim = 4

            else:
                extra_dim = 2


        input_dim = (
            num_kp
            *
            2
        )


    d_model = (
        input_dim
        *
        expand_ratio
        +
        extra_dim
    )


    # ========================================================
    # Construct SPARTA model
    # ========================================================

    model = TransformerPredictor(
        d_model,
        num_heads,
        latent_dim,
        num_layers,
        1000,
        device=str(device),
        dropout=dropout,
    )


    model = model.to(
        device
    )


    # ========================================================
    # Load trained weights
    # ========================================================

    state_dict = checkpoint[
        "state_dict"
    ]


    clean_state_dict = {}


    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                7:
            ]


        clean_state_dict[
            key
        ] = value


    model.load_state_dict(
        clean_state_dict,
        strict=True,
    )


    model.eval()


    config = {

        "seg_len":
            seg_len,

        "num_kp":
            num_kp,

        "token_config":
            token_config,

        "relative":
            relative,

        "traj":
            traj,

        "d_model":
            d_model,

        "num_heads":
            num_heads,

        "latent_dim":
            latent_dim,

        "num_layers":
            num_layers,

        "dropout":
            dropout,
    }


    return (
        model,
        checkpoint,
        config,
    )





# ============================================================
# SPARTA forward
# ============================================================

def run_model(
    model,
    x,
):

    # Same inference path used by your loading/test script.

    return model(
        x,
        x,
    )


# ============================================================
# Benchmark
# ============================================================

def benchmark_model(
    model,
    seg_len,
    d_model,
    device,
):

    # --------------------------------------------------------
    # Synthetic SPARTA input
    #
    # [B, T, d_model]
    #
    # B = batch size
    # T = frames in one pose window
    #
    # One tensor corresponds to one pose window per
    # batch element.
    # --------------------------------------------------------

    torch.manual_seed(
        0
    )


    x = torch.randn(
        BATCH_SIZE,
        seg_len,
        d_model,
        dtype=torch.float32,
        device=device,
    )


    # ========================================================
    # Warm-up
    # ========================================================

    print(
        f"Running {WARMUP_ITERS} "
        f"warm-up iterations..."
    )


    with torch.inference_mode():

        for _ in range(
            WARMUP_ITERS
        ):

            _ = run_model(
                model,
                x,
            )


    synchronize(
        device
    )


    # ========================================================
    # Latency
    #
    # One forward = one pose window per batch element.
    # ========================================================

    print(
        f"Running {LATENCY_ITERS} "
        f"latency iterations..."
    )


    latency_ms = []


    with torch.inference_mode():

        for _ in range(
            LATENCY_ITERS
        ):

            synchronize(
                device
            )


            start = (
                time.perf_counter()
            )


            _ = run_model(
                model,
                x,
            )


            synchronize(
                device
            )


            end = (
                time.perf_counter()
            )


            elapsed_ms = (
                end
                -
                start
            ) * 1000.0


            latency_ms.append(
                elapsed_ms
            )


    mean_latency_ms = float(
        np.mean(
            latency_ms
        )
    )



    # --------------------------------------------------------
    # Amortized latency per window
    #
    # Each forward processes BATCH_SIZE windows.
    # For BATCH_SIZE=1:
    #
    # latency / window = forward latency
    # --------------------------------------------------------

    latency_per_window_ms = (
        mean_latency_ms
        /
        BATCH_SIZE
    )


    # ========================================================
    # Throughput
    # ========================================================

    print(
        f"Running {THROUGHPUT_ITERS} "
        f"throughput iterations..."
    )


    with torch.inference_mode():

        synchronize(
            device
        )


        start = (
            time.perf_counter()
        )


        for _ in range(
            THROUGHPUT_ITERS
        ):

            _ = run_model(
                model,
                x,
            )


        synchronize(
            device
        )


        end = (
            time.perf_counter()
        )


    total_time = (
        end
        -
        start
    )


    # --------------------------------------------------------
    # Complete forward calls / second
    # --------------------------------------------------------

    forwards_per_second = (
        THROUGHPUT_ITERS
        /
        total_time
    )


    # --------------------------------------------------------
    # Pose windows / second
    #
    # Each forward processes BATCH_SIZE pose windows.
    # --------------------------------------------------------

    windows_per_second = (
        THROUGHPUT_ITERS
        *
        BATCH_SIZE
        /
        total_time
    )


    return {

        "mean_latency_ms":
            mean_latency_ms,


        "latency_per_window_ms":
            latency_per_window_ms,

        "forwards_per_second":
            forwards_per_second,

        "windows_per_second":
            windows_per_second,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Command-line arguments
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SPARTA "
            "on CPU or GPU."
        )
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to pretrained "
            "SPARTA checkpoint."
        ),
    )


    parser.add_argument(
        "--device",
        type=str,
        choices=[
            "cpu",
            "gpu",
            "cuda",
        ],
        default="gpu",
        help=(
            "Execution device: "
            "cpu, gpu, or cuda."
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # Device selection
    # ========================================================

    requested_device = (
        args.device.lower()
    )


    if requested_device in [
        "gpu",
        "cuda",
    ]:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "GPU/CUDA was requested, "
                "but CUDA is not available."
            )


        device = torch.device(
            "cuda"
        )


        device_label = (
            "gpu"
        )


    else:

        device = torch.device(
            "cpu"
        )


        device_label = (
            "cpu"
        )


    # ========================================================
    # Device information
    # ========================================================

    print()
    print(
        "=" * 80
    )


    if device.type == "cuda":

        print(
            "Execution device     : GPU"
        )

        print(
            "GPU                  : "
            f"{torch.cuda.get_device_name(device)}"
        )


    else:

        print(
            "Execution device     : CPU"
        )

        print(
            "CPU threads          : "
            f"{torch.get_num_threads()}"
        )


    print(
        "=" * 80
    )


    # ========================================================
    # Load SPARTA
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "MODEL: SPARTA"
    )

    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )

    print(
        "=" * 80
    )


    model, checkpoint, config = (
        load_model(
            args.checkpoint,
            device,
        )
    )


    # ========================================================
    # Model configuration
    # ========================================================

    seg_len = config[
        "seg_len"
    ]

    num_kp = config[
        "num_kp"
    ]

    d_model = config[
        "d_model"
    ]


    input_shape = (
        BATCH_SIZE,
        seg_len,
        d_model,
    )




    # ========================================================
    # Print model information
    # ========================================================

    print()

    print(
        f"Segment length       : "
        f"{seg_len}"
    )

    print(
        f"Number of keypoints  : "
        f"{num_kp}"
    )

    print(
        f"Token configuration  : "
        f"{config['token_config']}"
    )

    print(
        f"Relative             : "
        f"{config['relative']}"
    )

    print(
        f"Trajectory           : "
        f"{config['traj']}"
    )

    print(
        f"d_model              : "
        f"{d_model}"
    )

    print(
        f"Number of heads      : "
        f"{config['num_heads']}"
    )

    print(
        f"Latent dimension     : "
        f"{config['latent_dim']}"
    )

    print(
        f"Number of layers     : "
        f"{config['num_layers']}"
    )

    print(
        f"Input shape          : "
        f"{input_shape}"
    )


    print()


    # ========================================================
    # Benchmark
    # ========================================================

    result = benchmark_model(

        model=model,

        seg_len=seg_len,

        d_model=d_model,

        device=device,
    )


    # ========================================================
    # Store metadata
    # ========================================================

    result.update({

        "model":
            "SPARTA",

        "device":
            device_label,

        "checkpoint":
            args.checkpoint,

        "batch_size":
            BATCH_SIZE,

        "segment_length":
            seg_len,

        "num_keypoints":
            num_kp,

        "token_config":
            config[
                "token_config"
            ],

        "d_model":
            d_model,

        "input_shape":
            str(
                input_shape
            ),

    })


    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "-" * 80
    )


    print(
        f"Execution device        : "
        f"{device_label.upper()}"
    )


    print(
        f"Mean latency / forward  : "
        f"{result['mean_latency_ms']:.3f} ms"
    )




    print(
        f"Latency / window        : "
        f"{result['latency_per_window_ms']:.3f} ms/window"
    )


    print(
        f"Model forwards / sec    : "
        f"{result['forwards_per_second']:.2f}"
    )


    print(
        f"Windows / sec           : "
        f"{result['windows_per_second']:.2f}"
    )


  


    print(
        "-" * 80
    )


    # ========================================================
    # Save CSV
    # ========================================================

    output_path = Path(
        f"{OUTPUT_CSV}_{device_label}.csv"
    )


    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                result.keys(),
        )


        writer.writeheader()

        writer.writerow(
            result
        )


    print(
        f"\nResults saved to: "
        f"{output_path.resolve()}"
    )


    # ========================================================
    # Cleanup
    # ========================================================

    del model


    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()