import time
import csv
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# TS-GAD project root
# ============================================================

PROJECT_ROOT = "/home/nrashvan/TSGAD"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# Project imports
# ============================================================

import lib.dist as dist
from models import VAE


# ============================================================
# Benchmark configuration
# ============================================================

WARMUP_ITERS = 50
LATENCY_ITERS = 200
THROUGHPUT_ITERS = 200

BATCH_SIZE = 1

OUTPUT_CSV = "tsgad_edge_benchmark"


# ============================================================
# Device synchronization
# ============================================================

def synchronize(device):
    """
    Synchronize CUDA only when running on GPU.

    On CPU this function does nothing.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)




# ============================================================
# Extract checkpoint state_dict
# ============================================================

def get_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        dict,
    ):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove DataParallel "module." prefix if present
    # --------------------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if key.startswith(
            "module."
        ):

            new_key = key[
                7:
            ]

        clean_state_dict[
            new_key
        ] = value


    return clean_state_dict


# ============================================================
# Load TS-GAD
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    # ========================================================
    # Model configuration
    #
    # Same configuration as your successful loading script.
    # ========================================================

    latent_dim = 10
    seg_len = 24

    conv = False
    graph = True

    headless = False

    tcvae = True
    mss = True

    dropout = 0.3

    conv_oper = "sagc"
    act = "relu"

    mse = False

    alpha = 1
    gamma = 1

    exclude_mutinfo = False


    # ========================================================
    # Distributions
    # ========================================================

    prior_dist = dist.Normal()
    q_dist = dist.Normal()


    # ========================================================
    # Construct model
    # ========================================================

    vae = VAE(

        z_dim=
            latent_dim,

        # IMPORTANT:
        # Use the selected benchmark device,
        # not torch.cuda.is_available().
        use_cuda=
            (
                device.type
                ==
                "cuda"
            ),

        device=
            str(device),

        prior_dist=
            prior_dist,

        q_dist=
            q_dist,

        include_mutinfo=
            not exclude_mutinfo,

        tcvae=
            tcvae,

        conv=
            conv,

        graph=
            graph,

        mss=
            mss,

        drop_out=
            dropout,

        conv_oper=
            conv_oper,

        act=
            act,

        headless=
            headless,

        input_frames=
            seg_len,

        mse=
            mse,

        alpha=
            alpha,

        gamma=
            gamma,
    )


    vae = vae.to(
        device
    )


    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    state_dict = get_state_dict(
        checkpoint
    )


    vae.load_state_dict(
        state_dict,
        strict=True,
    )


    vae.eval()


    config = {

        "latent_dim":
            latent_dim,

        "seg_len":
            seg_len,

        "num_keypoints":
            18,

        "channels":
            2,

        "conv":
            conv,

        "graph":
            graph,

        "headless":
            headless,

        "tcvae":
            tcvae,

        "mss":
            mss,

        "dropout":
            dropout,

        "conv_oper":
            conv_oper,

        "act":
            act,

        "mse":
            mse,

        "alpha":
            alpha,

        "gamma":
            gamma,
    }


    return (
        vae,
        checkpoint,
        config,
    )


# ============================================================
# TS-GAD inference
# ============================================================

def run_model(
    model,
    x,
):
    """
    Same inference path used in your original TS-GAD test:

        _, z_params, _ = vae.encode(x)

    We benchmark the encoder because this is the path
    used to obtain the latent representation.
    """

    output1, z_params, output3 = (
        model.encode(
            x
        )
    )


    return (
        output1,
        z_params,
        output3,
    )


# ============================================================
# Benchmark
# ============================================================

def benchmark_model(
    model,
    channels,
    seg_len,
    num_keypoints,
    device,
):

    # --------------------------------------------------------
    # Synthetic pose input
    #
    # [B, C, T, V]
    #
    # B = batch size
    # C = 2 coordinates
    # T = 24 frames
    # V = 18 keypoints
    # --------------------------------------------------------

    torch.manual_seed(
        0
    )


    x = torch.randn(
        BATCH_SIZE,
        channels,
        seg_len,
        num_keypoints,
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
    # Per-window latency
    #
    # One forward processes BATCH_SIZE windows.
    # For BATCH_SIZE = 1:
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


    # ========================================================
    # Throughput
    # ========================================================

    forwards_per_second = (
        THROUGHPUT_ITERS
        /
        total_time
    )


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

        "input_shape":
            str(
                tuple(
                    x.shape
                )
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Arguments
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TS-GAD "
            "on CPU or GPU."
        )
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to pretrained "
            "TS-GAD VAE checkpoint."
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
            "cuda:0"
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
    # Load model
    # ========================================================

    print()
    print(
        "=" * 80
    )


    print(
        "MODEL: TS-GAD"
    )


    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )


    print(
        "=" * 80
    )


    (
        model,
        checkpoint,
        config,

    ) = load_model(

        checkpoint_path=
            args.checkpoint,

        device=
            device,
    )


    # ========================================================
    # Configuration
    # ========================================================

    seg_len = config[
        "seg_len"
    ]

    channels = config[
        "channels"
    ]

    num_keypoints = config[
        "num_keypoints"
    ]

    latent_dim = config[
        "latent_dim"
    ]


    input_shape = (
        BATCH_SIZE,
        channels,
        seg_len,
        num_keypoints,
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
        f"{num_keypoints}"
    )


    print(
        f"Channels             : "
        f"{channels}"
    )


    print(
        f"Input shape          : "
        f"{input_shape}"
    )


    print(
        f"Latent dimension     : "
        f"{latent_dim}"
    )


    print(
        f"Graph model          : "
        f"{config['graph']}"
    )


    print(
        f"Conv model           : "
        f"{config['conv']}"
    )


    print(
        f"Conv operation       : "
        f"{config['conv_oper']}"
    )


    print(
        f"TCVAE                : "
        f"{config['tcvae']}"
    )


    print(
        f"MSS                  : "
        f"{config['mss']}"
    )


    print(
        f"Dropout              : "
        f"{config['dropout']}"
    )


  



    print()


    # ========================================================
    # Benchmark
    # ========================================================

    result = benchmark_model(

        model=
            model,

        channels=
            channels,

        seg_len=
            seg_len,

        num_keypoints=
            num_keypoints,

        device=
            device,
    )


    # ========================================================
    # Store metadata
    # ========================================================

    result.update({

        "model":
            "TS-GAD",

        "device":
            device_label,

        "checkpoint":
            args.checkpoint,

        "batch_size":
            BATCH_SIZE,

        "segment_length":
            seg_len,

        "channels":
            channels,

        "num_keypoints":
            num_keypoints,

        "latent_dim":
            latent_dim,


        "graph":
            config[
                "graph"
            ],

        "conv":
            config[
                "conv"
            ],

        "conv_oper":
            config[
                "conv_oper"
            ],
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