import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

from models.STG_NF.model_pose import STG_NF


# ============================================================
# Configuration
# ============================================================

WARMUP_ITERS = 50
LATENCY_ITERS = 200
THROUGHPUT_ITERS = 200

BATCH_SIZE = 1

OUTPUT_CSV = "stgnf_edge_benchmark"


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
# Extract checkpoint state dictionary
# ============================================================

def get_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
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
    # Remove DataParallel "module." prefix
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
# Mark ActNorm layers initialized
# ============================================================

def mark_actnorm_initialized(model):

    num_actnorm = 0

    for module in model.modules():

        class_name = (
            module.__class__.__name__
        )

        if "ActNorm" not in class_name:
            continue


        # ----------------------------------------------------
        # Some STG-NF versions use "inited"
        # ----------------------------------------------------

        if hasattr(
            module,
            "inited",
        ):

            if torch.is_tensor(
                module.inited
            ):

                module.inited.fill_(
                    True
                )

            else:

                module.inited = True


        # ----------------------------------------------------
        # Other versions use "initialized"
        # ----------------------------------------------------

        if hasattr(
            module,
            "initialized",
        ):

            if torch.is_tensor(
                module.initialized
            ):

                module.initialized.fill_(
                    True
                )

            else:

                module.initialized = True


        num_actnorm += 1


    return num_actnorm


# ============================================================
# Load STG-NF
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    # ========================================================
    # Model configuration
    #
    # Same configuration used in your successful loading test.
    # ========================================================

    model_args = {

        "pose_shape":
            (2, 24, 18),

        "hidden_channels":
            0,

        "K":
            8,

        "L":
            1,

        "R":
            3.0,

        "actnorm_scale":
            1.0,

        "flow_permutation":
            "permute",

        "flow_coupling":
            "affine",

        "LU_decomposed":
            True,

        "learn_top":
            False,

        "edge_importance":
            False,

        "temporal_kernel_size":
            None,

        "strategy":
            "uniform",

        "max_hops":
            8,

        "device":
            str(device),
    }


    # ========================================================
    # Construct model
    # ========================================================

    model = STG_NF(
        **model_args
    )


    model = model.to(
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


    model.load_state_dict(
        state_dict,
        strict=True,
    )


    # ========================================================
    # ActNorm
    # ========================================================

    num_actnorm = (
        mark_actnorm_initialized(
            model
        )
    )


    model.eval()


    return (
        model,
        checkpoint,
        model_args,
        num_actnorm,
    )


# ============================================================
# STG-NF forward
# ============================================================

def run_model(
    model,
    x,
):

    # STG-NF returns:
    #
    # z   = latent representation
    # nll = negative log likelihood

    z, nll = model(
        x
    )

    return (
        z,
        nll,
    )


# ============================================================
# Benchmark
# ============================================================

def benchmark_model(
    model,
    pose_shape,
    device,
):

    channels = pose_shape[
        0
    ]

    seg_len = pose_shape[
        1
    ]

    num_keypoints = pose_shape[
        2
    ]


    # --------------------------------------------------------
    # Random pose input
    #
    # [B, C, T, V]
    #
    # B = batch
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
    # Latency per individual pose window
    #
    # One forward contains BATCH_SIZE windows.
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
    # Throughput definitions
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
            "Benchmark STG-NF "
            "on CPU or GPU."
        )
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to pretrained "
            "STG-NF checkpoint."
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
    # Print device
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
        "MODEL: STG-NF"
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
        model_args,
        num_actnorm,

    ) = load_model(

        checkpoint_path=
            args.checkpoint,

        device=
            device,
    )


    # ========================================================
    # Configuration
    # ========================================================

    pose_shape = (
        model_args[
            "pose_shape"
        ]
    )


    channels = pose_shape[
        0
    ]

    seg_len = pose_shape[
        1
    ]

    num_keypoints = pose_shape[
        2
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
        f"Pose shape           : "
        f"{pose_shape}"
    )

    print(
        f"Segment length       : "
        f"{seg_len}"
    )

    print(
        f"Number of keypoints  : "
        f"{num_keypoints}"
    )

    print(
        f"Input shape          : "
        f"{input_shape}"
    )

    print(
        f"K                    : "
        f"{model_args['K']}"
    )

    print(
        f"L                    : "
        f"{model_args['L']}"
    )

    print(
        f"R                    : "
        f"{model_args['R']}"
    )

    print(
        f"Hidden channels      : "
        f"{model_args['hidden_channels']}"
    )

    print(
        f"ActNorm layers       : "
        f"{num_actnorm}"
    )

    print()


    # ========================================================
    # Benchmark
    # ========================================================

    result = benchmark_model(

        model=model,

        pose_shape=pose_shape,

        device=device,
    )


    # ========================================================
    # Store metadata
    # ========================================================

    result.update({

        "model":
            "STG-NF",

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


        "K":
            model_args[
                "K"
            ],

        "L":
            model_args[
                "L"
            ],

        "R":
            model_args[
                "R"
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