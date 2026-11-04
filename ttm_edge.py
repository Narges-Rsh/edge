import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

from utils.model_padding import PoseTTMVAD


# ============================================================
# Configuration
# ============================================================

WARMUP_ITERS = 50
LATENCY_ITERS = 200
THROUGHPUT_ITERS = 200

BATCH_SIZE = 1


EXPERIMENTS = [
    {
        "name": "DualMem-VAD",

        "checkpoint":
            "/home/nrashvan/ttm_test/Token/ttm_best.pth.tar",

        # Used ONLY if ttm_sequence_length is not stored
        # in the checkpoint.
        "ttm_steps_fallback": 4,
    },
]


OUTPUT_CSV = "ttm_edge_benchmark"


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
# Load model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_config" not in checkpoint:

        raise KeyError(
            f"No model_config found in checkpoint:\n"
            f"{checkpoint_path}"
        )


    config = checkpoint[
        "model_config"
    ]


    # --------------------------------------------------------
    # Reconstruct model exactly as trained
    # --------------------------------------------------------

    model = PoseTTMVAD(

        num_keypoints=
            config["num_keypoints"],

        seg_len=
            config["seg_len"],

        input_dim=
            config.get(
                "input_dim",
                2,
            ),

        embed_dim=
            config["embed_dim"],

        process_size=
            config["process_size"],

        memory_size=
            config["memory_size"],

        num_layers=
            config["num_layers"],

        mlp_dim=
            config["mlp_dim"],

        num_heads=
            config["num_heads"],

        dropout=
            config.get(
                "dropout",
                0.0,
            ),

        memory_mode=
            config.get(
                "memory_mode",
                "TL",
            ),

        use_positional_embedding=True,

        num_static_slots=
            config["num_static_slots"],

        window_num_layers=
            config.get(
                "window_num_layers",
                2,
            ),
    )


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )


    model = model.to(
        device
    )

    model.eval()


    return (
        model,
        config,
        checkpoint,
    )


# ============================================================
# Model forward
# ============================================================

def run_model(
    model,
    x,
    valid_mask,
    checkpoint,
):

    recon_threshold = checkpoint.get(
        "recon_gate_threshold",
        None,
    )

    static_threshold = checkpoint.get(
        "static_gate_threshold",
        None,
    )


    # --------------------------------------------------------
    # Deployment/evaluation gating
    # --------------------------------------------------------

    if (
        recon_threshold is not None
        and
        static_threshold is not None
    ):

        return model(
            x,
            valid_mask=valid_mask,
            memory_update_mode="hard",
            recon_threshold=recon_threshold,
            static_threshold=static_threshold,
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return model(
        x,
        valid_mask=valid_mask,
        memory_update_mode="always",
    )



# ============================================================
# Benchmark
# ============================================================

def benchmark_model(
    model,
    checkpoint,
    num_steps,
    seg_len,
    input_dim,
    num_keypoints,
    device,
):

    # --------------------------------------------------------
    # Synthetic input
    #
    # [B, S, C, W, V]
    #
    # B = batch size
    # S = number of sequential windows
    # C = pose coordinate dimension
    # W = frames per window
    # V = number of keypoints
    # --------------------------------------------------------

    x = torch.randn(
        BATCH_SIZE,
        num_steps,
        input_dim,
        seg_len,
        num_keypoints,
        device=device,
        dtype=torch.float32,
    )


    # All synthetic windows are valid

    valid_mask = torch.ones(
        BATCH_SIZE,
        num_steps,
        dtype=torch.bool,
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
                valid_mask,
                checkpoint,
            )


    synchronize(device)


    # ========================================================
    # Latency
    #
    # Each measurement = one complete sequence forward.
    # ========================================================

    latency_ms = []


    print(
        f"Running {LATENCY_ITERS} "
        f"latency iterations..."
    )


    with torch.inference_mode():

        for _ in range(
            LATENCY_ITERS
        ):

            # Required for accurate CUDA timing.
            # Does nothing on CPU.
            synchronize(device)


            start = (
                time.perf_counter()
            )


            _ = run_model(
                model,
                x,
                valid_mask,
                checkpoint,
            )


            synchronize(device)


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
    # Amortized latency per pose window
    # --------------------------------------------------------

    latency_per_window_ms = (
        mean_latency_ms
        /
        (
            BATCH_SIZE
            *
            num_steps
        )
    )


    # ========================================================
    # Throughput
    #
    # One continuous timing interval.
    # ========================================================

    print(
        f"Running {THROUGHPUT_ITERS} "
        f"throughput iterations..."
    )


    with torch.inference_mode():

        synchronize(device)


        start = (
            time.perf_counter()
        )


        for _ in range(
            THROUGHPUT_ITERS
        ):

            _ = run_model(
                model,
                x,
                valid_mask,
                checkpoint,
            )


        synchronize(device)


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

    # Number of complete person sequences processed / second

    sequences_per_second = (
        THROUGHPUT_ITERS
        *
        BATCH_SIZE
        /
        total_time
    )


    # Number of individual pose windows processed / second

    windows_per_second = (
        THROUGHPUT_ITERS
        *
        BATCH_SIZE
        *
        num_steps
        /
        total_time
    )


    return {

        "mean_latency_ms":
            mean_latency_ms,


        "latency_per_window_ms":
            latency_per_window_ms,

        "sequences_per_second":
            sequences_per_second,

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
            "Benchmark DualMem-VAD "
            "on CPU or GPU."
        )
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
            "cpu, gpu, or cuda"
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

        device_label = "gpu"


    else:

        device = torch.device(
            "cpu"
        )

        device_label = "cpu"


    # ========================================================
    # Device information
    # ========================================================

    print()
    print("=" * 80)


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


    print("=" * 80)


    all_results = []


    # ========================================================
    # Benchmark experiments
    # ========================================================

    for experiment in EXPERIMENTS:

        name = experiment[
            "name"
        ]


        checkpoint_path = experiment[
            "checkpoint"
        ]


        print()
        print("=" * 80)


        print(
            f"MODEL: {name}"
        )


        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )


        print("=" * 80)


        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model, config, checkpoint = (
            load_model(
                checkpoint_path,
                device,
            )
        )


        # ====================================================
        # Read actual checkpoint configuration
        # ====================================================

        seg_len = int(
            config[
                "seg_len"
            ]
        )


        input_dim = int(
            config.get(
                "input_dim",
                2,
            )
        )


        num_keypoints = int(
            config[
                "num_keypoints"
            ]
        )


        # ----------------------------------------------------
        # Number of sequential TTM windows
        # ----------------------------------------------------

        if (
            "ttm_sequence_length"
            in config
        ):

            num_steps = int(
                config[
                    "ttm_sequence_length"
                ]
            )


            step_source = (
                "checkpoint"
            )


        else:

            num_steps = int(
                experiment[
                    "ttm_steps_fallback"
                ]
            )


            step_source = (
                "fallback"
            )


        # ====================================================
        # Input shape
        # ====================================================

        input_shape = (
            BATCH_SIZE,
            num_steps,
            input_dim,
            seg_len,
            num_keypoints,
        )


       



        # ====================================================
        # Print model configuration
        # ====================================================

        print()


        print(
            f"Segment length       : "
            f"{seg_len}"
        )


        print(
            f"TTM steps            : "
            f"{num_steps} "
            f"({step_source})"
        )


        print(
            f"Input shape          : "
            f"{input_shape}"
        )



        # ====================================================
        # Benchmark
        # ====================================================

        result = benchmark_model(

            model=model,

            checkpoint=checkpoint,

            num_steps=num_steps,

            seg_len=seg_len,

            input_dim=input_dim,

            num_keypoints=num_keypoints,

            device=device,
        )


        # ====================================================
        # Store metadata
        # ====================================================

        result.update({

            "model":
                name,

            "device":
                device_label,

            "checkpoint":
                checkpoint_path,

            "segment_length":
                seg_len,

            "ttm_steps":
                num_steps,

            "input_shape":
                str(
                    input_shape
                ),

            

        
        })


        all_results.append(
            result
        )


        # ====================================================
        # Individual results
        # ====================================================

        print()
        print("-" * 80)


        print(
            f"Execution device        : "
            f"{device_label.upper()}"
        )


        print(
            f"Mean latency / sequence : "
            f"{result['mean_latency_ms']:.3f} ms"
        )



        print(
            f"Latency / window        : "
            f"{result['latency_per_window_ms']:.3f} ms/window"
        )


        print(
            f"Sequences / sec         : "
            f"{result['sequences_per_second']:.2f}"
        )


        print(
            f"Windows / sec           : "
            f"{result['windows_per_second']:.2f}"
        )





        print("-" * 80)


        # ====================================================
        # Release model
        # ====================================================

        del model


        if device.type == "cuda":

            torch.cuda.empty_cache()


    # ========================================================
    # Final table
    # ========================================================

    print()
    print("=" * 145)

    print(
        "FINAL EDGE BENCHMARK"
    )

    print("=" * 145)


    header = (

        f"{'Model':<16}"

        f"{'Device':<9}"

        f"{'W':<6}"

        f"{'S':<6}"


        f"{'Seq Lat(ms)':<14}"

        f"{'Win Lat(ms)':<14}"

        f"{'Seq/s':<12}"

        f"{'Win/s':<12}"
    )


    print(
        header
    )


    print(
        "-" * 145
    )


    for r in all_results:

        print(

            f"{r['model']:<16}"

            f"{r['device']:<9}"

            f"{r['segment_length']:<6}"

            f"{r['ttm_steps']:<6}"

           
            f"{r['mean_latency_ms']:<14.3f}"

            f"{r['latency_per_window_ms']:<14.3f}"

            f"{r['sequences_per_second']:<12.2f}"

            f"{r['windows_per_second']:<12.2f}"
        )


    print("=" * 145)


    # ========================================================
    # Save CSV
    # ========================================================

    if all_results:

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
                    all_results[0].keys(),
            )


            writer.writeheader()


            writer.writerows(
                all_results
            )


        print(
            f"\nResults saved to: "
            f"{output_path.resolve()}"
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()