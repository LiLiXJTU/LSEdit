import argparse
import datetime
import time

import torch


BYTES_PER_MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor and reserve GPU memory.")
    parser.add_argument(
        "--mb",
        type=int,
        default=71680,
        help="GPU memory to reserve in MiB. Default: 71680, about 70 GiB.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=86400,
        help="How long to keep the memory allocated. Default: 86400, 24 hours.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=4,
        help="CUDA device index. Default: 4.",
    )
    parser.add_argument(
        "--threshold-mb",
        type=int,
        default=71680,
        help="Start reserving only when free memory is greater than this MiB value. Default: 71680.",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=1200,
        help="Seconds between free-memory checks. Default: 1200, 20 minutes.",
    )
    parser.add_argument(
        "--utilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep running GPU operations after memory is reserved. Default: enabled.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=20,
        help="Synchronize after this many GPU operation loops. Default: 20.",
    )
    return parser.parse_args()


def has_enough_free_memory(free_bytes, threshold_bytes):
    return free_bytes > threshold_bytes


def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def allocate_memory(mb, device):
    bytes_to_reserve = mb * BYTES_PER_MIB
    elements = bytes_to_reserve // 4
    tensor = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
    tensor.fill_(1.0)
    torch.cuda.synchronize()
    return tensor


def wait_until_memory_is_available(device, threshold_mb, check_interval):
    threshold_bytes = threshold_mb * BYTES_PER_MIB

    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mb = free_bytes // BYTES_PER_MIB
        total_mb = total_bytes // BYTES_PER_MIB

        if has_enough_free_memory(free_bytes, threshold_bytes):
            log(f"cuda:{device} free memory is {free_mb} MiB / {total_mb} MiB. Starting reservation.")
            return

        log(
            f"cuda:{device} free memory is {free_mb} MiB / {total_mb} MiB; "
            f"needs more than {threshold_mb} MiB. Checking again in {check_interval} seconds."
        )
        time.sleep(check_interval)


def keep_gpu_busy(tensor, seconds, sync_every):
    end_time = time.monotonic() + seconds
    loops = 0

    while time.monotonic() < end_time:
        tensor.mul_(1.000001).add_(0.000001)
        loops += 1

        if loops % sync_every == 0:
            torch.cuda.synchronize()

    torch.cuda.synchronize()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script needs an NVIDIA GPU with CUDA.")

    torch.cuda.set_device(args.device)
    wait_until_memory_is_available(args.device, args.threshold_mb, args.check_interval)

    tensor = allocate_memory(args.mb, args.device)
    log(f"Reserved about {args.mb} MiB on cuda:{args.device} for {args.seconds} seconds.")
    log("Press Ctrl+C to release early.")

    try:
        if args.utilization:
            log("GPU utilization mode is enabled.")
            keep_gpu_busy(tensor, args.seconds, args.sync_every)
        else:
            time.sleep(args.seconds)
    except KeyboardInterrupt:
        log("Releasing GPU memory.")
    finally:
        del tensor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
