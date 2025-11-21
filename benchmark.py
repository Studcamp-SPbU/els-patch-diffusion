import time
import torch
from statistics import mean, stdev

from utils.data import get_dataset
from utils.noise_schedules import cosine_noise_schedule
from utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine


def benchmark_bbels(K_values, repeats=3):
    # 1. Load dataset & metadata
    dataset, meta = get_dataset("fashion_mnist", root="./data")
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2. Load scales
    raw_scales = torch.load("./files/scales_FashionMNIST_ResNet_zeros_conditonal.pt")
    if isinstance(raw_scales, torch.Tensor):
        scales = raw_scales.int().tolist()
    else:
        scales = [int(s) for s in raw_scales]

    # fixed class
    class_id = 3
    label = torch.tensor([class_id], device=device)

    # 3. Test loop
    for K in K_values:
        print(f"\n======================")
        print(f"   Testing K = {K}")
        print(f"======================")

        timings = []

        for rep in range(repeats):
            print(f"  Run {rep+1}/{repeats}...")

            # Build backbone with current K
            backbone = LocalEquivBordersScoreModule(
                dataset=dataset,
                kernel_size=3,
                batch_size=64,
                image_size=image_size,
                channels=in_channels,
                schedule=cosine_noise_schedule,
                max_samples=None,
                shuffle=False,
                topk=K,   # <-- HERE K is used
            )

            machine = ScheduledScoreMachine(
                backbone,
                in_channels=in_channels,
                imsize=image_size,
                default_time_steps=len(scales),
                noise_schedule=cosine_noise_schedule,
                score_backbone=True,
                scales=scales,
            ).to(device)

            seed = torch.randn(1, in_channels, image_size, image_size, device=device)

            # Time measurement
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                _ = machine(seed.clone(), nsteps=len(scales), label=label, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            print(f"    Time: {dt:.2f} sec")
            timings.append(dt)

        # Summaries
        print(f"\nRESULT for K={K}:")
        print(f"  Mean time: {mean(timings):.2f} s")
        if repeats > 1:
            print(f"  Std dev  : {stdev(timings):.2f} s")
        print("--------------------------")


if __name__ == "__main__":
    # Choose any K values you want to test
    K_values = [3]
    benchmark_bbels(K_values, repeats=3)