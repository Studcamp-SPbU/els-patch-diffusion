import os
import matplotlib.pyplot as plt
import torch

from utils.data import get_dataset
from utils.noise_schedules import cosine_noise_schedule
from utils.idealscore import (
    LocalEquivBordersScoreModule,
    ScheduledScoreMachine,
)
from utils.functions import denorm, get_next_image_path, load_kernel_schedule


def build_els_machine(dataset_name: str, device: torch.device):
    """
    Собирает dataset, meta, backbone + ScheduledScoreMachine для ELS.
    Возвращает (machine, scales, dataset, meta).
    """
    dataset, meta = get_dataset(dataset_name, root="./data")
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]

    backbone = LocalEquivBordersScoreModule(
        dataset=dataset,
        kernel_size=3,
        batch_size=64,
        image_size=image_size,
        channels=in_channels,
        schedule=cosine_noise_schedule,
        max_samples=None,
        shuffle=False,
    )

    if dataset_name == "fashion_mnist":
        scales_path = "./files/scales_FashionMNIST_ResNet_zeros_conditonal.pt"
    elif dataset_name == "cifar10":
        scales_path = "./files/scales_CIFAR10_ResNet_zeros_conditional.pt"
    else:
        raise ValueError(f"Unknown dataset for ELS: {dataset_name}")

    scales = load_kernel_schedule(scales_path)
    if isinstance(scales, torch.Tensor):
        scales = scales.int().tolist()
    elif isinstance(scales, (list, tuple)):
        scales = [int(s.item() if hasattr(s, "item") else s) for s in scales]
    else:
        raise TypeError(f"Unexpected scales type: {type(scales)}")

    machine = ScheduledScoreMachine(
        backbone,
        in_channels=in_channels,
        imsize=image_size,
        default_time_steps=len(scales),
        noise_schedule=cosine_noise_schedule,
        score_backbone=True,
        scales=scales,
    ).to(device)

    return machine, scales, dataset, meta


def run_els_single(
    dataset_name: str,
    device: torch.device,
    class_id: int,
    output_dir: str = "els_single",
) -> None:
    """
    Генерирует одну ELS-картинку заданного класса и ищет ближайший train пример.
    """
    os.makedirs(output_dir, exist_ok=True)

    machine, scales, dataset, meta = build_els_machine(dataset_name, device)
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]

    label = torch.tensor([class_id], device=device)
    print("Class id:", class_id)

    seed = torch.randn(1, in_channels, image_size, image_size, device=device)

    with torch.no_grad():
        img = machine(seed.clone(), nsteps=len(scales), label=label, device=device)
    img = img.detach().cpu()
    gen = img[0]

    # stack train images for nearest neighbor search
    print("Stacking train images for nearest neighbor...")
    with torch.no_grad():
        all_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))])
        all_flat = all_imgs.view(len(dataset), -1)
        gen_flat = gen.view(1, -1)
        dists = torch.norm(all_flat - gen_flat, dim=1)
        min_dist, min_idx = torch.min(dists, dim=0)
        nearest_img = all_imgs[min_idx]

    print(f"Nearest train index: {min_idx.item()}, dist: {min_dist.item():.4f}")

    gen_denorm = denorm(gen, meta["mean"], meta["std"]).cpu()
    nearest_denorm = denorm(nearest_img, meta["mean"], meta["std"]).cpu()

    if gen_denorm.dim() == 3:
        gen_denorm = gen_denorm.permute(1, 2, 0)
    if nearest_denorm.dim() == 3:
        nearest_denorm = nearest_denorm.permute(1, 2, 0)

    gen_denorm = gen_denorm.numpy()
    nearest_denorm = nearest_denorm.numpy()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    if gen_denorm.shape[-1] == 1:
        ax[0].imshow(gen_denorm[..., 0], cmap="gray")
    else:
        ax[0].imshow(gen_denorm)
    ax[0].set_title(f"ELS sample (class {class_id})")
    ax[0].axis("off")

    if nearest_denorm.shape[-1] == 1:
        ax[1].imshow(nearest_denorm[..., 0], cmap="gray")
    else:
        ax[1].imshow(nearest_denorm)
    ax[1].set_title(f"Nearest train #{min_idx.item()}")
    ax[1].axis("off")

    plt.tight_layout()
    out_path = get_next_image_path(output_dir, prefix=f"class{class_id}_", ext=".png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def run_els_pairs(
    dataset_name: str,
    device: torch.device,
    pairs_per_class: int,
    output_dir: str = "els_pairs",
) -> None:
    """
    Генерирует по pairs_per_class пар (ELS + ближайший train) для каждого класса.
    """
    os.makedirs(output_dir, exist_ok=True)

    machine, scales, dataset, meta = build_els_machine(dataset_name, device)
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]
    num_classes = meta["num_classes"]

    print("Stacking train images...")
    with torch.no_grad():
        all_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))])
        all_flat = all_imgs.view(len(dataset), -1)

    for class_id in range(num_classes):
        print(f"\nGenerating for class {class_id}...")
        for j in range(pairs_per_class):
            label = torch.tensor([class_id], device=device)
            seed = torch.randn(1, in_channels, image_size, image_size, device=device)

            with torch.no_grad():
                img = machine(
                    seed.clone(), nsteps=len(scales), label=label, device=device
                )

            img = img.detach().cpu()
            gen = img[0]

            with torch.no_grad():
                gen_flat = gen.view(1, -1)
                dists = torch.norm(all_flat - gen_flat, dim=1)
                min_dist, min_idx = torch.min(dists, dim=0)
                nearest_img = all_imgs[min_idx]

            gen_denorm = denorm(gen, meta["mean"], meta["std"]).squeeze().numpy()
            nearest_denorm = (
                denorm(nearest_img, meta["mean"], meta["std"]).squeeze().numpy()
            )

            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(gen_denorm, cmap="gray")
            ax[0].set_title(f"ELS (class {class_id})")
            ax[0].axis("off")

            ax[1].imshow(nearest_denorm, cmap="gray")
            ax[1].set_title(f"Nearest #{min_idx.item()}")
            ax[1].axis("off")

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"class{class_id}_pair{j}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"  Saved {out_path}     (dist={min_dist.item():.4f})")
