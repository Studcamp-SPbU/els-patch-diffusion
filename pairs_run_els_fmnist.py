import os
import torch
import matplotlib.pyplot as plt

from utils.data import get_dataset
from utils.noise_schedules import cosine_noise_schedule
from utils.idealscore import (
    LocalEquivBordersScoreModule,
    LocalEquivScoreModule,
    ScheduledScoreMachine,
)

OUTPUT_DIR = "els_pairs_fmnist"


def denorm(x, mean, std):
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return x * std_t + mean_t


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Датасет FashionMNIST
    dataset, meta = get_dataset("fashion_mnist", root="./data")
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]
    num_classes = meta["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2. ELS-backbone (патчи со всего датасета)
    backbone = LocalEquivScoreModule(
        dataset=dataset,
        kernel_size=3,
        batch_size=64,
        image_size=image_size,
        channels=in_channels,
        schedule=cosine_noise_schedule,
        max_samples=20000,
        shuffle=False,
        topk=64,
    )

    # 3. scales P(t)
    raw_scales = torch.load("./files/scales_FashionMNIST_ResNet_zeros_conditonal.pt")
    if isinstance(raw_scales, torch.Tensor):
        scales = raw_scales.int().tolist()
    elif isinstance(raw_scales, (list, tuple)):
        scales = [int(s.item() if hasattr(s, "item") else s) for s in raw_scales]
    else:
        raise TypeError(f"Unexpected scales type: {type(raw_scales)}")

    # 4. Машина обратной диффузии
    machine = ScheduledScoreMachine(
        backbone,
        in_channels=in_channels,
        imsize=image_size,
        default_time_steps=len(scales),
        noise_schedule=cosine_noise_schedule,
        score_backbone=True,
        scales=scales,
    ).to(device)

    # 5. Подготовить train-картинки один раз для поиска ближайшего
    print("Stacking train images...")
    with torch.no_grad():
        all_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))]) 
        all_flat = all_imgs.view(len(dataset), -1)

    #  ГЕНЕРАЦИЯ ПАР (5 на класс)

    pairs_per_class = 5

    for class_id in range(num_classes):
        print(f"\nGenerating for class {class_id}...")
        for j in range(pairs_per_class):
            label = torch.tensor([class_id], device=device)

            seed = torch.randn(1, in_channels, image_size, image_size, device=device)

            with torch.no_grad():
                img = machine(seed.clone(), nsteps=len(scales), label=label, device=device)

            img = img.detach().cpu()
            gen = img[0]  

            # nearest neighbor
            with torch.no_grad():
                gen_flat = gen.view(1, -1)
                dists = torch.norm(all_flat - gen_flat, dim=1)
                min_dist, min_idx = torch.min(dists, dim=0)
                nearest_img = all_imgs[min_idx]

            # денормализация
            gen_denorm = denorm(gen, meta["mean"], meta["std"]).squeeze().numpy()
            nearest_denorm = denorm(nearest_img, meta["mean"], meta["std"]).squeeze().numpy()

            # Сохранение пары
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(gen_denorm, cmap="gray")
            ax[0].set_title(f"ELS (class {class_id})")
            ax[0].axis("off")

            ax[1].imshow(nearest_denorm, cmap="gray")
            ax[1].set_title(f"Nearest #{min_idx.item()}")
            ax[1].axis("off")

            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, f"class{class_id}_pair{j}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"  Saved {out_path}     (dist={min_dist.item():.4f})")


if __name__ == "__main__":
    main()
