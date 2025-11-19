import torch
import matplotlib.pyplot as plt

from utils.data import get_dataset
from utils.noise_schedules import cosine_noise_schedule
from utils.idealscore import (
    LocalEquivBordersScoreModule,
    ScheduledScoreMachine,
)


def denorm(x, mean, std):
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return x * std_t + mean_t


def main():
    # 1. Датасет FashionMNIST
    # dataset, meta = get_dataset("fashion_mnist", root="./data")
    dataset, meta = get_dataset("cifar10", root="./data")
    image_size = meta["image_size"]
    in_channels = meta["num_channels"]
    num_classes = meta["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2. ELS-backbone — берём весь датасет
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

    # 3. Загрузка масштабов 
    # raw_scales = torch.load("./files/scales_FashionMNIST_ResNet_zeros_conditonal.pt")
    raw_scales = torch.load("./files/scales_CIFAR10_ResNet_zeros_conditional.pt")

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

    # ГЕНЕРАЦИЯ 

    class_id = 3
    label = torch.tensor([class_id], device=device)
    print("Class id:", class_id)

    seed = torch.randn(1, in_channels, image_size, image_size, device=device)


    # # грузим шум из файла:
    # noise = torch.load("seed.pt")   
    # if isinstance(noise, dict):
    #     noise = noise["x"]
    # noise = noise.to(device)
    # if noise.dim() == 3:
    #     noise = noise.unsqueeze(0)  
    # seed = noise  



    img = machine(seed.clone(), nsteps=len(scales), label=label, device=device)
    img = img.detach().cpu() 

    # 5. Ищем ближайший train-пример (для интереса)
    with torch.no_grad():
        gen = img[0]                    
        gen_flat = gen.view(1, -1)

        all_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))])  
        all_flat = all_imgs.view(len(dataset), -1)

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
    plt.show()


if __name__ == "__main__":
    main()
