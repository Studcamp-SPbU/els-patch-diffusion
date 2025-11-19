import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from utils.data import get_dataset


def denorm(x, mean, std):
    """Обратно к [0,1] из нормализованного пространства."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return x * std_t + mean_t


def image_to_patches(x: torch.Tensor, kernel_size: int = 7, stride: int = 1):
    """
    x: [1, C, H, W]
    -> patches: [N_patches, C*k*k], L: число патчей (для обратной сборки)
    """
    B, C, H, W = x.shape
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
    # [B, C*k*k, L]
    P = unfold(x)
    B, CK2, L = P.shape
    patches = P.transpose(1, 2).reshape(B * L, CK2)
    return patches, L


def patches_to_image(
    patches: torch.Tensor,
    L: int,
    image_size: int,
    kernel_size: int = 7,
    stride: int = 1,
    channels: int = 1,
):
    """
    patches: [N_patches, C*k*k]
    -> img: [1, C, H, W]
    """
    B = 1
    CK2 = channels * kernel_size * kernel_size

    P = patches.view(B, L, CK2).transpose(1, 2)  # [B, CK2, L]

    fold = nn.Fold(
        output_size=(image_size, image_size),
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

    img = fold(P)

    # нормировка по количеству перекрытий
    ones = torch.ones(1, CK2, L, device=patches.device)
    norm = fold(ones)
    img = img / norm.clamp_min(1e-8)

    return img


@torch.no_grad()
def build_patch_database(
    dataset,
    kernel_size: int = 7,
    batch_size: int = 64,
    max_images: int | None = 500,
    device: str | torch.device = "cpu",
):
    device = torch.device(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_patches = []
    images_used = 0

    # сколько всего картинок планируем обработать
    total_imgs = max_images if max_images is not None else len(dataset)

    pbar = tqdm(total=total_imgs, desc="Building patch DB")

    for imgs, _ in loader:
        if max_images is not None and images_used >= max_images:
            break

        b = imgs.size(0)
        # чтобы не выйти за предел max_images
        if max_images is not None and images_used + b > max_images:
            b = max_images - images_used
            imgs = imgs[:b]

        imgs = imgs.to(device)
        B, C, H, W = imgs.shape

        unfold = nn.Unfold(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        P = unfold(imgs)  # [B, C*k*k, L]
        B, CK2, L = P.shape
        patches = P.transpose(1, 2).reshape(B * L, CK2)
        all_patches.append(patches.cpu())

        images_used += b
        pbar.update(b)

    pbar.close()

    db_patches = torch.cat(all_patches, dim=0)
    return db_patches


@torch.no_grad()
def knn_search_patches(
    query_patches: torch.Tensor,
    db_patches: torch.Tensor,
    device: str | torch.device = "cpu",
    q_batch: int = 32,
):
    device = torch.device(device)
    query_patches = query_patches.to(device)
    db_patches = db_patches.to(device)

    N_q = query_patches.size(0)
    N_db = db_patches.size(0)
    print(f"Searching kNN: N_q={N_q}, N_db={N_db}")

    indices_list = []

    pbar = tqdm(range(0, N_q, q_batch), desc="kNN search")

    for start in pbar:
        end = min(start + q_batch, N_q)
        q = query_patches[start:end]

        dist = torch.cdist(q, db_patches)  # [q_batch, N_db]
        _, idx = torch.min(dist, dim=1)
        indices_list.append(idx.cpu())

    indices = torch.cat(indices_list, dim=0)
    return indices


@torch.no_grad()
def generate_from_noise_knn(
    noise_image: torch.Tensor,
    db_patches: torch.Tensor,
    image_size: int,
    channels: int,
    kernel_size: int = 7,
    device: str | torch.device = "cpu",
):
    """
    noise_image: [1, C, H, W] (нормальный шум в пространстве датасета)
    """
    device = torch.device(device)
    noise_image = noise_image.to(device)
    db_patches = db_patches.to(device)

    # режем шумное изображение на патчи
    q_patches, L = image_to_patches(noise_image, kernel_size=kernel_size, stride=1)

    # для каждого патча ищем ближайший в базе
    nn_indices = knn_search_patches(q_patches, db_patches, device=device)
    nearest_patches = db_patches[nn_indices]  # [N_q, C*k*k]

    # склеиваем обратно в картинку
    recon = patches_to_image(
        nearest_patches,
        L=L,
        image_size=image_size,
        kernel_size=kernel_size,
        stride=1,
        channels=channels,
    )
    return recon


def main():
    os.makedirs("knn_results", exist_ok=True)

    dataset, meta = get_dataset("fashion_mnist", root="./data")
    image_size = meta["image_size"]  # 32
    channels = meta["num_channels"]  # 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # база патчей
    kernel_size = 7
    max_images = 10000  # можно увеличить, если памяти хватает
    print(f"Building patch DB (k={kernel_size}, max_images={max_images})...")
    db_patches = build_patch_database(
        dataset,
        kernel_size=kernel_size,
        batch_size=64,
        max_images=max_images,
        device=device,
    )
    print("DB patches shape:", db_patches.shape)

    # берём нормальный шум
    noise = torch.randn(1, channels, image_size, image_size)

    # kNN-проекция шума на manifold патчей
    knn_img = generate_from_noise_knn(
        noise_image=noise,
        db_patches=db_patches,
        image_size=image_size,
        channels=channels,
        kernel_size=kernel_size,
        device=device,
    )

    # денормализуем для отображения
    noise_denorm = denorm(noise[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()
    knn_denorm = denorm(knn_img[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()

    # сохраняем
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(noise_denorm, cmap="gray")
    ax[0].set_title("Исходный шум")
    ax[0].axis("off")

    ax[1].imshow(knn_denorm, cmap="gray")
    ax[1].set_title("kNN-патчи из FashionMNIST")
    ax[1].axis("off")

    plt.tight_layout()
    out_path = "knn_results/knn_from_noise.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
