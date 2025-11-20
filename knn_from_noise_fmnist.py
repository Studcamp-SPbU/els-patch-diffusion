import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from utils.data import get_dataset


def get_next_image_path(folder: str, prefix="img_", ext=".png"):
    """Формируем название картинки в папке."""
    os.makedirs(folder, exist_ok=True)
    # считываем существующие файлы
    files = os.listdir(folder)
    nums = []

    for f in files:
        if f.startswith(prefix) and f.endswith(ext):
            try:
                n = int(f[len(prefix) : -len(ext)])
                nums.append(n)
            except:
                pass

    next_num = (max(nums) + 1) if nums else 1
    filename = f"{prefix}{next_num:03d}{ext}"
    return os.path.join(folder, filename)


def load_seed_or_random(path: str, channels: int, image_size: int, device):
    """
    Если существует файл path:
        • загружает seed
        • приводит к форме [1, C, H, W]
    Иначе:
        • создаёт torch.randn(...)
    """

    if os.path.exists(path):
        print(f"Loading seed from {path}...")
        seed = torch.load(path, map_location=device)

        # если seed упакован как dict
        if isinstance(seed, dict):
            if "x" in seed:
                seed = seed["x"]
            else:
                # берём первое значение
                seed = next(iter(seed.values()))

        # если [C, H, W]
        if seed.dim() == 3:
            seed = seed.unsqueeze(0)

        # если размер не совпадает — ошибка
        if (
            seed.shape[1] != channels
            or seed.shape[2] != image_size
            or seed.shape[3] != image_size
        ):
            raise ValueError(
                f"Seed shape mismatch: expected [1,{channels},{image_size},{image_size}], got {tuple(seed.shape)}"
            )

        seed = seed.to(device)
        print("Seed loaded.")
        return seed

    # если файла нет — генерируем шум
    print(f"No seed file found at {path}. Generating random noise.")
    return torch.randn(1, channels, image_size, image_size, device=device)


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

    # размеры патчей на каждом шаге
    kernel_schedule = [
        17,
        17,
        15,
        15,
        15,
        13,
        13,
        13,
        11,
        11,
        11,
        9,
        9,
        9,
        7,
        7,
        5,
        5,
        3,
        3,
    ]

    max_images = 500

    # стартовый шум
    x = torch.randn(1, channels, image_size, image_size)
    x0 = x.clone()  # сохраним для сравнения

    # кэш баз патчей по размеру окна
    db_cache: dict[int, torch.Tensor] = {}

    for step, k in enumerate(kernel_schedule, start=1):
        print(f"\n=== Step {step}/{len(kernel_schedule)} — kernel_size={k} ===")

        if k not in db_cache:
            print(f"Building patch DB for k={k}...")
            db_patches = build_patch_database(
                dataset,
                kernel_size=k,
                batch_size=64,
                max_images=max_images,
                device=device,
            )
            db_cache[k] = db_patches
            print(f"DB[k={k}] patches shape:", db_patches.shape)
        else:
            db_patches = db_cache[k]
            print(f"Reusing patch DB for k={k}, shape={db_patches.shape}")

        # проекция текущей картинки x на manifold k×k-патчей
        x = generate_from_noise_knn(
            noise_image=x,
            db_patches=db_patches,
            image_size=image_size,
            channels=channels,
            kernel_size=k,
            device=device,
        )

    # исходный шум vs результат после всех проходов
    noise_denorm = denorm(x0[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()
    final_denorm = denorm(x[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(noise_denorm, cmap="gray")
    ax[0].set_title("Исходный шум")
    ax[0].axis("off")

    ax[1].imshow(final_denorm, cmap="gray")
    ax[1].set_title("Мульти-скейл kNN-патчи")
    ax[1].axis("off")

    plt.tight_layout()

    folder = "knn_results"
    out_path = get_next_image_path(folder)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
