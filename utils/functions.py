import os
import torch
import torch.nn as nn


def get_next_image_path(folder: str, prefix: str = "img_", ext: str = ".png") -> str:
    """Формируем следующее имя файла в папке."""
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(folder)
    nums = []

    for f in files:
        if f.startswith(prefix) and f.endswith(ext):
            try:
                n = int(f[len(prefix) : -len(ext)])
                nums.append(n)
            except Exception:
                pass

    next_num = (max(nums) + 1) if nums else 1
    filename = f"{prefix}{next_num:03d}{ext}"
    return os.path.join(folder, filename)


def load_seed_or_random(
    path: str, channels: int, image_size: int, device: torch.device
) -> torch.Tensor:
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

        if isinstance(seed, dict):
            if "x" in seed:
                seed = seed["x"]
            else:
                seed = next(iter(seed.values()))

        if seed.dim() == 3:
            seed = seed.unsqueeze(0)

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

    print(f"No seed file found at {path}. Generating random noise.")
    return torch.randn(1, channels, image_size, image_size, device=device)


def denorm(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Обратно к [0,1] из нормализованного пространства."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return x * std_t + mean_t


def image_to_patches(
    x: torch.Tensor,
    kernel_size: int = 7,
    stride: int = 1,
) -> tuple[torch.Tensor, int]:
    """
    x: [1, C, H, W]
    -> patches: [N_patches, C*k*k], L: число патчей (для обратной сборки)
    """
    B, C, H, W = x.shape
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
    P = unfold(x)  # [B, C*k*k, L]
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
) -> torch.Tensor:
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

    ones = torch.ones(1, CK2, L, device=patches.device)
    norm = fold(ones)
    img = img / norm.clamp_min(1e-8)

    return img


def save_comparison_figure(
    x0: torch.Tensor,
    x_final: torch.Tensor,
    meta,
    title_final: str,
    output_dir: str,
    prefix: str = "img_",
) -> str:
    """
    Рисует картинку «исходный шум vs результат», сохраняет в файл и возвращает путь.
    """
    os.makedirs(output_dir, exist_ok=True)

    noise_denorm = denorm(x0[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()
    final_denorm = denorm(x_final[0], meta["mean"], meta["std"]).cpu().squeeze().numpy()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(noise_denorm, cmap="gray")
    ax[0].set_title("Исходный шум")
    ax[0].axis("off")

    ax[1].imshow(final_denorm, cmap="gray")
    ax[1].set_title(title_final)
    ax[1].axis("off")

    plt.tight_layout()

    out_path = get_next_image_path(output_dir, prefix=prefix, ext=".png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")
    return out_path


def load_kernel_schedule(FILE: str):
    raw = torch.load(FILE)
    if isinstance(raw, torch.Tensor):
        arr = raw.int().tolist()
    elif isinstance(raw, (list, tuple)):
        arr = [int(x.item() if hasattr(x, "item") else x) for x in raw]
    else:
        raise TypeError(f"Unsupported scales format: {type(raw)}")

    arr = list(reversed(arr))
    return arr
