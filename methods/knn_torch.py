import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.functions import (
    load_seed_or_random,
    image_to_patches,
    patches_to_image,
    save_comparison_figure
)


@torch.no_grad()
def build_patch_database_torch(
    dataset,
    kernel_size: int = 7,
    batch_size: int = 64,
    max_images: int | None = 500,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Собирает базу патчей [N_db, C*k*k] для заданного kernel_size.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_patches = []
    images_used = 0
    total_imgs = max_images if max_images is not None else len(dataset)

    pbar = tqdm(total=total_imgs, desc=f"Building patch DB (k={kernel_size})")

    for imgs, _ in loader:
        if max_images is not None and images_used >= max_images:
            break

        b = imgs.size(0)
        if max_images is not None and images_used + b > max_images:
            b = max_images - images_used
            imgs = imgs[:b]

        imgs = imgs.to(device)
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
def knn_search_patches_torch(
    query_patches: torch.Tensor,
    db_patches: torch.Tensor,
    k: int = 3,
    device: torch.device = torch.device("cpu"),
    q_batch: int = 32,
) -> torch.Tensor:
    """
    query_patches: [N_q, D]
    db_patches:    [N_db, D]
    ->
        indices:   [N_q, k] индексы k ближайших патчей в базе
    """
    query_patches = query_patches.to(device)
    db_patches = db_patches.to(device)

    N_q = query_patches.size(0)
    N_db = db_patches.size(0)
    print(f"Searching kNN (torch): N_q={N_q}, N_db={N_db}, k={k}")

    indices_list = []

    for start in tqdm(range(0, N_q, q_batch), desc="kNN search"):
        end = min(start + q_batch, N_q)
        q = query_patches[start:end]  # [q_batch, D]

        dist = torch.cdist(q, db_patches)  # [q_batch, N_db]
        _, idxs = torch.topk(dist, k, dim=1, largest=False)  # [q_batch, k]
        indices_list.append(idxs.cpu())

    indices = torch.cat(indices_list, dim=0)  # [N_q, k]
    return indices


@torch.no_grad()
def project_image_knn_torch(
    x: torch.Tensor,
    db_patches: torch.Tensor,
    image_size: int,
    channels: int,
    kernel_size: int,
    k_neighbors: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Один шаг проекции изображения x через torch.cdist kNN.
    """
    x = x.to(device)
    db_patches = db_patches.to(device)

    q_patches, L = image_to_patches(
        x,
        kernel_size=kernel_size,
        stride=1,
    )  # [N_q, D]

    nn_indices = knn_search_patches_torch(
        query_patches=q_patches,
        db_patches=db_patches,
        k=k_neighbors,
        device=device,
    )  # [N_q, k]

    neighbors = db_patches[nn_indices.to(db_patches.device)]  # [N_q, k, D]
    avg_patches = neighbors.mean(dim=1)  # [N_q, D]

    recon_img = patches_to_image(
        avg_patches,
        L=L,
        image_size=image_size,
        kernel_size=kernel_size,
        stride=1,
        channels=channels,
    )
    return recon_img


def run_multiscale_knn_torch(
    dataset,
    meta,
    device: torch.device,
    kernel_schedule,
    max_images: int,
    k_neighbors: int,
    seed_path: str,
    output_dir: str = "knn_results_torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Мульти-скейл генерация через kNN (torch.cdist).
    Возвращает (final_image, initial_noise).
    """
    image_size = meta["image_size"]
    channels = meta["num_channels"]

    x = load_seed_or_random(seed_path, channels, image_size, device)
    x0 = x.clone()

    db_cache: dict[int, torch.Tensor] = {}

    for step, k in enumerate(kernel_schedule, start=1):
        print(
            f"\n=== [torch kNN] Step {step}/{len(kernel_schedule)} — kernel_size={k} ==="
        )

        if k not in db_cache:
            db_patches = build_patch_database_torch(
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

        x = project_image_knn_torch(
            x,
            db_patches=db_patches,
            image_size=image_size,
            channels=channels,
            kernel_size=k,
            k_neighbors=k_neighbors,
            device=device,
        )

    save_comparison_figure(
        x0=x0,
        x_final=x,
        meta=meta,
        title_final="Multiscale kNN (torch)",
        output_dir=output_dir,
        prefix="knn_torch_",
    )

    return x, x0
