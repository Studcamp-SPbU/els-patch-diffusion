import faiss
import torch
import torch.nn as nn

from utils.functions import (
    load_seed_or_random,
    image_to_patches,
    patches_to_image,
    save_comparison_figure
)


@torch.no_grad()
def build_faiss_index_for_kernel(
    dataset,
    kernel_size: int,
    batch_size: int = 64,
    max_images: int | None = 500,
    target_label: int | None = None,
) -> faiss.Index:
    """
    Стримингово строит FAISS IndexFlatL2 по патчам k*k.
    Если target_label не None — используем только картинки с этим label.
    """
    sample_img, _ = dataset[0]
    C, H, W = sample_img.shape
    dim = C * kernel_size * kernel_size

    index = faiss.IndexFlatL2(dim)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images_used = 0

    if target_label is None:
        print(f"Building FAISS index for k={kernel_size}, max_images={max_images} (all classes)...")
    else:
        print(
            f"Building FAISS index for k={kernel_size}, max_images={max_images}, "
            f"target_label={target_label}..."
        )

    for imgs, labels in loader:
        if target_label is not None:
            mask = (labels == target_label)
            if not mask.any():
                continue
            imgs = imgs[mask]

        b = imgs.size(0)
        if b == 0:
            continue

        if max_images is not None:
            if images_used >= max_images:
                break
            if images_used + b > max_images:
                b = max_images - images_used
                imgs = imgs[:b]

        unfold = nn.Unfold(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        patches = unfold(imgs)  
        B, CK2, L = patches.shape
        patches = patches.permute(0, 2, 1).reshape(B * L, CK2)  

        patches_np = patches.detach().contiguous().numpy().astype("float32")
        index.add(patches_np)

        images_used += b

    print(f"  FAISS index built: ntotal={index.ntotal}, dim={dim}, images_used={images_used}")
    return index


@torch.no_grad()
def faiss_knn_average_patches(
    query_patches: torch.Tensor,
    index: faiss.Index,
    k_neighbors: int,
    device: torch.device = torch.device("cpu"),
    q_batch: int = 4096,
) -> torch.Tensor:
    """
    query_patches: [N_q, D] (torch)
    ->
        avg_patches: [N_q, D] (torch)
    """
    q = query_patches.detach().float().cpu().numpy()
    N_q, D = q.shape

    all_avg = []

    for start in range(0, N_q, q_batch):
        end = min(start + q_batch, N_q)
        q_chunk = q[start:end]

        _, I = index.search(q_chunk, k_neighbors)  

        flat_indices = I.reshape(-1)
        neigh = index.reconstruct_batch(flat_indices)  
        neigh = neigh.reshape(-1, k_neighbors, D)  

        avg = neigh.mean(axis=1) 
        all_avg.append(torch.from_numpy(avg))

    avg_patches = torch.cat(all_avg, dim=0).to(device)
    return avg_patches


@torch.no_grad()
def project_image_knn_faiss(
    x: torch.Tensor,
    index: faiss.Index,
    image_size: int,
    channels: int,
    kernel_size: int,
    k_neighbors: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Один шаг проекции изображения x через FAISS kNN.
    """
    x = x.to(device)

    q_patches, L = image_to_patches(
        x,
        kernel_size=kernel_size,
        stride=1,
    ) 

    avg_patches = faiss_knn_average_patches(
        query_patches=q_patches,
        index=index,
        k_neighbors=k_neighbors,
        device=device,
    ) 

    recon_img = patches_to_image(
        avg_patches,
        L=L,
        image_size=image_size,
        kernel_size=kernel_size,
        stride=1,
        channels=channels,
    )
    return recon_img


def run_multiscale_knn_faiss(
    dataset,
    meta,
    device: torch.device,
    kernel_schedule,
    max_images: int,
    k_neighbors: int,
    seed_path: str,
    output_dir: str = "knn_results_faiss",
    class_label: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Мульти-скейл генерация через kNN (FAISS).
    Если class_label не None — база патчей строится по картинкам только этого класса.
    Возвращает (final_image, initial_noise).
    """
    image_size = meta["image_size"]
    channels = meta["num_channels"]

    x = load_seed_or_random(seed_path, channels, image_size, device)
    x0 = x.clone()

    index_cache: dict[tuple[int, int | None], faiss.Index] = {}

    for step, k in enumerate(kernel_schedule, start=1):
        print(
            f"\n=== [FAISS kNN] Step {step}/{len(kernel_schedule)} "
            f"— kernel_size={k}, class_label={class_label} ==="
        )

        cache_key = (k, class_label)

        if cache_key not in index_cache:
            index = build_faiss_index_for_kernel(
                dataset=dataset,
                kernel_size=k,
                batch_size=64,
                max_images=max_images,
                target_label=class_label,
            )
            index_cache[cache_key] = index
        else:
            index = index_cache[cache_key]
            print(
                f"Reusing FAISS index for k={k}, class_label={class_label}, "
                f"ntotal={index.ntotal}"
            )

        x = project_image_knn_faiss(
            x,
            index=index,
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
        title_final="Multiscale kNN (FAISS)",
        output_dir=output_dir,
        prefix="knn_faiss_",
    )

    return x, x0
