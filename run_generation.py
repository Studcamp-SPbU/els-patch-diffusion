import argparse
import torch

from utils.data import get_dataset
from methods.knn_torch import run_multiscale_knn_torch
from methods.knn_faiss import run_multiscale_knn_faiss
from methods.els import run_els_single, run_els_pairs
from utils.functions import load_kernel_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified generation script: kNN (torch/FAISS) & ELS"
    )
    parser.add_argument(
        "--method",
        choices=["knn", "faiss", "els"],
        required=True,
        help="Способ генерации: knn / faiss / els",
    )
    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "cifar10"],
        default="fashion_mnist",
        help="Имя датасета для get_dataset",
    )
    parser.add_argument(
        "--seed-path",
        type=str,
        default="seed.pt",
        help="Путь к seed для kNN методов (если нет — сгенерируется шум)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Максимум картинок из train для базы патчей (kNN методы)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=3,
        help="k для kNN",
    )
    parser.add_argument(
        "--pairs-per-class",
        type=int,
        default=0,
        help="Для ELS: если >0, генерируем столько пар на класс; если 0 — один пример",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=3,
        help="Для одиночной ELS-генерации: id класса",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset, meta = get_dataset(args.dataset, root="./data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", meta["name"])

    kernel_schedule = load_kernel_schedule("files/scales_FashionMNIST_UNet_zeros_conditonal.pt")


    if args.method == "knn":
        run_multiscale_knn_torch(
            dataset=dataset,
            meta=meta,
            device=device,
            kernel_schedule=kernel_schedule,
            max_images=args.max_images,
            k_neighbors=args.k_neighbors,
            seed_path=args.seed_path,
            output_dir="knn_results_torch",
        )

    elif args.method == "faiss":
        run_multiscale_knn_faiss(
            dataset=dataset,
            meta=meta,
            device=device,
            kernel_schedule=kernel_schedule,
            max_images=args.max_images,
            k_neighbors=args.k_neighbors,
            seed_path=args.seed_path,
            output_dir="knn_results_faiss",
        )

    elif args.method == "els":
        if args.pairs_per_class > 0:
            run_els_pairs(
                dataset_name=args.dataset,
                device=device,
                pairs_per_class=args.pairs_per_class,
                output_dir=f"els_pairs_{args.dataset}",
            )
        else:
            run_els_single(
                dataset_name=args.dataset,
                device=device,
                class_id=args.class_id,
                output_dir=f"els_single_{args.dataset}",
            )


if __name__ == "__main__":
    main()
