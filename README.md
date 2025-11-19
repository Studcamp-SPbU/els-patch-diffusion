# ELS Patch-Based Diffusion

Реализация Equivariant Local Score (ELS) по статье
**“An Analytic Theory of Creativity in Convolutional Diffusion Models”**
[https://arxiv.org/abs/2412.20292](https://arxiv.org/abs/2412.20292)

---

## О проекте

Этот репозиторий содержит полную практическую реализацию основных идей из статьи
*“An Analytic Theory of Creativity in Convolutional Diffusion Models” (2024)*,
где вводится аналитическая модель диффузии ELS (Equivariant Local Score).

ELS — это неконволюционная, патчевая, полностью аналитическая версия диффузионной модели, которая генерирует изображения, приближая идеальный скор-функционал через патчи тренировочного датасета.

Проект включает:

* реализацию модулей
  `LocalScoreModule`, `LocalEquivScoreModule`, `LocalEquivBordersScoreModule`
* реализацию шага обратной диффузии
  (`ScheduledScoreMachine`)
* запуск генерации на датасетах
* поиск ближайшего реального изображения для анализа качества
* готовые файлы со скейлами патчей из оригинального кода

---

## Структура проекта

```
project/
│── run_els.py                 # пример генерации 
│── pairs_run_els_fmnist.py    # генерация пар ELS + nearest neighbor
│
├── utils/
│   ├── data.py                # загрузка датасетов + метаданные
│   ├── idealscore.py          # реализация ELS / LS / Ideal score
│   └── noise_schedules.py     # β(t) schedules
│
├── files/                     # предобученные файлы со скейлами P(t)
    ├── scales_FashionMNIST_ResNet_zeros_conditional.pt
    ├── scales_CIFAR10_ResNet_zeros_conditional.pt
    └── ...
```

---

# Запуск проекта

## 1. Установка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```
---

## 2. Подготовка данных

Данные не лежат в репозитории, их нужно скачать автоматически. Просто запустить скрипт – он сам скачает:

```bash
python3 run_els.py
```

---

## 3. Генерация изображения через ELS 

```bash
python3 run_els.py
```

Скрипт:

* создаёт seed шум `x_T`
* запускает обратную диффузию через `ScheduledScoreMachine`
* восстанавливает изображение класса `class_id`
* ищет ближайший train-пример для анализа качества
* отображает 2 изображения: ELS-сэмпл и nearest neighbor

---

## 4. Использование сохранённого шума

Можно сохранить шум и потом использовать в скрипте, раскомментировав соответствующие строки.

---

## 5. Генерация пар «ELS изображение + ближайший train»

```bash
python3 pairs_run_els_fmnist.py
```

Скрипт сохранит картинки в папку:

```
els_pairs_fmnist/
    class3_pair0.png
    class3_pair1.png
    ...
```

---

# Ссылки

* **Статья**
  [https://arxiv.org/abs/2412.20292](https://arxiv.org/abs/2412.20292)

