# Система распознавания эмоций по изображению лица

Программная система распознавания эмоций человека по изображению лица на основе свёрточных нейронных сетей (CNN). Поддерживает обучение модели на датасете изображений и инференс в реальном времени с веб-камеры.

## Возможности

- Подготовка датасета: стратифицированное разбиение на train/val/test (70%/15%/15%)
- Обучение CNN-модели на изображениях 48×48 в градациях серого
- Распознавание 7 базовых эмоций: angry, disgust, fear, happy, neutral, sad, surprise
- Работа в реальном времени: детекция лиц (Haar Cascade) и классификация эмоций с выводом на видеокадр

## Требования

- Python 3.10+
- Веб-камера (для инференса в реальном времени)

## Установка

1. Клонируйте репозиторий и перейдите в каталог проекта:

```bash
cd vkr
```

2. Создайте виртуальное окружение (рекомендуется):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
pip install tensorflow seaborn
```

## Структура проекта

```
vkr/
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py   # Подготовка и разбиение датасета
│   │   └── dataloaders.py       # Генераторы для обучения
│   ├── models/
│   │   └── cnn_baseline.py      # Архитектура CNN
│   ├── utils/
│   │   └── metrics.py           # Метрики и матрица ошибок
│   ├── train.py                 # Обучение модели
│   └── inference.py             # Инференс с веб-камеры
├── data/
│   ├── raw/                     # Исходные изображения
│   └── processed/               # Обработанные train/val/test
├── results/
│   └── models/                  # Сохранённые модели (.h5)
├── detect_emotion.py            # Альтернативный скрипт инференса
└── requirements.txt
```

## Подготовка датасета

Разместите изображения в `src/data/raw/test/` в подпапках по классам:

```
src/data/raw/test/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Запустите подготовку:

```bash
python src/data/prepare_dataset.py
```

Результат: каталог `data/processed/` с разбиением train/val/test.

## Обучение модели

```bash
python src/train.py
```

Модель сохранится в `results/models/emotion_cnn.h5`.

## Распознавание эмоций в реальном времени

```bash
python src/inference.py
```

- Откроется окно с видеопотоком с камеры
- Найденные лица будут обведены, над ними отображается распознанная эмоция
- Выход: нажмите `q`

> **Примечание:** `detect_emotion.py` использует модель `emotion_detection_model.h5`. Рекомендуется использовать `src/inference.py` с моделью `results/models/emotion_cnn.h5`.

## Распознаваемые эмоции

| Класс    | Описание     |
|----------|--------------|
| angry    | Злость       |
| disgust  | Отвращение   |
| fear     | Страх        |
| happy    | Радость      |
| neutral  | Нейтральная  |
| sad      | Грусть       |
| surprise | Удивление    |

## Архитектура модели

CNN из 4 свёрточных блоков:

- Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
- Conv2D(256) → Conv2D(256) → MaxPool
- Flatten → Dense(512) → Dropout(0.5) → Dense(7, softmax)

Вход: изображение 48×48×1 (grayscale).

## Технологии

- **TensorFlow/Keras** — обучение и инференс
- **OpenCV** — детекция лиц, работа с видео
- **scikit-learn** — разбиение датасета
