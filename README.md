# ECHO Attention Fusion Lab

## 1. Описание проекта

**ECHO Attention Fusion Lab** — лабораторная по мультимодальной бинарной классификации данных датасета ECHO.

В проекте исследуется, насколько эффективно задача классификации решается при использовании разных источников информации:

1. только текста;
2. только аудио;
3. совместного использования текста и аудио с помощью механизма Attention Fusion.

Каждый объект датасета содержит аудиозапись, текст и бинарную метку `label` со значением `0` или `1`.

Пример структуры `metadata.csv`:

```csv
audio_file,text,label,voice_gender
train_00000_male.mp3,"Example text",0,male
train_00001_female.mp3,"Another example",1,female
```

Основные поля:

| Поле | Описание |
|---|---|
| `audio_file` | имя аудиофайла |
| `text` | текст, связанный с аудиозаписью |
| `label` | целевой бинарный класс `0` или `1` |
| `voice_gender` | дополнительная метаинформация о голосе |

В текущей реализации `voice_gender` не используется как входной признак.

Для эксперимента используются данные из папок:

```text
english
french
spanish
```

В проекте сравниваются три модели:

- **Text-only** — использует только текст;
- **Audio-only** — использует только аудиопризнаки;
- **Attention Fusion** — объединяет признаки текста и аудио.

Качество моделей сравнивается с помощью:

- Accuracy;
- Macro-F1;
- Validation Loss.

Обучение выполняется с использованием PyTorch. Для веб-интерфейса и API используется FastAPI, а запуск проекта изолирован с помощью Docker.

---

## 2. Структура проекта

```text
ECHO/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── services.py
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   └── ECHO/
│       ├── english/
│       │   ├── train/
│       │   └── validation/
│       ├── french/
│       │   ├── train/
│       │   └── validation/
│       └── spanish/
│           ├── train/
│           └── validation/
├── artifacts/
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

### `app/config.py`

Содержит глобальные настройки проекта:

- путь к датасету;
- путь к моделям и результатам;
- sample rate;
- число MFCC;
- размер batch;
- количество эпох;
- learning rate;
- устройство обучения.

При доступной CUDA обучение выполняется на GPU.

### `app/data.py`

Отвечает за загрузку и предварительную обработку данных.

Основные компоненты:

- `ECHODataRepository` — чтение `metadata.csv`;
- `CharacterTokenizer` — преобразование текста в последовательность индексов;
- `AudioFeatureExtractor` — извлечение MFCC из аудио;
- `MultimodalDataset` — формирование объектов для PyTorch `DataLoader`.

Аудио:

1. загружается;
2. переводится в mono;
3. ресемплируется;
4. обрезается или дополняется до заданной длины;
5. преобразуется в MFCC;
6. нормализуется.

Часть предобработки выполняется на CPU, поэтому высокая загрузка процессора во время обучения является нормальной даже при использовании CUDA.

### `app/models.py`

Содержит архитектуры трёх исследуемых моделей:

```text
TextOnlyClassifier
AudioOnlyClassifier
AttentionFusionClassifier
```

### `app/services.py`

Содержит основную логику обучения и инференса.

`ModelFactory` создаёт выбранную архитектуру.

`TrainingService`:

- загружает train и validation;
- создаёт tokenizer;
- создаёт `DataLoader`;
- обучает три модели;
- вычисляет Accuracy, Macro-F1 и Loss;
- сохраняет модели;
- сохраняет историю обучения и метаданные.

`InferenceService`:

- загружает обученные модели;
- подготавливает новый текст и аудио;
- выполняет предсказание;
- возвращает вероятности классов;
- для Fusion-модели возвращает attention-информацию.

### `app/main.py`

FastAPI-приложение.

Основные маршруты:

```text
GET  /
GET  /api/results
POST /api/train
POST /api/predict
```

### `app/static/`

Содержит веб-интерфейс проекта:

- `index.html` — структура страницы;
- `styles.css` — стили;
- `app.js` — взаимодействие с API и отображение результатов.

### `artifacts/`

После обучения здесь сохраняются:

```text
text.pt
audio.pt
fusion.pt
metadata.json
```

### `train.py`

Отдельная точка запуска обучения без веб-интерфейса:

```bash
python train.py
```

### `Dockerfile`

Описывает Docker-образ приложения.

### `docker-compose.yml`

Используется для запуска приложения, подключения датасета, каталога `artifacts` и NVIDIA GPU.

---

## 3. Использованные методы

### 3.1 Text-only

Архитектура:

```text
Text
  ↓
Character Tokenizer
  ↓
Embedding
  ↓
Bidirectional GRU
  ↓
Text Embedding
  ↓
Linear Classifier
  ↓
Class 0 / Class 1
```

`Embedding` преобразует индексы символов в векторное представление.

Далее используется двунаправленный GRU, позволяющий учитывать контекст последовательности в обоих направлениях.

На выходе получается текстовый вектор признаков, который передаётся в линейный классификатор.

### 3.2 Audio-only

В аудиомодели исходный аудиосигнал сначала преобразуется в MFCC.

Архитектура:

```text
Audio
  ↓
MFCC
  ↓
Conv1D
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv1D
  ↓
BatchNorm
  ↓
ReLU
  ↓
AdaptiveAvgPool
  ↓
Audio Embedding
  ↓
Linear Classifier
  ↓
Class 0 / Class 1
```

MFCC используются как компактное спектральное представление аудиосигнала.

Свёрточная сеть выделяет локальные закономерности из последовательности MFCC.

### 3.3 Attention Fusion

Основной исследуемый метод проекта — мультимодальное объединение текстовых и аудиопризнаков.

Используется **Intermediate Fusion**, также называемый **Feature-Level Fusion**.

В отличие от Early Fusion, исходные текст и аудио не объединяются непосредственно до обработки.

В отличие от Late Fusion, модель не объединяет только готовые предсказания независимых классификаторов.

Сначала каждая модальность обрабатывается собственным энкодером:

```text
Text
  ↓
Text Encoder
  ↓
Text Embedding

Audio
  ↓
Audio Encoder
  ↓
Audio Embedding
```

После этого оба embedding приводятся к общей размерности:

```text
Text Embedding  ──→ Linear Projection ──→ Text Token
Audio Embedding ──→ Linear Projection ──→ Audio Token
```

Получаются два modality token:

```text
[text_token, audio_token]
```

К ним применяется **Multi-Head Self-Attention**:

```text
Text Token ─┐
            ├─→ Multi-Head Self-Attention
Audio Token ┘
                    ↓
             Residual Connection
                    ↓
                LayerNorm
                    ↓
               Mean Pooling
                    ↓
                   MLP
                    ↓
             Class 0 / Class 1
```

Attention позволяет модели учитывать взаимосвязь между текстовым и аудиопредставлением при формировании общего признакового вектора.

Attention-веса используются как дополнительная информация о распределении внимания между modality tokens и не трактуются как точный причинный процент вклада каждой модальности.

---

## 4. Запуск проекта

### Требования

Для запуска необходимы:

- Docker;
- Docker Compose;
- NVIDIA GPU для CUDA-обучения;
- WSL2 + Docker Desktop при запуске из Windows;
- доступ GPU из Docker.

### Сборка

```bash
docker compose build
```

### Запуск

```bash
docker compose up -d
```

веб-интерфейс будет доступен по адресу:

```text
http://localhost:8000
```
### Запуск обучения

Через API:

```http
POST /api/train?epochs=5&batch_size=16
```

Либо через CLI:

```bash
python train.py
```

---

## 5. Результаты

Эксперимент проводился в течение пяти эпох с `batch_size=16`.

Распределение классов:

```text
Train:
class 0 = 1552
class 1 = 1181

Validation:
class 0 = 313
class 1 = 236
```

Размеры выборок:

```text
Train samples      = 2733
Validation samples = 549
```

### Text-only

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.6625 | 0.6492 | 0.6393 | 0.6146 |
| 2 | 0.6256 | 0.6435 | **0.6630** | 0.6510 |
| 3 | 0.5986 | 0.6439 | 0.6375 | 0.6293 |
| 4 | 0.5766 | 0.6413 | 0.6521 | 0.6248 |
| 5 | 0.5465 | 0.6548 | 0.6594 | **0.6582** |

Итог пятой эпохи:

```text
Accuracy = 65.94 %
Macro-F1 = 65.82 %
Val Loss = 0.6548
```

Максимальная Accuracy Text-only:

```text
66.30 % на эпохе 2
```

### Audio-only

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.6796 | 0.6772 | 0.5738 | 0.5347 |
| 2 | 0.6656 | 0.6617 | 0.6029 | 0.5675 |
| 3 | 0.6545 | 0.6763 | 0.5902 | 0.4285 |
| 4 | 0.6371 | 0.6438 | 0.6211 | 0.5975 |
| 5 | 0.6217 | **0.6389** | **0.6612** | **0.6310** |

Итог пятой эпохи:

```text
Accuracy = 66.12 %
Macro-F1 = 63.10 %
Val Loss = 0.6389
```

### Attention Fusion

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.6630 | 0.6420 | 0.6412 | 0.6154 |
| 2 | 0.6320 | 0.6321 | 0.6503 | 0.6292 |
| 3 | 0.6071 | 0.6404 | 0.6375 | 0.6297 |
| 4 | 0.5958 | 0.6476 | **0.6758** | 0.6244 |
| 5 | 0.5774 | **0.6263** | 0.6703 | **0.6362** |

Итог пятой эпохи:

```text
Accuracy = 67.03 %
Macro-F1 = 63.62 %
Val Loss = 0.6263
```

Максимальная Accuracy всего эксперимента:

```text
67.58 % на эпохе 4
```

### Сравнение моделей

Для единообразного сравнения используются результаты пятой эпохи.

| Модель | Validation Accuracy | Macro-F1 | Validation Loss |
|---|---:|---:|---:|
| Text-only | 65.94 % | **65.82 %** | 0.6548 |
| Audio-only | 66.12 % | 63.10 % | 0.6389 |
| Attention Fusion | **67.03 %** | 63.62 % | **0.6263** |

По итоговой Accuracy лучший результат показала **Attention Fusion**:

```text
67.03 %
```

Она же получила минимальный Validation Loss:

```text
0.6263
```

По Macro-F1 лучший результат пятой эпохи показала **Text-only**:

```text
65.82 %
```

Если рассматривать максимальную Accuracy за все эпохи, лучший результат также принадлежит Attention Fusion:

```text
67.58 % на эпохе 4
```

## 6. Вывод

В работе были реализованы и сравнены три подхода:

```text
Text-only
Audio-only
Attention Fusion
```

Text-only показала лучший итоговый Macro-F1.

Audio-only показала, что аудиомодальность сама по себе содержит полезную информацию для классификации.

Attention Fusion показала лучшую итоговую Accuracy, минимальный Validation Loss и максимальную Accuracy среди всех моделей и эпох.

Таким образом, объединение текстовых и аудиопризнаков через **Multi-Head Self-Attention** дало небольшое преимущество по Accuracy по сравнению с использованием каждой модальности отдельно.

При этом разница между моделями невелика, поэтому результаты относятся именно к текущей архитектуре, гиперпараметрам и пяти эпохам обучения и не доказывают безусловное превосходство мультимодального подхода.
