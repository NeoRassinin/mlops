# Классификация тональности текста на ClearML

Учебный проект по MLOps. Простую модель определения тональности (позитив/негатив)
я провёл через весь жизненный цикл в ClearML: от загрузки данных до работающего
веб-интерфейса.

Модель намеренно простая — `TF-IDF + логистическая регрессия`. Цель проекта не в
точности модели, а в том, чтобы выстроить всю инфраструктуру MLOps:

**данные → обучение на агенте → реестр моделей → сервинг по HTTP → UI**

```
data/prepare_data.py   создаёт датасет (CSV)
src/upload_dataset.py  заливает его в ClearML как версионированный Dataset
src/train.py           обучает модель на агенте и логирует всё в ClearML
src/register_best.py   публикует лучшую модель в Model Registry
serving/               поднимает HTTP-эндпоинт через clearml-serving
ui/app.py              Streamlit-интерфейс, который ходит в эндпоинт по HTTP
```

## Что нужно для запуска

* Docker Desktop.
* ClearML-сервер (у меня поднят локально в `/opt/clearml`, порты 8008/8080/8081).
* Настроенный `~/clearml.conf` (ключи из веб-интерфейса ClearML).
* Python-окружение:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## Этап 0. Инфраструктура

Поднимаю сервер ClearML и агента, который будет выполнять задачи из очереди `students`.

```bash
# сервер
cd /opt/clearml && docker compose up -d

# агент на очереди students (в отдельном терминале, пусть висит)
.venv/bin/clearml-agent daemon --queue students --create-queue --foreground
```

**Как проверить:** в веб-интерфейсе (http://localhost:8080) → *Workers & Queues* —
виден агент на очереди `students`. Всё, что отправлено в эту очередь, считает агент,
а не моя машина напрямую.

---

## Этап 1. Датасет

Готовлю данные и заливаю их в ClearML как версионированный датасет.

```bash
.venv/bin/python data/prepare_data.py                  # -> data/sentiment.csv (2000 строк)
.venv/bin/python src/upload_dataset.py --version 1.0.0 # печатает DATASET_ID
```

**Как проверить:** раздел *Datasets* → `sentiment-reviews` с версией `1.0.0`.
Обучение потом берёт данные именно по `dataset_id`, а не из локальной папки.

---

## Этап 2. Обучение через агента (2 эксперимента)

`train.py` не обучает модель локально — он отправляет задачу в очередь, а дальше всё
делает агент: ставит окружение, скачивает датасет из ClearML, обучает и логирует
гиперпараметры, метрики (accuracy, f1), confusion matrix (картинкой) и сохраняет модель.

```bash
DSID=<DATASET_ID из этапа 1>

# эксперимент 1
.venv/bin/python src/train.py --dataset-id $DSID --queue students --ngram-max 1 --max-features 2000 --C 1.0
# эксперимент 2 (другие параметры)
.venv/bin/python src/train.py --dataset-id $DSID --queue students --ngram-max 2 --max-features 8000 --C 4.0
```

**Как проверить:** в *Experiments* две завершённые задачи с разными параметрами и
метриками, у каждой есть confusion matrix и артефакт `model`. Запустить новый прогон
можно прямо из интерфейса: правый клик по задаче → *Clone* → поменять параметры → *Enqueue*.

> Маленький нюанс: чтобы агент мог получить код, у репозитория должен быть git-remote
> (подойдёт GitHub). Без него ClearML не передаёт скрипт и агент падает.

---

## Этап 3. Реестр моделей

Выбираю лучшую модель и публикую её в Model Registry.

```bash
.venv/bin/python src/register_best.py   # печатает MODEL_ID
```

**Как проверить:** раздел *Models* → `sentiment-clf` со статусом **Published**, с версией,
тегами (`best`, `production`) и метриками. Published-модель — это запись реестра, а не
просто файл-артефакт задачи.

---

## Этап 4. Эндпоинт (clearml-serving)

Поднимаю сервинг и деплою модель **из реестра** — никаких локальных `.pkl`.

```bash
cd serving
cp .env.example .env            # вписать свои ключи ClearML
.venv/bin/clearml-serving create --name "sentiment-serving"   # id -> в .env
docker compose --env-file .env up -d clearml-serving-inference

.venv/bin/clearml-serving --id <SERVICE_ID> model add \
    --engine custom --endpoint "sentiment" \
    --model-id <MODEL_ID> --preprocess "preprocess.py" \
    --name "sentiment-clf" --project "sentiment-mlops"
```

**Как проверить:**

```bash
curl -X POST http://localhost:9090/serve/sentiment -H "Content-Type: application/json" \
  -d '{"text": "amazing wonderful delightful"}'     # -> positive
curl -X POST http://localhost:9090/serve/sentiment -H "Content-Type: application/json" \
  -d '{"text": "terrible awful useless waste"}'      # -> negative
```

> Датасет синтетический с узким словарём, поэтому модель уверенно реагирует на явные
> сентимент-слова (amazing, terrible, great…), а на произвольном тексте даёт ~50/50.
> Для лабы это нормально — важен сам пайплайн.

---

## Этап 5. Интерфейс (Streamlit)

```bash
SERVING_URL=http://localhost:9090/serve/sentiment .venv/bin/streamlit run ui/app.py
```

Поле для текста, кнопка **Predict**, показывает ответ модели, уверенность и время
ответа (latency). Если эндпоинт недоступен — показывает понятную ошибку. Важное: UI
ходит в модель только по HTTP и сам её не загружает.

---

## Результаты этого прогона

| | значение |
|---|---|
| Датасет | `sentiment-reviews` v1.0.0 |
| Эксперимент 1 | ngram 1, feat 2000, C 1.0 → acc 0.8875 / f1 0.8874 |
| Эксперимент 2 | ngram 2, feat 8000, C 4.0 → acc 0.89 / f1 0.89 (лучший) |
| Модель в реестре | `sentiment-clf`, Published, теги `best/production` |
| Эндпоинт | `POST http://localhost:9090/serve/sentiment` |
| UI | http://localhost:8501 |

---

## Скриншоты

Сложил доказательства по этапам в папку `screenshots/`.

**Этап 0 — агент и очередь**
![Агент в UI](screenshots/00-agent-worker.png)
![Очередь students](screenshots/00-queue.png)

**Этап 1 — датасет**
![Dataset v1.0.0](screenshots/01-dataset.png)

**Этап 2 — два эксперимента**
![Список экспериментов](screenshots/02-experiments-list.png)
![Параметры и воркер](screenshots/02-experiment-info.png)
![Метрики](screenshots/02-metrics.png)
![Confusion matrix](screenshots/02-confusion-matrix.png)
![Артефакт модели](screenshots/02-artifact.png)

**Этап 3 — реестр**
![Опубликованная модель](screenshots/03-model-registry.png)

**Этап 4 — эндпоинт**
![Запросы curl](screenshots/04-endpoint-curl.png)
![Конфиг эндпоинта](screenshots/04-serving-config.png)

**Этап 5 — интерфейс**
![Предсказание](screenshots/05-ui-prediction.png)
![Ошибка при недоступном эндпоинте](screenshots/05-ui-error.png)

---

## Остановить всё

```bash
cd serving && docker compose down
cd /opt/clearml && docker compose down
# агента — Ctrl+C в его терминале
```
