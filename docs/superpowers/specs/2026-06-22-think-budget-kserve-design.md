# Дизайн: think-budget в билд kserve (vLLM 0.23)

**Дата:** 2026-06-22
**Ветка реализации:** `v0.18.0-inferencevalve-vllm0.23-thinkbudget` (ответвить от `v0.18.0-inferencevalce-vllm0.23-nightly`)

## Цель

vLLM logits-процессор `think-budget` (контроль бюджета токенов мышления в reasoning-моделях:
Qwen3/Qwen3.5/DeepSeek-R1) должен быть доступен внутри vllm023-образа huggingfaceserver.
Процессор включается **вручную** при запуске пода; код kserve не меняется.

`think-budget` — самодостаточный пакет из одного модуля `think_budget.py` (версия 0.4.0,
зависимость — только `torch`). Подключается к vLLM флагом
`--logits-processors think_budget:ThinkBudgetProcessor` и настраивается через env-переменные
(`THINK_BUDGET`, `THINK_ENSURE_END_BEFORE_EOS`, `THINK_EOS_PROB_THRESHOLD`,
`THINK_BAN_EOS_AFTER_THINK_END_TOKENS` и др.). Требует vLLM >= 0.11 (V1 engine).

## Контекст и ограничения

Два факта текущего билда определяют дизайн:

1. **Контекст docker-сборки — папка `python/`** (`Makefile:595`: `cd python && ... -f ... .`).
   Пакет сейчас лежит в **корне репозитория**
   (`think-budget-feature-ensure-think-end-before-eos/`, отслеживается git, 8 файлов),
   то есть **вне** контекста сборки — `COPY` до него не дотягивается.
2. **`huggingfaceserver` ставится с `--no-deps`** (`huggingface_server_vllm023.Dockerfile:58`).
   Это намеренно: иначе разрешение зависимостей из его `pyproject.toml` даунгрейднет
   `torch`/`vllm`/`transformers` из base-образа. Следствие: **добавление зависимости в
   `pyproject.toml` само по себе НЕ установит пакет в образ** — нужен явный шаг установки.

Дополнительно: huggingfaceserver уже прокидывает аргументы в vLLM через `make_arg_parser`
(`python/huggingfaceserver/huggingfaceserver/vllm/utils.py:61`), поэтому
`--logits-processors` поддерживается **без правок кода kserve** — нужно лишь, чтобы модуль
`think_budget` был импортируем в образе.

## Уровень активации

Выбран **уровень 1 — «доступен, включается вручную»**: модуль установлен в образ; чтобы
активировать, при запуске пода передаются `--logits-processors think_budget:ThinkBudgetProcessor`
и env-переменные. Без них образ ведёт себя как раньше — нулевой риск регрессии для существующих
деплоев.

Отклонённые альтернативы: «включён по умолчанию через env» (требует правки кода huggingfaceserver,
связывает kserve с пакетом) и «жёстко включён всегда» (негибко, влияет на нерассуждающие модели).

## Изменения

Все изменения — на отдельной ветке `v0.18.0-inferencevalve-vllm0.23-thinkbudget`.

### 1. Перенос пакета в `python/think-budget/`

```bash
git mv think-budget-feature-ensure-think-end-before-eos python/think-budget
```

Переносится вся директория целиком (`think_budget.py`, `pyproject.toml`, `README.md`,
`test_budget.py`, `test_processor_unit.py`, `serve.sh`, `slurm/`). Содержимое файлов не меняется.
Имя каталога становится чистым `think-budget` (совпадает с `name` пакета); импортируемый модуль
остаётся `think_budget`.

Каталог становится сиблингом `kserve`, `storage`, `huggingfaceserver` — ложится в существующую
структуру `python/`. Файлы `serve.sh`/`slurm/` — standalone-хелперы запуска вне kserve. Шаг
`COPY think-budget think-budget` (раздел 3) копирует каталог целиком, поэтому они попадают в
build-слой, но остаются инертными: `uv pip install .` устанавливает только модуль `think_budget`
(`py-modules = ["think_budget"]` в `pyproject.toml` пакета), а хелперы просто лежат в рабочем
каталоге и на рантайм не влияют. В репозитории они остаются как самодокументация пакета.

### 2. Локальная зависимость в `python/huggingfaceserver/pyproject.toml`

Добавить в список `dependencies` по образцу существующих локальных путей:

```toml
dependencies = [
    "kserve[llm] @ file:///${PROJECT_ROOT}/../kserve",
    "kserve-storage @ file:///${PROJECT_ROOT}/../storage",
    "think-budget @ file:///${PROJECT_ROOT}/../think-budget",   # новое
    "transformers>=4.53.2",
    ...
]
```

`PROJECT_ROOT` = `python/huggingfaceserver`, поэтому `../think-budget` указывает на новый каталог.
Эта строка обслуживает **локальную разработку** (`uv sync` / `pip install .` без `--no-deps`
подтянет пакет) и документирует зависимость. В docker-образе из-за `--no-deps` она не
срабатывает — установку в образ выполняет шаг 3.

### 3. Шаг установки в `python/huggingface_server_vllm023.Dockerfile`

Добавить явный шаг по шаблону блока `storage` (строки 52-54), сразу после него и до
блока `huggingfaceserver`:

```dockerfile
COPY storage storage
RUN --mount=type=cache,target=/root/.cache/uv cd storage \
    && uv pip install --system . --no-cache --no-deps

COPY think-budget think-budget                                   # новое
RUN --mount=type=cache,target=/root/.cache/uv cd think-budget \  # новое
    && uv pip install --system . --no-cache --no-deps            # новое

COPY huggingfaceserver huggingfaceserver
RUN --mount=type=cache,target=/root/.cache/uv cd huggingfaceserver \
    && uv pip install --system . --no-cache --no-deps
```

`--no-deps` — потому что единственная зависимость пакета (`torch`) уже присутствует в vllm-base;
переустанавливать её не нужно. Объём — **только этот Dockerfile**; `huggingface_server_vllm020.Dockerfile`
и стандартный `huggingface_server.Dockerfile` не трогаем (пакет требует vLLM >= 0.11, а текущая
работа — про 0.23).

### Что НЕ меняется

Код kserve/huggingfaceserver, `Makefile`, `kserve-images.env`, CI-конфигурация. Фича не добавляет
сборочной цели и не меняет имя образа.

## Активация в рантайме (справочно)

Образ ведёт себя как раньше, пока процессор не включён явно. Фрагмент `InferenceService`:

```yaml
spec:
  predictor:
    model:
      args:
        - --logits-processors=think_budget:ThinkBudgetProcessor
      env:
        - name: THINK_BUDGET
          value: "1024"
        - name: THINK_ENSURE_END_BEFORE_EOS
          value: "1"
        - name: THINK_EOS_PROB_THRESHOLD
          value: "0.2"
```

Per-request оверрайды (`vllm_xargs: {think_budget: 256}`) работают «из коробки» — их обеспечивает
сам процессор. Полный список env-переменных и примеры клиента — в `python/think-budget/README.md`.

## Имя docker-образа

Рекомендуемый тег: `${KO_DOCKER_REPO}/huggingfaceserver:v0.23.0-thinkbudget`.

vllm023-Dockerfile не подключён к `Makefile`/`kserve-images.env`, поэтому собирается вручную и имя
задаётся флагом `-t`. Дизайн намеренно не добавляет Makefile-цель для этой сборки.

## Тестирование и проверка

1. **Юнит-тесты пакета** (переезжают вместе с ним) — убедиться, что относительные импорты не поехали
   после переноса:
   ```bash
   cd python/think-budget && python -m pytest -q
   ```
2. **Проверка сборки образа** — собрать и проверить импортируемость + резолвинг entrypoint-имени:
   ```bash
   cd python && docker buildx build \
       -f huggingface_server_vllm023.Dockerfile \
       -t ${KO_DOCKER_REPO}/huggingfaceserver:v0.23.0-thinkbudget .
   docker run --rm ${KO_DOCKER_REPO}/huggingfaceserver:v0.23.0-thinkbudget \
       python3 -c "import think_budget; from think_budget import ThinkBudgetProcessor; print('ok')"
   ```
   Ловит обе главные ошибки: пакет не скопировался/не установился, либо имя
   `think_budget:ThinkBudgetProcessor` не резолвится.
3. **Smoke-тест рантайма** (опционально, нужна GPU) — поднять контейнер с reasoning-моделью и
   `--logits-processors think_budget:ThinkBudgetProcessor`, послать запрос, убедиться, что бюджет
   мышления соблюдается (по таблице README: budget=64 → ~86 reasoning-токенов).

CI kserve не трогаем; новый пакет в общий тест-прогон не включаем (он самодостаточный, со своими
тестами).

## Критерии готовности

- [ ] Пакет лежит в `python/think-budget/`, его тесты проходят.
- [ ] `python/huggingfaceserver/pyproject.toml` содержит локальную зависимость `think-budget`.
- [ ] `huggingface_server_vllm023.Dockerfile` содержит шаг COPY+install для `think-budget`.
- [ ] Внутри собранного образа `from think_budget import ThinkBudgetProcessor` работает.
- [ ] Код kserve, Makefile, kserve-images.env, CI — без изменений.
