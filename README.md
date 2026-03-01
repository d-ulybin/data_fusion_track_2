# Data Fusion 2026 — Track 2 "Киберполка"

1 место. OOF 0.8483 / Public 0.8536.

Multi-label classification: предсказание вероятностей открытия 41 банковского продукта для 250K клиентов. Метрика — Macro ROC-AUC.

## Feature Engineering

Все данные анонимизированы (750K train + 250K test, ~2400 фичей).

- **Null Pattern PCA** (20 компонент) — паттерны пропусков в extra features предиктивны (sparse SVD на бинарной null-матрице 2241 колонок)
- **Ratio features** (9 пар) — отношения числовых фичей дают сигнал до -0.556 корреляции с таргетами, ортогональны diff features
- **Individual null indicators** — отбор по корреляции (|corr| > 0.05) с таргетами, ~124 индикатора из 131 числовых
- **Дедупликация** — MD5 хеш колонок extra features, удаление 6 дублей категорий
- **Frequency encoding** — на train+test для консистентности
- **Row stats** — std, skew по строкам (main + extra отдельно)

## Модели

### NN (OOF 0.8333)
- **DAE pretraining** — Denoising Autoencoder (1024→512→256 bottleneck) на 1M строк (train+test), swap noise 15%. 256d embeddings как дополнительные фичи
- **PLR encoding** — Piecewise Linear encoding (24 quantile bins) для каждой числовой фичи
- **TabM** (k=16) — Batch Ensemble с rank-1 адаптерами, 3 residual блока (384d)
- **ASL** — Asymmetric Loss (gamma_neg=4, clip=0.05), лучше BCE для imbalanced multi-label
- **Manifold Mixup** (alpha=0.2) — на случайном слое
- **SWA top-5** — среднее 5 лучших чекпоинтов

### LightGBM L7 (OOF 0.8348)
- Optuna HPO (30 trials), лучшие параметры вшиты в скрипт: lr=0.050, leaves=34, colsample=20%
- MultilabelStratifiedKFold, 41 бинарных классификатора × 5 фолдов

### PyBoost (OOF 0.8362)
- SketchBoost (joint multi-output), Optuna HPO (25 trials), лучшие параметры вшиты
- Требует NVIDIA GPU (cupy + py-boost)

### LGBM L8 — cross-target meta (OOF 0.8433)
- Base features + OOF predictions от NN + LGBM + PyBoost (120 мета-фичей)
- Own-target маскирование: для target_i свои 3 OOF колонки → NaN (без утечки)

## Блендинг и стекинг

- **Rank blend** — per-target weight optimization (grid search на OOF)
- **Stacking** — Ridge + LGBM meta-learner на 369 мета-фичах (base + diffs + products)
- **Combo** — 15% rank(meta) + 85% rank(blend) → OOF 0.8483

## Запуск

```bash
pip install -r requirements.txt
# Положить данные в Data/

python 01_feature_engineering.py   # FE → features/
python 02_train_nn.py              # NN → checkpoints_nn/
python 03_train_lgbm.py            # LGBM → checkpoints_lgbm/
python 04_train_pyboost.py         # PyBoost → checkpoints_pyboost/ (CUDA only)
python 05_train_lgbm_meta.py       # LGBM meta → checkpoints_lgbm_meta/
python 06_blend.py                 # Blend → submissions/blend.parquet
python 07_stacking.py              # Stacking → submissions/stacking.parquet
```

Шаги 02, 03, 04 можно запускать параллельно (после 01). Шаг 05 — после 02+03+04.

## Credits

- @Gofat — идея cross-target meta-features
- @ivanich_spb — идея PyBoost/SketchBoost
