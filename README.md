# Migration Prediction System
Прогнозирование миграционных потоков населения с использованием Random Forest и визуализация результатов.

---
## Установка
1. **Клонируйте репозиторий**:
```bash
git clone https://github.com/romanroff/migration-model.git
```
2. **Создайте и активируйте виртуальное окружение** (рекомендуется):
```bash
python -m venv .venv
```
3. **Установите пакет**:
```bash
pip install .
```
---
## Использование
### Конфигурация
В config.py необходимо настроить территорию для загрузки.

### Обучение модели
```py
python train.py
```

### Валидация модели
```py
python validation.py
```

Результат можно открыть в `migration_map_py.html` файле.