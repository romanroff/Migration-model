import pickle
from pathlib import Path

import pandas as pd


def _load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _normalize_full_state_datasets(full_state_datasets_raw, state_order: list[str]) -> dict:
    if isinstance(full_state_datasets_raw, dict):
        return full_state_datasets_raw
    if isinstance(full_state_datasets_raw, list):
        if len(full_state_datasets_raw) != len(state_order):
            raise ValueError("Не удалось сопоставить полный набор данных со списком штатов.")
        return dict(zip(state_order, full_state_datasets_raw))
    raise TypeError("Неподдерживаемый формат полного набора данных.")


def _subtract_test_rows(full_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = full_df.merge(test_df, on=list(full_df.columns), how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)


def build_full_train_datasets(full_state_datasets: dict, test_state_datasets: dict) -> dict:
    """Собирает полный train без утечки тестовых строк."""
    return {
        state: _subtract_test_rows(full_df, test_state_datasets[state])
        for state, full_df in full_state_datasets.items()
    }


def load_datasets(base_path: str = "data", train_variant: str = "full") -> tuple:
    """Загрузка train/test датасетов.

    Параметр train_variant:
    - ``full``: полный train без тестовых строк (рекомендуется)
    - ``sample``: старый уменьшенный train по 1000 строк на штат
    """
    data_path = Path(base_path)

    sample_train = _load_pickle(data_path / "fold1_sample_dataframes_2022_11_02-02_54_03.pkl")
    test = _load_pickle(data_path / "fold2_dataframes_2022_11_02-02_54_03.pkl")
    full_state_datasets_raw = _load_pickle(data_path / "list_states_dataframes_2022_11_02-02_53_58.pkl")

    full_state_datasets = _normalize_full_state_datasets(
        full_state_datasets_raw,
        list(sample_train.keys()),
    )

    if train_variant == "sample":
        train_state_datasets = sample_train
    elif train_variant == "full":
        train_state_datasets = build_full_train_datasets(full_state_datasets, test)
    else:
        raise ValueError("train_variant должен быть 'full' или 'sample'.")

    return train_state_datasets, test, full_state_datasets
