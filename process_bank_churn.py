"""
process_bank_churn.py

Попередня обробка даних для змагання Bank Customer Churn Prediction.

Логіка модуля: кожна функція робить рівно одну дію. Функції `fit_*`
викликаються ТІЛЬКИ на тренувальних даних, `apply_*` — на будь-яких.
Такий поділ робить data leakage структурно неможливим.

Приклад використання:
    from process_bank_churn import preprocess_data, preprocess_new_data

    data = preprocess_data(raw_df, scaler_numeric=False)   # для дерев
    X_train, train_targets = data['X_train'], data['train_targets']

    X_test = preprocess_new_data(
        test_raw_df,
        input_cols=data['input_cols'],
        scaler=data['scaler'],
        encoder=data['encoder'],
        numeric_cols=data['numeric_cols'],
        categorical_cols=data['categorical_cols'],
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL: str = 'Exited'
ID_COLS: List[str] = ['id', 'CustomerId', 'Surname']


def select_input_columns(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Обрати назви колонок, які підуть на вхід моделі.

    Ідентифікатори (`id`, `CustomerId`) та `Surname` виключаються: перші два
    не несуть інформації про поведінку клієнта, а прізвище дало б сотні
    one-hot колонок і зробило б дерево неінтерпретованим.

    Args:
        df: Сирий датафрейм.
        target_col: Назва цільової колонки.
        drop_cols: Колонки, які треба виключити. За замовчуванням `ID_COLS`.

    Returns:
        Список назв вхідних колонок (до кодування).
    """
    drop_cols = ID_COLS if drop_cols is None else drop_cols
    excluded = set(drop_cols) | {target_col}
    return [col for col in df.columns if col not in excluded]


def split_train_val(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбити сирі дані на тренувальний і валідаційний піднабори.

    Розбиття стратифіковане за цільовою колонкою: класи незбалансовані
    (~20% відтоку), тож без `stratify` частка класу 1 у val могла б
    відрізнятися від train і зробити оцінку метрики нестабільною.

    Args:
        df: Сирий датафрейм.
        target_col: Колонка для стратифікації.
        test_size: Частка валідаційної вибірки.
        random_state: Фіксація випадковості для відтворюваності.

    Returns:
        Кортеж (train_df, val_df).
    """
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )


def separate_inputs_targets(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Розділити датафрейм на вхідні ознаки та цільову змінну.

    Args:
        df: Піднабір даних (train або val).
        input_cols: Назви вхідних колонок.
        target_col: Назва цільової колонки.

    Returns:
        Кортеж (inputs, targets). Ціль приводиться до int.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy().astype(int)
    return inputs, targets


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Визначити числові та категоріальні колонки за їх dtype.

    Args:
        df: Датафрейм вхідних ознак.

    Returns:
        Кортеж (numeric_cols, categorical_cols).
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def fit_scaler(df: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    """
    Навчити StandardScaler на числових колонках ТРЕНУВАЛЬНИХ даних.

    Args:
        df: Тренувальні вхідні дані.
        numeric_cols: Назви числових колонок.

    Returns:
        Навчений StandardScaler.
    """
    return StandardScaler().fit(df[numeric_cols])


def apply_scaler(
    df: pd.DataFrame,
    numeric_cols: List[str],
    scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    """
    Застосувати навчений скейлер до числових колонок.

    Якщо `scaler` дорівнює None (режим `scaler_numeric=False`), дані
    повертаються без змін — деревам масштабування не потрібне.

    Args:
        df: Вхідні дані будь-якого піднабору.
        numeric_cols: Назви числових колонок.
        scaler: Навчений скейлер або None.

    Returns:
        Копія датафрейму з масштабованими числовими колонками.
    """
    df = df.copy()
    if scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def fit_encoder(df: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Навчити OneHotEncoder на категоріальних колонках ТРЕНУВАЛЬНИХ даних.

    `handle_unknown='ignore'` захищає від категорії, якої не було в train:
    замість помилки такий рядок отримає нулі в усіх one-hot колонках.

    Args:
        df: Тренувальні вхідні дані.
        categorical_cols: Назви категоріальних колонок.

    Returns:
        Навчений OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return encoder.fit(df[categorical_cols])


def apply_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """
    Закодувати категоріальні колонки навченим енкодером.

    Оригінальні текстові колонки видаляються — модель приймає лише числа.

    Args:
        df: Вхідні дані будь-якого піднабору.
        categorical_cols: Назви категоріальних колонок.
        encoder: Навчений OneHotEncoder.

    Returns:
        Копія датафрейму без текстових колонок і з доданими one-hot колонками.
    """
    df = df.copy()
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoded_cols,
        index=df.index,
    )
    return pd.concat([df.drop(columns=categorical_cols), encoded], axis=1)


def build_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """
    Зібрати фінальну матрицю ознак: масштабування + кодування.

    Викликається однаково для train, val і test — саме тому порядок
    колонок на виході гарантовано збігається.

    Args:
        df: Вхідні дані будь-якого піднабору.
        numeric_cols: Назви числових колонок.
        categorical_cols: Назви категоріальних колонок.
        scaler: Навчений скейлер або None.
        encoder: Навчений енкодер.

    Returns:
        Датафрейм, готовий подаватися в модель.
    """
    df = apply_scaler(df, numeric_cols, scaler)
    df = apply_encoder(df, categorical_cols, encoder)
    return df


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Повний цикл підготовки сирих тренувальних даних.

    Порядок кроків принциповий: спершу розбиття, потім навчання скейлера
    та енкодера ТІЛЬКИ на train. Інакше статистики валідаційної вибірки
    протекли б у препроцесинг і оцінка якості була б завищеною.

    Args:
        raw_df: Сирі дані з `train.csv`.
        scaler_numeric: Чи масштабувати числові ознаки. Для дерев — False.
        test_size: Частка валідаційної вибірки.
        random_state: Фіксація випадковості.

    Returns:
        Словник з ключами:
            X_train, train_targets, X_val, val_targets,
            input_cols (фінальні колонки X у правильному порядку),
            numeric_cols, categorical_cols, scaler, encoder.
    """
    raw_input_cols = select_input_columns(raw_df)

    train_df, val_df = split_train_val(raw_df, TARGET_COL, test_size, random_state)

    train_inputs, train_targets = separate_inputs_targets(train_df, raw_input_cols)
    val_inputs, val_targets = separate_inputs_targets(val_df, raw_input_cols)

    numeric_cols, categorical_cols = get_column_types(train_inputs)

    scaler = fit_scaler(train_inputs, numeric_cols) if scaler_numeric else None
    encoder = fit_encoder(train_inputs, categorical_cols)

    X_train = build_features(train_inputs, numeric_cols, categorical_cols, scaler, encoder)
    X_val = build_features(val_inputs, numeric_cols, categorical_cols, scaler, encoder)

    input_cols = X_train.columns.tolist()
    X_val = X_val[input_cols]

    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'scaler': scaler,
        'encoder': encoder,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Підготувати нові дані (`test.csv`) вже навченими скейлером та енкодером.

    Жодного `fit` тут немає і бути не може: нові дані не мають впливати на
    параметри препроцесингу. Функція також не чіпає цільову колонку —
    у тестових даних змагання її просто немає.

    Args:
        new_df: Сирі нові дані.
        input_cols: Фінальний перелік колонок X з `preprocess_data`.
        scaler: Навчений скейлер або None.
        encoder: Навчений енкодер.
        numeric_cols: Назви числових колонок.
        categorical_cols: Назви категоріальних колонок.

    Returns:
        Датафрейм із тими самими колонками і в тому самому порядку, що X_train.
    """
    df = new_df[numeric_cols + categorical_cols].copy()
    df = build_features(df, numeric_cols, categorical_cols, scaler, encoder)
    return df.reindex(columns=input_cols, fill_value=0)