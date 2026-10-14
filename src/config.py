"""Константы проекта. Поля как в таблице транзакций MBD-mini."""
import numpy as np

CAT_FIELDS = [
    "event_type", "event_subtype", "currency",
    "src_type11", "src_type12", "dst_type11", "dst_type12",
    "src_type21", "src_type22", "src_type31", "src_type32",
]
TARGETS = ["target_1", "target_2", "target_3", "target_4"]
BASE_DAY = np.datetime64("2020-01-01")  # день 0, это среда
BASE_DOW = 2  # понедельник = 0

# тест и валидация по фолдам авторов датасета
TEST_FOLD = 0
VALID_FOLD = 1
