"""Thứ tự nhãn thống nhất cho dữ liệu, checkpoint và giao diện."""

LABELS: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

LABELS_VI: tuple[str, ...] = (
    "Bìa carton",
    "Thủy tinh",
    "Kim loại",
    "Giấy",
    "Nhựa",
    "Rác khác",
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

