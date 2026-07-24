from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


class BaseRecord:
    record_name: Optional[str]
    fs: Optional[float]
    n_sig: int
    adc_gain: Union[List[float], np.ndarray, None]
    adc_zero: Union[List[int], np.ndarray, None]
    baseline: Union[List[int], np.ndarray, None]
    adc_res: Union[List[int], np.ndarray, None]
    init_value: Union[List[int], np.ndarray, None]
    checksum: Union[List[int], np.ndarray, None]
    units: Optional[List[str]]
    sig_name: Optional[List[str]]
    p_signal: Optional[np.ndarray]
    d_signal: Optional[np.ndarray]
    sig_len: int
    comments: Optional[List[str]]
    input_fields: Any


class Record(BaseRecord):
    ...


class MultiRecord(BaseRecord):
    ...


def rdrecord(
    record_name: Union[str, Path],
    *,
    channels: Optional[Sequence[int]] = ...,
    physical: bool = ...,
    **kwargs: Any,
) -> Union[Record, MultiRecord]:
    ...


def rdann(record_name: Union[str, Path], extension: str, **kwargs: Any) -> Any:
    ...


def rdheader(record_name: Union[str, Path], **kwargs: Any) -> Union[Record, MultiRecord]:
    ...


def dl_database(
    db_dir: str,
    records: Optional[Union[str, Sequence[str]]] = ...,
    dl_dir: Optional[str] = ...,
    **kwargs: Any,
) -> None:
    ...


def dl_files(
    db_dir: str,
    dl_dir: str,
    files: Sequence[str],
    **kwargs: Any,
) -> None:
    ...
