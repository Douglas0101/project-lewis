"""MetricTracker — Callback Keras para persistir métricas por época no SQLite.

Envia loss, métricas de validação e learning rate para o banco de tracking,
permitindo acompanhamento detalhado do treinamento sem depender apenas de
arquivos de log.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

import tensorflow as tf

LOGGER = logging.getLogger("lewis.callbacks.metric_tracker")


class MetricTracker(tf.keras.callbacks.Callback):
    """Persiste métricas do ``model.fit`` no banco de tracking por época.

    Parameters
    ----------
    run_id : int
        ID da run no banco de tracking.
    metrics : list[str], optional
        Chaves do ``logs`` a persistir. Se None, persiste todas as chaves
        numéricas conhecidas (loss, val_loss, accuracy, val_accuracy,
        learning_rate, val_F1_macro etc.).
    namespace : str
        Namespace usado na tabela ``metric``.
    """

    def __init__(
        self,
        run_id: int,
        metrics: Optional[List[str]] = None,
        namespace: str = "history",
        session_factory: Optional[Any] = None,
    ):
        super().__init__()
        self.run_id = run_id
        self.metrics = metrics
        self.namespace = namespace
        self._session_factory = session_factory
        self._session: Optional[Any] = None
        self._repo: Optional[Any] = None

    def _ensure_repo(self) -> bool:
        """Inicializa sessão e repositório de métricas."""
        if self._repo is not None:
            return True
        try:
            from src.tracking.db import get_session
            from src.tracking.repositories import MetricRepository
            from src.tracking.schemas import MetricCreate

            self._session = self._session_factory() if self._session_factory else get_session()
            assert self._session is not None
            self._repo = MetricRepository(self._session)
            self._metric_create_cls = MetricCreate
            return True
        except Exception:
            LOGGER.exception("MetricTracker: falha ao inicializar repositório")
            return False

    def _close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # nosec B110
                pass
            self._session = None
            self._repo = None

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Persiste métricas da época no banco."""
        if not logs:
            return

        if not self._ensure_repo():
            return

        try:
            items: List[Any] = []
            for name, value in logs.items():
                if self.metrics is not None and name not in self.metrics:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                items.append(
                    self._metric_create_cls(
                        run_id=self.run_id,
                        namespace=self.namespace,
                        name=name,
                        value=float(value),
                        step=int(epoch) + 1,
                    )
                )

            if items:
                assert self._repo is not None
                self._repo.create_many(items)
                assert self._session is not None
                self._session.commit()
        except Exception:
            LOGGER.exception("MetricTracker: falha ao registrar métricas da época %d", epoch + 1)
            try:
                if self._session is not None:
                    self._session.rollback()
            except Exception:  # nosec B110
                pass

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        self._close()
