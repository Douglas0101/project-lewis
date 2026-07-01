"""Row-Level Security por owner_id."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence, Tuple, Union


class RowLevelSecurity:
    """Injeta filtro owner_id em queries SQL parametrizadas."""

    def apply_filter(
        self,
        sql: str,
        user_id: str,
        roles: Sequence[str] = (),
    ) -> Tuple[str, Union[Tuple[str, ...], Dict[str, Any]]]:
        if "admin" in roles:
            return sql, ()

        normalized = sql.strip()
        has_named = ":" in normalized and re.search(r":\w+", normalized) is not None
        placeholder = "owner_id = :owner_id" if has_named else "owner_id = ?"

        params: Union[Tuple[str, ...], Dict[str, Any]] = (
            {"owner_id": user_id} if has_named else (user_id,)
        )

        where_match = re.search(r"\bWHERE\b", normalized, re.IGNORECASE)
        if where_match:
            before = normalized[: where_match.end()]
            after = normalized[where_match.end() :]
            new_sql = f"{before} {placeholder} AND{after}"
            return new_sql, params

        order_match = re.search(r"\bORDER\s+BY\b", normalized, re.IGNORECASE)
        if order_match:
            before = normalized[: order_match.start()]
            after = normalized[order_match.start() :]
            return f"{before} WHERE {placeholder} {after}", params

        return f"{normalized} WHERE {placeholder}", params
