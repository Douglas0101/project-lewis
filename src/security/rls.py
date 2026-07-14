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
        table_alias: str | None = None,
        owner_column: str = "owner_id",
    ) -> Tuple[str, Union[Tuple[str, ...], Dict[str, Any]]]:
        if not user_id or not str(user_id).strip():
            return sql, ()

        lowered_roles = {str(role).strip().lower() for role in roles}
        if "admin" in lowered_roles:
            return sql, ()

        normalized = sql.strip()
        has_named = self._has_named_bind(normalized)
        column_ref = f"{table_alias}.{owner_column}" if table_alias else owner_column
        placeholder = f"{column_ref} = :owner_id" if has_named else f"{column_ref} = ?"

        params: Union[Tuple[str, ...], Dict[str, Any]] = (
            {"owner_id": user_id} if has_named else (user_id,)
        )

        where_match = self._top_level_keyword_index(normalized, "WHERE")
        if where_match is not None:
            where_idx, keyword = where_match
            before = normalized[:where_idx]
            after = normalized[where_idx + len(keyword) :]
            return f"{before}{keyword} {placeholder} AND{after}", params

        for keyword in ("GROUP BY", "ORDER BY", "HAVING", "LIMIT"):
            match = self._top_level_keyword_index(normalized, keyword)
            if match is not None:
                idx, matched_text = match
                before = normalized[:idx]
                after = normalized[idx:]
                return f"{before} WHERE {placeholder} {after}", params

        return f"{normalized} WHERE {placeholder}", params

    @staticmethod
    def _has_named_bind(sql: str) -> bool:
        """Detecta binds nomeados fora de literais de string."""
        stripped = re.sub(r"'[^']*'", "''", sql)
        return ":" in stripped and re.search(r":\w+", stripped) is not None

    @staticmethod
    def _top_level_keyword_index(sql: str, keyword: str) -> Tuple[int, str] | None:
        """Retorna o índice e o texto exato de uma palavra-chave SQL no nível top-level.

        Ignora ocorrências dentro de parênteses e literais de string com aspas
        simples. Keyword pode ser uma ou duas palavras (ex.: ``WHERE`` ou
        ``GROUP BY``).
        """
        # Mascara literais de string para não confundirem palavras-chave,
        # preservando o comprimento para que os índices se válidos na SQL
        # original.
        stripped = re.sub(r"'[^']*'", lambda m: " " * len(m.group()), sql)
        masked = []
        depth = 0
        for ch in stripped:
            if ch == "(":
                depth += 1
                masked.append(" ")
            elif ch == ")":
                depth = max(0, depth - 1)
                masked.append(" ")
            elif depth > 0:
                masked.append(" ")
            else:
                masked.append(ch)
        masked_sql = "".join(masked)

        parts = keyword.split()
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in parts) + r"\b"
        match = re.search(pattern, masked_sql, re.IGNORECASE)
        if match is None:
            return None
        start = match.start()
        # Recupera o texto exto da palavra-chave na SQL original, preservando
        # a capitalização real.
        return start, sql[start : start + len(match.group())]
