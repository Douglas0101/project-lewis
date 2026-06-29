"""Helper para obter hash curto do commit git atual."""

from __future__ import annotations

from typing import Optional


def git_commit_short() -> Optional[str]:
    """Retorna hash curto do commit atual, se disponível."""
    import subprocess  # nosec B404

    try:
        result = subprocess.run(  # nosec B603,B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:  # nosec B110
        return None
