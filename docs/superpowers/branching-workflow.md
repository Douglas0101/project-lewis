# Fluxo de Branches do Project-Lewis

Este documento define o fluxo de branches adotado pelo Project-Lewis a partir de 2026-06-21. O objetivo é organizar atualizações, implementações e engenharias alfa/beta antes da integração na branch principal (`master`).

## Branches permanentes

| Branch | Propósito | Quem pode fazer push direto |
|--------|-----------|-----------------------------|
| `master` | Código estável e pronto para release/produção. | Ninguém (apenas via Pull Request). |
| `develop` | Branch fixa de integração para features, atualizações e validações alfa/beta. | Ninguém (apenas via Pull Request). |

## Branches temporárias

| Padrão | Quando usar | Exemplo |
|--------|-------------|---------|
| `feature/<nome-curto>` | Novas funcionalidades ou melhorias. | `feature/slha-hardening` |
| `fix/<nome-curto>` | Correções de bugs. | `fix/discovery-timeout` |
| `docs/<nome-curto>` | Alterações exclusivas em documentação. | `docs/slha-engineering-review` |
| `release/<versao>` | Preparação de release (opcional). | `release/v1.0.0` |
| `hotfix/<nome-curto>` | Correções urgentes aplicáveis diretamente em `master` (raro). | `hotfix/ci-token` |

## Fluxo de trabalho

```text
feature/*  ──┐
fix/*      ──┤
docs/*     ──┼──>  develop  ──(quando estável)──>  master
release/*  ──┘        ↑                              ↑
hotfix/*  ────────────┘ (somente em emergência)      │
                                                     │
                                               (apenas PR)
```

1. **Criar branch temporária** a partir de `develop` (ou de `master` para hotfix).
2. **Desenvolver e testar** localmente na branch temporária.
3. **Abrir Pull Request** para `develop`.
4. **Revisar e aprovar** o PR (revisão humana obrigatória para código crítico, firmware e LGPD).
5. **Fazer merge** em `develop`.
6. **Quando `develop` estiver estável**, abrir PR de `develop` para `master`.

## Alfa e beta

- **Alfa:** trabalho em andamento e integrações recentes na branch `develop`. Pode conter instabilidade controlada.
- **Beta:** versões candidatas à release, geralmente representadas por tags (`vX.Y.Z-beta.N`) ou branches `release/*` criadas a partir de `develop`.

## Convenções

- Use nomes curtos, descritivos e em inglês para branches temporárias.
- Commits devem seguir o padrão semântico (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Nunca faça push direto em `master` ou `develop`.
- Sempre execute `make lint && make test` antes de abrir um PR.

## Proteção de branches (recomendação)

Configurar no repositório remoto:

- `master`: exigir PR, aprovação mínima de 1 revisor, checks de CI passando.
- `develop`: exigir PR e checks de CI passando.

> A configuração de proteção depende de permissões administrativas no GitHub e deve ser feita manualmente por um mantenedor.
