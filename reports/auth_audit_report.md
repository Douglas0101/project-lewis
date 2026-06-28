# Relatório de Auditoria — Autenticação & Segurança

**Projeto:** Project-Lewis  
**Documento de referência:** `docs/SDD_Project-Lewis_v3.md`  
**Data:** 2026-06-20  
**Executor:** Agent SDD-Strict (revisão humana obrigatória — Regra 15)  
**Escopo:** Pente fino de autenticação, secrets, credenciais hardcoded e LGPD/PII.

---

## 1. Resumo Executivo

O Project-Lewis é um pipeline de ML embarcado para classificação de arritmias ECG em STM32F4. **Não possui camada de autenticação** implementada (sem login, tokens, JWT, OAuth, endpoints protegidos, etc.).

A auditoria concluiu que:

| Item | Status | Evidência |
|------|--------|-----------|
| Camada de autenticação no código | ❌ Não existe | `src/`, `firmware/src/`, `scripts/`, `tests/` sem referências a auth |
| Dependências de segurança (bcrypt/argon2/jwt) | ❌ Não listadas | `pyproject.toml` sem `passlib`, `bcrypt`, `argon2`, `pyjwt`, etc. |
| Credenciais hardcoded | ✅ Ausentes | Grep por padrões de senha/token/key não retornou ocorrências |
| Secrets no CI | ✅ Parametrizados | `${{ secrets.NAME }}` em `.github/workflows/*.yml` |
| PII em logs/headers Python | ✅ Não identificado | Nenhum log de nome, CPF, e-mail, paciente, SSN |
| PII em headers C do firmware | ✅ Não identificado | `firmware/src/` sem referências a patient/name/email/etc. |
| `.gitignore` para secrets | ✅ Presente | `.env` listado em `.gitignore` |
| SDD consistente com arquitetura | ✅ Ajustado | Regra 13 é condicional; QG14/QG15 reservados |

---

## 2. Metodologia

1. **Mapeamento no SDD:** extração de todas as ocorrências de `auth|login|password|senha|token|jwt|oauth|argon2|bcrypt|secret|credential|security|LGPD|PII`.
2. **Varredura de código:** `grep -RinE` em `src/`, `tests/`, `scripts/`, `firmware/src/`, `firmware/scripts/`.
3. **Verificação de dependências:** inspeção de `pyproject.toml`.
4. **Verificação de CI:** inspeção de `.github/workflows/*.yml`.
5. **Verificação de credenciais hardcoded:** padrão `(password|passwd|secret|key|token)\s*[:=]\s*["'][^"']+["']`.
6. **Verificação de PII em logs:** padrão de logging + termos de identificação pessoal.
7. **Verificação de PII no firmware:** busca por `patient|name|cpf|email|ssn|pii` em `firmware/src/`.

---

## 3. Achados Detalhados

### 3.1 SDD v3 — Menções a autenticação

O documento trata autenticação de forma **condicional** e consistente:

- **Regra 13:** "Senhas hasheadas com Argon2id/bcrypt — se houver camada de auth"
- **Regra 14:** "LGPD: nenhum PII em logs — anonimização de dados pessoais"
- **Regra 15:** "Revisão humana obrigatória — para código crítico (security, firmware, LGPD)"
- **QG14:** Reservado — segurança/LGPD no firmware
- **QG15:** Reservado — OTA/update seguro
- **Template Security Review (C05/C08):** inclui verificação de PII, inputs, LGPD, headers de segurança e checksums

**Ação realizada:** adicionadas notas clarificadoras no SDD, AGENTS.md e `.kimi/sdd-context.md` indicando que a Regra 13 é condicional e que QG14/QG15 estão reservados para implementação futura.

### 3.2 Código — Ausência de camada de auth

A única ocorrência de termos relacionados foi a palavra **"authoritative"** em comentários sobre a lista de IDs do dataset SVDB (`src/data/download_mitbih.py`, `tests/test_download.py`), sem relação com autenticação.

### 3.3 CI/CD — Secrets

Todos os secrets são consumidos via `${{ secrets.NAME }}`:

- `DVC_REMOTE_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Não há valores hardcoded.

### 3.4 LGPD/PII

Não foram encontradas referências a dados pessoais identificáveis em logs ou headers. Nomes de variáveis como `record_path.name`, `dataset_name`, `zip_cache.name` referem-se a nomes de arquivos/datasets, não a pessoas.

---

## 4. Riscos Identificados

| Risco | Severidade | Mitigação atual | Ação recomendada |
|-------|------------|-----------------|------------------|
| Futura introdução de camada de auth sem seguir Regra 13 | Média | Regra documentada | Gate de revisão humana (Regra 15) |
| QG14/QG15 permanecerem "reservados" sem planejamento | Baixa | Documentados como futuros | Incluir no roadmap quando houver OTA/interface web |
| `.env` não ignorado | N/A | `.env` está em `.gitignore` | Manter `.gitignore` atualizado |
| Vazamento de secrets por dependência | Baixa | Nenhuma dependência de auth | Revisar `pyproject.toml` antes de adicionar libs de auth |

---

## 5. Recomendações

1. **Manter a ausência de auth** como decisão arquitetural explícita enquanto o projeto for um pipeline edge fechado.
2. **Se auth for introduzida futuramente:**
   - Usar **Argon2id** (preferencial) ou **bcrypt** com salt aleatório por senha.
   - Nunca armazenar senhas em texto plano ou com hash fraco (MD5/SHA1).
   - Parametrizar secrets via variáveis de ambiente/GitHub Secrets.
   - Passar por revisão humana obrigatória (Regra 15).
3. **Ativar QG14 e QG15** somente quando houver firmware com LGPD/segurança ativa ou mecanismo OTA.
4. **Manter o template Security Review** atualizado; ele agora inclui verificação de credenciais hardcoded e secrets parametrizados no CI.

---

## 6. Ações Realizadas

- [x] Mapeadas todas as menções a autenticação/LGPD no `docs/SDD_Project-Lewis_v3.md`.
- [x] Verificado código-fonte: sem camada de autenticação, sem credenciais hardcoded.
- [x] Verificado CI/CD: secrets parametrizados corretamente.
- [x] Verificado LGPD/PII: sem exposição identificável em logs e headers C.
- [x] Ajustado `docs/SDD_Project-Lewis_v3.md` com notas clarificadoras (Regra 13 condicional, QG14/QG15 reservados, template Security Review enriquecido).
- [x] Sincronizado `AGENTS.md` e `.kimi/sdd-context.md` com a mesma nota de segurança.
- [x] Emitido este relatório em `reports/auth_audit_report.md`.

---

## 7. Conclusão

A arquitetura atual do Project-Lewis é **compatível** com as diretrizes de segurança do SDD v3. A autenticação é tratada como requisito condicional, e não há exposição de credenciais ou PII. Os ajustes documentais realizados deixam explícito que a Regra 13 só se aplica caso uma camada de auth seja introduzida, e que QG14/QG15 estão reservados para futuras evoluções do sistema.
