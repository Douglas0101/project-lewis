# Guia de Contribuição — Project-Lewis

Este documento define as regras e o fluxo de trabalho para contribuir com o **Project-Lewis**, sistema de classificação de arritmias ECG em edge (STM32F4).

Leia também o [`AGENTS.md`](./AGENTS.md) e o [`README.md`](./README.md) antes de começar.

---

## 1. Configuração do ambiente

Use os comandos abaixo para verificar e preparar o ambiente:

```bash
make doctor   # Verifica dependências, versões e permissões
make setup    # Instala dependências, hooks e configurações locais
make help     # Lista todos os comandos disponíveis no Makefile
```

> **Requisitos mínimos:** Python 3.12.x, TensorFlow 2.21, uv (Astral), Docker, arm-none-eabi-gcc 13.3.rel1, Renode 1.15.3.

---

## 2. Fluxo de branches

O repositório segue o modelo **Git Flow simplificado**:

| Branch | Propósito |
|--------|-----------|
| `main` | Código pronto para produção. Apenas merges via PR aprovado. |
| `develop` | Branch de integração (opcional). Usada quando houver desenvolvimento paralelo intenso. |
| `feature/<nome>` | Nova funcionalidade, melhoria ou experimento. |
| `fix/<nome>` | Correção de bug. |
| `docs/<nome>` | Alterações em documentação, guias ou comentários. |

Exemplos de nomes:

```text
feature/quantizacao-int8
fix/overflow-filtro-bandpass
docs/atualiza-quality-gates
```

---

## 3. Convenção de commits

Utilizamos [Conventional Commits](https://www.conventionalcommits.org/) com mensagens em **inglês técnico** e, quando necessário, descrições em **português** para contexto do domínio.

### Tipos permitidos

| Tipo | Uso |
|------|-----|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Alteração em documentação |
| `style` | Formatação, ponto e vírgula, etc. (sem mudança de lógica) |
| `refactor` | Refatoração de código |
| `perf` | Melhoria de performance |
| `test` | Adição ou correção de testes |
| `chore` | Tarefas de build, configuração, dependências |
| `ci` | Alterações em CI/CD |

### Exemplos (inglês técnico)

```bash
feat: add 500 Hz resample pipeline
fix: fix memory leak in TFLM arena
docs: update QG8 thresholds in AGENTS.md
test: add int8 bit-exactness tests vs Python
refactor: simplify AMPT feature extraction
```

Para mudanças maiores, inclua um corpo explicativo:

```bash
feat(qg12): validate RAM arena limits in firmware

- Set tensor_arena default to 48 KB.
- Add limit assertion in TFLM init.
- Update test_arena_limits.py with configurable threshold.
```

---

## 4. Checklist antes de abrir Pull Request

Antes de solicitar revisão, execute localmente:

```bash
make lint   # black, isort, flake8, mypy, bandit
make test   # pytest unitário e de integração
```

Certifique-se de que:

- [ ] `make lint` passa sem erros.
- [ ] `make test` passa sem falhas.
- [ ] A cobertura de testes não regrediu.
- [ ] O código segue o estilo do projeto.
- [ ] Não há dados pessoais (PII) em logs ou arquivos de saída (LGPD).
- [ ] A documentação relevante foi atualizada (`AGENTS.md`, `docs/`, `README.md`).

---

## 5. Regras de code review

Todo código deve passar por revisão antes do merge:

1. **Pelo menos uma aprovação** de um mantenedor ou revisores designados.
2. **CI verde:** todos os checks automatizados devem passar.
3. **Compatibilidade:** o código deve ser compatível com **Python 3.12** e **TensorFlow 2.21**.
4. **Quality gates:** mudanças em camadas críticas devem respeitar os QGs definidos em [`AGENTS.md`](./AGENTS.md).
5. **Revisão humana obrigatória** para código de firmware, segurança ou tratamento de dados sensíveis.

---

## 6. Ferramentas de qualidade

As ferramentas abaixo são executadas automaticamente no `make lint` e nos hooks de `pre-commit`:

| Ferramenta | Função |
|------------|--------|
| `black` | Formatação automática de código Python |
| `isort` | Ordenação de imports |
| `flake8` | Lint de estilo e complexidade |
| `mypy` | Verificação estática de tipos |
| `bandit` | Análise de segurança |
| `pre-commit` | Execução de hooks antes de cada commit |

Para instalar os hooks localmente:

```bash
pre-commit install
```

Para rodar as ferramentas manualmente:

```bash
pre-commit run --all-files
```

---

## 7. Dúvidas?

Consulte os documentos da camada correspondente em `docs/Camada-XX-*.md` ou abra uma issue para discutir alterações maiores antes de implementar.
