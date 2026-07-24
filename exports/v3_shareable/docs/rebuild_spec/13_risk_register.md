# 13 — Registro de Riscos

**Status:** PROPOSTO — vivo; revisão a cada etapa ratificada
**Data:** 2026-07-18
**Entregável:** `risk_register` (16)

---

| ID | Risco | Prob. | Impacto | Detecção | Mitigação | Dono | Status |
|---|---|---|---|---|---|---|---|
| R-01 | Desalinhamento temporal residual após correção (DQ-01) | média | crítico | gates G-T1…G-T8 (02); correlação janela×sinal | reprocessamento v3.1 com novo hash | eng. dados | aberto |
| R-02 | Unidade de feature errada recorrente (DQ-02) | baixa | alto | schema com unidade/relógio + teste dual-clock (02 §2) | regra de CI: feature temporal sem unidade falha o build | eng. dados | aberto |
| R-03 | Suporte insuficiente de FUSION (1.044 beats/45 pacientes, top-5=82%) | alta | alto | contagem por paciente no manifest; IC bootstrap | D3 (AFIB como ritmo alivia F); pisos por classe (11 §4); abstenção | cientista | aberto |
| R-04 | AFIB sem validação externa suficiente mesmo com AFDB | média | alto | LODO + contagem de episódios | declarar limitação; buscar fonte adicional ratificada; não prometer gate de ritmo sem suporte | cientista | aberto |
| R-05 | Atalho de domínio residual (DQ-14) | média | crítico | probe dataset_id (06 §2); métricas condicionais | λ_inv (04); regra R-F1 (03); LODO | cientista | aberto |
| R-06 | Calibração mascarando não-generalização | média | alto | pré-condições 07 §2; ΔSe/ΔSpe no estado C | hard reject HR-13; proibição de calibrador por domínio | cientista | aberto |
| R-07 | IC amplo em doença rara lido como garantia | média | alto | bootstrap 10k publicado por classe | RR-02 (review obrigatório); abstenção; denominadores explícitos | bioestat. | aberto |
| R-08 | MIT-BIH ≠ população-alvo; skew de aquisição anos 80/90 | alta | alto | relatório de representatividade; validação externa (15 do relatório forense) | declarar escopo; validação prospectiva antes de qualquer uso clínico | clínico | aberto |
| R-09 | Orçamento edge estourado (FlatBuffer/arena/latência) | média | médio | QG6/QG7/QG9/QG12 por build | restrições duras em 04 §5; descarte de famílias que não cabem | firmware | aberto |
| R-10 | Backend Sigstore não ratificado; attestation fica shadow | alta | médio | estado `shadow=true` nos relatórios | fail-closed; quorum manual até ratificação | segurança | aberto |
| R-11 | LGPD: vazamento de PII em logs/manifests (IDs de paciente) | baixa | alto | varredura de logs; revisão de manifests | pseudonimização de patient_id; regra #14 do AGENTS.md | segurança | aberto |
| R-12 | Regeneração de dados sem invalidar baselines antigos (confusão de versões) | média | alto | `preprocessing_version` + hashes novos; ponteiro único de bundle | política 08/11; artefatos legados marcados inválidos | MLOps | aberto |
| R-13 | Ratificação implícita por omissão (decisões D1–D7 "passarem" sem registro) | média | alto | 12 exige registro com hash e aprovador | sem registro = PENDING; pipeline bloqueia | governança | aberto |
| R-14 | Escopo além de pesquisa confundido com uso clínico | média | crítico | declaração de escopo em todos os relatórios | estados finais sem autorização de deployment; revisão clínica futura separada | governança | aberto |
| R-15 | Run fora de protocolo (ex.: 2-fold em vez de 5-fold, como em jul/04) | média | alto | verificação config_hash × manifest (05 §3) | HR-08/HR-14; execução somente via CLI canônico com preflight | MLOps | aberto |
| R-16 | Duplicatas/conflitos residuais de label (DQ-04) | baixa | médio | dedup audit na regeneração v3 | regra de deduplicação ratificada; `reviewStatus` por instância | eng. dados | aberto |

## Notas

- Probabilidade/impacto qualitativos nesta etapa; quantificação exige dados v3.
- Nenhum risco é fechado por documento: fechamento exige evidência de gate verde registrada no
  bundle da geração correspondente.
- R-08/R-14: este projeto permanece **pesquisa** até decisão clínica e regulatória separada;
  nada nesta especificação declara garantia clínica.
