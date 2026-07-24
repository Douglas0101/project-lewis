# 08 — Schema do Bundle de Artefatos (indivisível)

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `artifact_bundle_schema` (11)
**Corrige:** DQ-06 (modelo/scaler/threshold de gerações diferentes; promoção com `passes_qg5=false`)

---

## 1. Princípio

Cada execução de treinamento/calibração produz **um** bundle indivisível. Nenhum arquivo
individual é promovido separadamente. Um artefato sem bundle completo e válido é, por definição,
`ORPHAN_ARTIFACT` e não pode ser carregado por nenhum caminho produtivo ou de teste.

## 2. Componentes obrigatórios (15)

```text
 1. raw-data manifest          (datasets, versões, hashes dos brutos)
 2. processed-data manifest    (hashes dos derivados v3)
 3. ontology                   (tabela única versionada, ver 01)
 4. preprocessing contract     (pipeline v3, ver 02/03)
 5. feature schema             (features v3.0.0, ver 03)
 6. patient split manifest     (folds/seeds congelados, ver 05)
 7. training configuration     (config completa + seeds)
 8. model                      (pesos + arquitetura)
 9. scaler                     (fit no treino do fold de origem)
10. calibrator                 (por tarefa/hierárquico, ver 07)
11. threshold policy           (co-produzida, justificada, congelada)
12. metrics                    (bateria completa + ICs, ver 05)
13. environment                (runtime identity, ver config/runtime_identity.json)
14. source revision            (commit + árvore)
15. risk report                (achados, limitações, waivers)
```

## 3. Digest composto

Cada componente i contribui `H_i = SHA256(bytes_canônicos_i)`. O digest do bundle:

```math
H_{\mathrm{bundle}} = \mathrm{SHA256}\!\left(
H_{data} \,\|\, H_{ontology} \,\|\, H_{preprocessing} \,\|\, H_{split} \,\|\,
H_{model} \,\|\, H_{scaler} \,\|\, H_{calibrator} \,\|\, H_{threshold} \,\|\, H_{metrics}
\right)
\]

**Serialização canônica obrigatória:** lista ordenada de pares `(nome_ascii, hash_hex_lowercase)`,
codificada com prefixo de comprimento por campo e rótulo de domínio
(`"project-lewis/bundle/v1"` como prefixo de domínio). **Proibida** concatenação textual simples
sem separação inequívoca de campos (evita colisões de enquadramento).

## 4. Regras estruturais

1. Modelo, scaler, calibrador e threshold apontam para **a mesma geração** (mesmo
   `training_run_id`); divergência → hard reject.
2. Threshold co-produzido com modelo+calibrador na validação interna; threshold órfão → hard
   reject (caso atual: 0,58 de jun/26 carregado com modelo de jul/04 → REJEITADO).
3. `ontology_hash` e `preprocessing_hash` idênticos em treino e em inferência; qualquer
   diferença → hard reject.
4. `patient split manifest` imutável após criação; regeneração exige novo `split_version`.
5. Bundle parcial não pode ser promovido, carregado ou avaliado como candidato de produção.
6. `passes_qg5=false` (ou qualquer gate ratificado falho) em `metrics` → promoção bloqueada.
7. Waiver não aprova: todo waiver gera `REVIEW_REQUIRED` com revisão humana registrada.
8. Carregadores (pipeline de inferência, testes QG) resolvem artefatos **somente** via bundle:
   path + hash verificados contra o manifest antes de desserializar.

## 5. Localização dos hashes de geração

`training_run_id` liga: split manifest, scaler (n_amostras de fit = tamanho do treino do fold),
threshold (validação do mesmo fold), métricas e modelo. Verificação cruzada obrigatória no
carregamento: o scaler deve ter `n_samples_seen_` consistente com o manifest do fold.

## 6. Critérios de aceite

1. Schema validado por contrato estrito (extra=forbid) — alinhado a
   `src/security/authenticated_decision_contracts.py`.
2. Teste de round-trip: gerar manifest → verificar → rejeitar qualquer componente alterado.
3. Teste negativo: bundle sem um componente → reject; bundle com componente de outra geração →
   reject.
