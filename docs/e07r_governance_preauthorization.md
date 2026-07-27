# E07R — preautorização de governança autônoma

- **Data:** 2026-07-26
- **Modo:** `AUTONOMOUS_PREAUTHORIZED`
- **Responsável automatizado:** `AUTONOMOUS_GOVERNANCE_PREAUTH`
- **Intervenção humana exigida nesta simulação:** não
- **Origem:** remediação do checkpoint E07 `BLOCKED`

## 1. Declaração

Esta execução representa uma simulação realista de governança experimental sem intervenção humana. Este documento registra — sem ampliar — a autorização explicitamente concedida pela instrução corrente do usuário. Ela permite corrigir leakage, publicar uma nova geração Stage 2 versionada e com custódia ordenada, regenerar evidência patient-disjoint e auditar sampling. Ela não reduz thresholds, não transforma pesquisa em publicação e não concede autoridade para promover modelos.

O bloqueio herdado está registrado em:

- `docs/e07_blocked_report.md`;
- `experiments/stage2_v2.4_research/E07_blocked_20260726.json`;
- SHA-256 do checkpoint: `d8a684c0e29a91033a60fa10c4f9257c7ece1059a19ebe5472b2accc70749f63`.

## 2. Fonte oficial de identidade

A governança aceita como fonte primária a documentação oficial do PhysioNet:

<https://physionet.org/physiobank/database/html/mitdbdir/intro.htm>

Trecho autenticador:

> “Records 201 and 202 came from the same male subject.”

A evidência cacheada e sua política de uso estão em `docs/physionet_mitdb_patient_statement.md`.

## 3. Ações preautorizadas

1. Criar e congelar mapeamentos `record_id → patient_id` versionados.
2. Vincular MIT-BIH 201 e 202 ao mesmo `patient_id`.
3. Reutilizar evidência de identidade autenticada já existente para MIT-BIH e INCART, validando seus hashes antes do consumo.
4. Tratar identidades não autenticadas de forma conservadora, sem afirmar identidade biológica inexistente e sem permitir que registros incertos atravessem folds.
5. Regenerar, sem tocar os parents, um Stage 2 r5 write-once com binding ordenado `sample_id`/`waveform_sha256`, clocks DQ-01/02 e escopo confirmatório MIT-BIH+INCART.
6. Criar splits outer e inner patient-disjoint `v4.0-patient-disjoint` sobre o r5 autorizado.
7. Marcar os splits record-disjoint antigos como `QUARANTINED_NOT_DELETED`, preservando bytes e paths históricos.
8. Criar manifests, relatórios e checkpoints novos, aditivos e content-addressed.
9. Implementar proteção lógica de escrita, freeze manifest e permissões somente leitura para pins.
10. Executar testes completos e preflight fail-closed.
11. Reexecutar E06.5-PD em 100 células como audit de viabilidade, com H6 pré-comprometido para E07 antes dos outer outcomes.
12. Se H6-PD produzir evidência válida, executar E07-PD em 150 células.
13. Assinar checkpoints de governança com `AUTONOMOUS_GOVERNANCE_PREAUTH`.

## 4. Limites e ações proibidas

Permanecem proibidos:

- publicação externa;
- promoção ou escrita em `models/`;
- alteração de `backup_v2.3/` ou da quarentena v3.1;
- sobrescrita de dataset, split, relatório, checkpoint ou run histórico;
- afrouxamento de QG5' ou do target F1(F) ≥ 0,50;
- uso de teste/validação para fitting de sampler, pesos, thresholds ou seleção;
- remoção ou mascaramento de folds problemáticos;
- cherry-picking de seeds, folds ou braços;
- declaração de ganho com leakage ou de prontidão sem todos os gates.

## 5. Política conservadora de identidade

- **MIT-BIH:** 201/202 formam um único paciente; demais records seguem a regra default documentada, salvo nova evidência oficial.
- **INCART:** consumir somente o mapeamento autenticado por headers WFDB e manifesto v3.1.0, após verificação de hash.
- **SVDB:** como a identidade biológica está `IDENTITY_UNVERIFIED`, `patient_id` permanece nulo. Os records recebem somente uma barreira não biológica de partição e ficam fora do fitting confirmatório, de S2/S4 e de métricas por paciente. Podem aparecer apenas como sensibilidade de domínio separada.

Desbalanceamento ou perda de suporte deve ser preservado e reportado; balanceamento nunca prevalece sobre disjointidade ou proveniência.

## 6. Gates mantidos

- QG5' Stage 2: F1(S) ≥ 0,55; F1(V) ≥ 0,70; F1(F) ≥ 0,15; F1-macro ≥ 0,45.
- Target principal/publicação: mean F1(F) ≥ 0,50.
- `models/`: somente leitura e intocado.
- Publicação: `HOLD` até satisfação integral dos gates, sem publicação externa nesta missão.
- Próxima etapa final: `NONE`.

## 7. Assinatura autônoma

```text
responsible = AUTONOMOUS_GOVERNANCE_PREAUTH
governance_mode = AUTONOMOUS_PREAUTHORIZED
date = 2026-07-26
publication_authority = NONE
model_promotion_authority = NONE
gate_relaxation_authority = NONE
```
