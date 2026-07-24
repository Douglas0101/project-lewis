# 11 — Política de Gates de Promoção

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `promotion_gate_policy` (14)

---

## 1. Forma da decisão

A promoção de qualquer bundle (08) exige: gates automáticos verdes **+** attestation válida (09)
**+** quorum humano registrado. Não existe aprovação automática de produção. Todo waiver gera
`REVIEW_REQUIRED` — nunca aprovação por exceção.

## 2. Hard reject (qualquer um bloqueia)

```text
HR-01  desalinhamento temporal (qualquer gate G-T1…G-T8 de 02 falho)
HR-02  leakage de paciente (paciente em mais de um split, em qualquer nível)
HR-03  classe sem definição clínica na ontologia ratificada
HR-04  modelo aprendendo identidade do dataset (DATASET_SHORTCUT_LEARNING, 06)
HR-05  modelo e calibrador de gerações diferentes (trainingRunId divergente)
HR-06  threshold órfão (não co-produzido com modelo+scaler+calibrador)
HR-07  outer test usado em qualquer seleção (features, arquitetura, preprocessing,
       threshold, calibração, seed, política de classes, resampling, class weights)
HR-08  bundle incompleto ou componente sem hash verificável
HR-09  assinatura/attestation inválida, expirada ou fora de sequência
HR-10  hash divergente em qualquer verificação de carregamento
HR-11  suporte estatístico insuficiente (pacientes/classe ou IC incompatível com a margem)
HR-12  desempenho crítico abaixo do gate ratificado (qualquer classe/tarefa crítica)
HR-13  calibração que melhora ECE mas piora sensibilidade crítica
HR-14  modelo reprovado em gate sendo promovido (passes_qg5=false ou equivalente)
```

(HR-14 codifica diretamente o incidente atual: o modelo em produção declarou
`passes_qg5=false` na run produtora.)

## 3. Review required (revisão humana obrigatória antes de prosseguir)

```text
RR-01  qualquer waiver
RR-02  doença rara com intervalo de confiança amplo
RR-03  calibrador específico de domínio (legítimo e declarado)
RR-04  drift externo relevante (10)
RR-05  mudança de prevalência
RR-06  mudança de equipamento/dispositivo/lead
RR-07  nova classe ou alteração de ontologia (nova ontologyVersion)
RR-08  adaptação pós-uso (qualquer transição além de LEVEL_1, 07 §6)
RR-09  evidência clínica ambígua
RR-10  regeneração de folds/splits (novo split_version)
```

## 4. Gates de desempenho (valores sujeitos a ratificação com dados v3)

Os gates científicos v2.x atuais (QG5': recall Anormal ≥ 0,30; F1-macro S2 ≥ 0,45 etc.) foram
definidos sob dados defeituosos e **não se transportam**. A tabela abaixo registra o formato;
os valores finais são `PROPOSED_REQUIRES_RATIFICATION` após a matriz 100 células (05):

| Tarefa | Métrica primária | Piso candidato | IC obrigatório | Status |
|---|---|---|---|---|
| beat: N/S/V/FUSION | F1 por classe + recall de triagem | definir pós-matriz | bootstrap 10k por paciente | PROPOSED_REQUIRES_RATIFICATION |
| beat: triagem anormal | recall | definir pós-matriz | idem | PROPOSED_REQUIRES_RATIFICATION |
| rhythm: AFIB/AFL | F1 + sensibilidade de episódio | definir pós-D3 | idem | PROPOSED_REQUIRES_RATIFICATION |
| quality/abstenção | risco–cobertura | definir | idem | PROPOSED_REQUIRES_RATIFICATION |
| calibração | NLL/Brier/ECE_c | não degradar vs baseline | idem | PROPOSED_REQUIRES_RATIFICATION |

## 5. Procedimento de promoção

1. Verificação automática HR-01…HR-14 contra o bundle e a attestation.
2. Revisão humana de qualquer RR-xx aplicável; registro de decisão (12) com hash.
3. Quorum (09 §5) e emissão da attestation; atualização do ponteiro de produção **somente**
   para o bundle inteiro.
4. Registro imutável da promoção (sequence monotônica, decisionId, validade UTC).

## 6. Critérios de aceite

1. Implementação da verificação HR-xx como contrato estrito com testes positivos e negativos.
2. Nenhum caminho alternativo de promoção (processo único).
3. Auditoria anual da política; mudanças = nova versão de política, nunca edição silenciosa.
