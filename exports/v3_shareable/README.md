# Project-Lewis v3 — Pacote Compartilhável (resultados da reconstrução)

**Gerado em:** 2026-07-22 · **Destinatário:** revisão técnica externa (Engenharia de Dados)
**Remetente:** Douglas Souza — Project-Lewis (classificação de arritmias ECG em edge, STM32F4)

---

## 1. O que é este pacote

Resultados completos da **reconstrução v3 do pipeline de dados** após auditoria forense que
encontrou (e provou) um defeito grave de relógio no pipeline anterior: índices de anotação em
frequência nativa (360/128/257 Hz) aplicados diretamente sobre o sinal reamostrado a 500 Hz,
desalinhando janelas e rótulos (drift até ~22 min) e corrompendo features RR por fator fixo por
dataset. O modelo anterior tinha ROC-AUC ≈ 0,55 (quase aleatório). Após a correção, a mesma
arquitetura atinge **AUC 0,90–0,97** — prova de que a falha era de dados, não de modelo.

## 2. Conteúdo

```text
results/
  MATRIX_SUMMARY.md      tabelas completas (agregados + fold×seed por família)
  ledger.json            índice das 100 células com hashes
  cells/*.json           100 células: métricas completas (AUC, PR-AUC, F1, MCC, recall,
                         precision, especificidade, Brier, ECE, matriz de confusão, suporte,
                         threshold da inner-val, runtime, hashes de dados/ontologia/split)
manifests/
  training_manifest.json  manifest v3.0.0 do dataset (contagens, versões)
  splits_v3/              manifests congelados de folds por paciente (+ afdb), com SHA-256
docs/
  forensic_data_quality_report_v1.0.md   auditoria forense completa (evidências e provas)
  rebuild_spec/                           especificação de reconstrução (14 docs: ontologia,
                                          alinhamento temporal, features, treino, calibração,
                                          bundle, attestation, monitoramento, gates, riscos)
```

## 3. Resultado central (beat task: N vs Anormal, inter-paciente)

| Família | ROC-AUC (25 células) |
|---|---|
| (a) CNN-1D waveform | 0,8985 ± 0,043 |
| (b) MLP features | 0,9655 ± 0,019 |
| (c) fusão CNN+features | 0,9665 ± 0,012 |
| (d) multitarefa beat+quality | 0,9032 ± 0,037 |

- (b) × (c): empate estatístico em AUC (Δ=+0,001, IC95% [−0,004; +0,007]); (b) tem melhor
  recall (+4,1pp), (c) melhor precision/F1/MCC — decisão final depende da métrica clínica
  primária (em ratificação).
- Cabeça de ritmo AFDB (auxiliar da família d): F1-macro 0,155 — **não aprende** com a
  arquitetura/suporte atuais (limitação documentada).
- Determinismo bit-a-bit comprovado: mesma célula (fold, seed, dados) → mesmo AUC.

## 4. Por que NÃO há pesos de modelo neste pacote

Por governança: pesos só são exportados pelo runner canônico vinculado à geração (bundle com
hashes modelo+scaler+calibrador+threshold). O runner de pesquisa usado na matriz foi arquivado
e o caminho canônico está em transição (task #3 do projeto). Este pacote é **somente resultados
e evidências**, suficiente para revisão técnica do pipeline e do protocolo.

## 5. Verificação de integridade

- Cada célula traz `data_hash`, `ontology_hash`, `preprocessing_hash`, `fold_manifest_sha256`.
- Os manifests de split trazem SHA-256 próprio por fold.
- Dados: MIT-BIH family (mitdb/svdb/incart) + AFDB (ritmo) — dados públicos PhysioNet,
  sem PII (IDs de registro públicos; LGPD preservada).
- Licenças: MIT-BIH (ODC-BY); citar PhysioNet em qualquer uso público.

## 6. Próximos passos previstos no projeto

1. Ratificação da família vencedora ((b) × (c)) pela métrica clínica primária.
2. Auditoria anti-atalho: probe `dataset_id` nos embeddings, leave-one-dataset-out, testes
   contrafactuais (amplitude/fs/padding/ruído).
3. Calibração por tarefa (temperature/vector/Dirichlet + hierárquica para classes raras) em
   partição independente; bundle v3 assinado (DSSE/in-toto/Sigstore, shadow).
4. Gates de promoção: threshold co-produzido, IC bootstrap por paciente, revisão humana.
