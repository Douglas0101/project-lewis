# Project-Lewis — Proveniência de Dados

> Documento exigido pela LGPD (Lei 13.709/2018) e GDPR para rastreabilidade de finalidade, base legal e sub-processadores de dados pessoais (mesmo de-identificados).

## 1. Inventário de Datasets

| Dataset | URL canônica | Source | Tamanho (bytes) | SHA256 | Verificado em |
| :--- | :--- | :--- | ---: | :--- | :--- |
| chapman_shaoxing | https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/ | physionet | 2458352898 | `2ac8e91a0830f8b1a7c3d06a698c0b27981adc4b6fc301149876d8dd126c3a83` | 2026-06-13T04:00:00+00:00 |
| mitdb | https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip | physionet | 77030320 | `47b26926927c11bd9174154d367c811afc1b186f650f8a205931ca8c520f0a87` | 2026-06-12T04:00:00+00:00 |
| svdb | https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip | physionet | 54576759 | `ef0cf2aae45234107f5239f31662b64afba2067d707bf52f3a3adbf2de9fd25b` | 2026-06-12T04:00:00+00:00 |
| afdb | https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip | physionet | 461033254 | `1f19cb2ac53b6d847f304cbd4bd11e7db32978c529213f88407e09053d8c817b` | 2026-06-12T04:00:00+00:00 |
| incartdb | https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip | physionet | 590842939 | `7be630362db0d593d26cb2b18a1c765078ba5281b956758ae6d78cb8c8c89a05` | 2026-06-12T04:00:00+00:00 |

## 2. Base Legal (LGPD Art. 7º)

- **Chapman-Shaoxing:** uso de pesquisa pública, dado de-identificado pelo publicador. IRB de Shaoxing People's Hospital e Ningbo First Hospital.
- **MIT-BIH Family e INCART:** pesquisa pública (PhysioNet), dados históricos de-identificados pelo publicador. Sem IRB formal (dados > 40 anos).

## 3. Finalidade

Treinamento, validação e exportação de modelo de classificação de arritmias cardíacas em ECG, embarcado em microcontrolador STM32F4. Não inclui re-identificação, não inclui comercialização de dados pessoais, não inclui cruzamento com outras bases.

## 4. Sub-processadores

| Nome | Serviço | Dados compartilhados | Localização |
| :--- | :--- | :--- | :--- |
| PhysioNet (MIT Lab) | Distribuição de datasets | Hash do dataset (auditoria) | EUA |
| Kaggle (Google) | Distribuição de Chapman | Hash do dataset (auditoria) | EUA |

## 5. Retenção e Descarte

- **Dados brutos (`data/raw_*/`):** retidos pelo tempo de vida do projeto. Ao final, delete via `rm -rf data/raw_*/` e arquive o SHA256 final em `data/audit/final_checksums.jsonl`.
- **Catálogo (`data/catalog/`):** retido indefinidamente (curado, pequeno).
- **DLQ e audit logs:** retidos por 7 anos (LGPD Art. 37) em cold storage.

## 6. Direitos do Titular

Por se tratar de dados públicos de-identificados pelo publicador, não há canal direto de exercício de direitos do titular (Art. 18 LGPD) pelo Project-Lewis. Encaminhar solicitações às instituições de origem.

## 7. Exceções de Processamento (Quality Gate QG1)

Durante a execução das Camadas 1 e 2 os headers de dois registros do Chapman-Shaoxing (`JS01052.hea` e `JS23074.hea`) foram encontrados com a linha de registro e a primeira linha de sinal concatenadas, impedindo a leitura pelo `wfdb`. Ambos foram corrigidos no `data/raw_chapman/` (backup em `.hea.bak`) e reprocessados com sucesso.

| Dataset | record_name | Arquivo bruto | Motivo | Status |
| :--- | :--- | :--- | :--- | :--- |
| chapman | `JS01052` | `data/raw_chapman/01/019/JS01052.hea` | Header com linhas concatenadas | Corrigido e processado |
| chapman | `S23074` | `data/raw_chapman/23/236/JS23074.hea` | Header com linhas concatenadas | Corrigido e processado |

> **Nota técnica:** os headers originais foram preservados com extensão `.hea.bak` para auditoria. O catálogo e o lineage foram atualizados; não há mais registros skipped por erro de parser.

## 8. Estratégia de Range Físico e Z-Score

A `Camada-02-Resample-Preprocessamento-v1.1.md` exige que o sinal convertido para unidades físicas esteja no intervalo `[-5, +5] mV`. Diversos registros brutos violam esse intervalo, especialmente em INCART (offsets de ganho chegam a ~±107 mV). Descartar esses registros reduziria a completude do fine-tuning em mais de 60%, comprometendo o treinamento.

Para manter a **completude** sem introduzir outliers no treinamento, o pipeline implementa clipping em duas etapas:

1. **Pós-filtro/detrend (`clip_outliers`):** valores fora de `[-5, +5] mV` são truncados **após** a remoção de baseline wander (filtro 0.5–40 Hz) e detrend linear. Isso preserva a morfologia QRS e remove picos residuais.
2. **Pós-normalização (`post_normalize_clip`):** z-scores residuais fora de `[-10, +10]` são truncados após o z-score global. Isso evita que registros com amplitude muito acima da média do dataset dominem gradientes durante o treinamento.

Ambos os passos são configuráveis em `config/preprocess_v1.0.yaml` e documentados em cada `data/lineage/{dataset}/{record}_lineage.json`.

## 9. Nota sobre a frequência do SVDB

A documentação `docs/Camada-01-Ingestao-v1.1.md` indica **250 Hz** para o MIT-BIH SVDB, mas os arquivos brutos baixados da PhysioNet e amplamente referenciados na literatura operam a **128 Hz**. O pipeline usa a frequência declarada no header de cada registro (`wfdb.rdheader().fs`), portanto o SVDB é resampleado de 128 Hz → 500 Hz. Recomenda-se revisar a especificação para refletir 128 Hz.

## 8. Histórico de Mudanças

| Data | Versão | Mudança | Responsável |
| :--- | :--- | :--- | :--- |
| 2026-06-09 | 1.0 | Criação | Douglas Souza |
| 2026-06-14 | 1.1 | Adicionada seção de exceções de processamento (skip S23074/Chapman) | Douglas Souza |

