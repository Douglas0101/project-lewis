# Evidência oficial de identidade — MIT-BIH 201/202

- **Fonte:** PhysioNet, *MIT-BIH Arrhythmia Database Directory — Introduction*
- **URL:** <https://physionet.org/physiobank/database/html/mitdbdir/intro.htm>
- **Data de acesso:** 2026-07-26
- **Proveniência offline:** `false`
- **Uso autorizado:** exclusivamente governança de identidade e geração de splits patient-disjoint

## Citação oficial

> “Records 201 and 202 came from the same male subject.”

SHA-256 UTF-8 da sentença citada, sem aspas tipográficas:

```text
3927379ff48c27a336d5c1a086cf1a45f7eff02f2b81f655025728bd9258bf0e
```

## Interpretação operacional

Para qualquer split do Stage 2 que contenha MIT-BIH:

- os registros `201` e `202` recebem o mesmo `patient_id`;
- ambos devem permanecer na mesma partição de cada fold;
- separá-los entre treino, validação ou teste constitui leakage por paciente;
- a citação não será usada para seleção de modelo, sampling, threshold ou avaliação clínica.

## Limite da evidência

A fonte autentica especificamente que 201 e 202 pertencem ao mesmo indivíduo. Ela não autoriza inferir outros vínculos entre registros sem evidência adicional. Para os demais registros MIT-BIH, aplica-se a política documentada `record_id_equals_patient_id_unless_official_evidence`.
