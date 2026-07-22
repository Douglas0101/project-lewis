# Política de Decisão Autenticada de Pesquisa v1

- **Policy ID:** `project-lewis/research-decision/v1`
- **Predicate type:** `https://project-lewis.dev/attestations/research-decision/v1`
- **Status:** `SHADOW_ONLY`
- **Versão:** `1.0.0`
- **Autoridade operacional:** nenhuma
- **Escopo executável:** calcular resultados de revisão para a transição `E06_5_AUDIT`

## 1. Limite de autoridade

Esta versão é deliberadamente fail-closed e não operacional. Ela não pode:

- assinar decisões;
- iniciar ou retomar E06.5, E07 ou E08;
- alterar CI/CD;
- aprovar publicação ou pesquisa concluída;
- liberar firmware, modelo ou deployment;
- substituir revisão clínica, estatística, de qualidade, segurança ou regulatória;
- consumir estado produtivo de antirreplay.

Os únicos resultados permitidos são:

1. `REJECTED_AUTHENTICATED`;
2. `INSUFFICIENT_EVIDENCE`;
3. `REVIEW_REQUIRED`;
4. `APPROVED_FOR_AUDIT`.

Todo relatório deve declarar `shadow=true` e `operational=false`. Mesmo
`APPROVED_FOR_AUDIT` representa somente o resultado contrafactual da política;
não constitui autorização executável.

## 2. Requisitos obrigatórios aprovados

### 2.1 Parsing e contratos

- JSON deve ser UTF-8, limitado em tamanho e rejeitado antes do Pydantic quando houver:
  chaves duplicadas, `NaN`, `Infinity`, `-Infinity`, tipos ambíguos ou campos desconhecidos.
- Contratos Pydantic v2 usam `strict=True`, `extra="forbid"` e modelos congelados.
- DSSE aceita apenas `payloadType=application/vnd.in-toto+json` e exatamente uma assinatura
  por bundle neste perfil.
- O Statement deve usar `_type=https://in-toto.io/Statement/v1` e o predicate type desta policy.
- Subject, policy e evidências são vinculados exclusivamente por SHA-256 lowercase.
- Nenhum path vindo do payload pode ser aberto. Arquivos são fornecidos pelo chamador por
  bindings confiáveis e root-constrained.

### 2.2 Criptografia e identidade

- O backend Sigstore é uma interface injetável; nenhum backend produtivo foi ratificado.
- Backend ausente, trusted time ausente ou falha de infraestrutura resulta em
  `REVIEW_REQUIRED`, nunca aprovação.
- Falha criptográfica comprovada, bundle hash divergente ou identidade não autorizada resulta em
  `REJECTED_AUTHENTICATED`.
- Papel, issuer e identity são derivados exclusivamente da allowlist local. Metadados submetidos
  não podem atribuir papel ao signer.
- Comparação de issuer e identity é byte-exata, sem normalização implícita ou regex permissiva.
- Quorum de `APPROVED_FOR_AUDIT`: um `EVIDENCE_BOT` e um `SCIENTIFIC_APPROVER`, identidades e
  authorization records distintos.
- Solicitante, evidence bot e scientific approver devem ser independentes.
- Relatórios não registram email/SAN nem seu hash previsível; registram somente authorization IDs
  locais e papéis.

### 2.3 Integridade, validade e replay

- Bundles usados no quorum precisam autenticar payload DSSE byte-identical.
- A aplicação deve analisar exatamente os bytes autenticados pelo backend, sem reserialização.
- Trusted time deve vir de fonte autenticada (`REKOR`, `RFC3161` ou equivalente ratificado).
- `decisionId` deve ser UUID canônico v4 ou v7.
- Nonce deve conter exatamente 256 bits base64url sem padding.
- Sequência é inteira estrita, não negativa e monotônica por subject/scope.
- `issuedOn`, `notBefore` e `expiresOn` são UTC `Z`; relógio e duração obedecem à policy local.
- Antirreplay v1 é somente em memória/fixture. Por isso nenhum resultado é operacional.
- Waiver sempre produz `REVIEW_REQUIRED`, com uma exceção de precedência: falhas de autenticidade
  (assinatura, hash, validade, identidade ou input malformado) são rejeitadas primeiro como
  `REJECTED_AUTHENTICATED`. Um waiver não pode suavizar uma rejeição criptográfica ou estrutural.
- Falta ou insuficiência de evidência produz `INSUFFICIENT_EVIDENCE`.

## 3. Gates existentes do Project-Lewis

Estes gates já existem no projeto e não são redefinidos por esta policy:

### 3.1 Pré-condições para solicitar audit E06.5

- preflight determinístico/CPU válido e content-addressed;
- source/runtime/config identities correspondentes;
- matriz congelada `baseline,H6,H11,H12 × folds 1..5 × seeds 17,29,43,71,101`;
- smoke exato de quatro células, fold 1/seed 17;
- `E06_5_SMOKE_PASS` derivado de `DONE`, não de constante declarada;
- artifact maps e SHA-256 válidos;
- zero patient overlap e zero outer-test selection/fitting;
- templates, preprocessamento, sampler, priors e métodos train-only;
- save/reload prediction delta `<=1e-7`;
- audit target com zero `DONE` antes da primeira autorização;
- `--force` nunca sobrescreve run finalizado.

### 3.2 Gates científicos existentes

- status E06 permanece
  `REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET / ROBUSTNESS_VALIDATION_REQUIRED`;
- seleção pareada usa folds/seeds idênticos;
- H11 não substitui H6 se o ganho não exceder variabilidade entre seeds;
- H12 não vence por diferença isolada de `0.0001`;
- QG5 estágio 2 mantém F1(S) `>=0.55`, F1(V) `>=0.70`, F1(F) `>=0.15` e
  F1-macro `>=0.45`;
- target final de publicação permanece F1(F) `>=0.50`;
- E07/E08 permanecem bloqueados até release válido de E06.5.

Os gates científicos acima não são usados por esta primeira policy para aprovar publicação ou
release. O único escopo calculável é a elegibilidade shadow para iniciar o audit.

## 4. Thresholds propostos — não executáveis

Os parâmetros abaixo são propostas de governança, não thresholds universais derivados dos papers
e não podem alterar resultado em v1:

| Parâmetro proposto | Valor de discussão |
| --- | ---: |
| Bootstrap pareado por paciente | 10.000 repetições |
| LCB 95% de superioridade | `delta F1 > 0` |
| LCB auxiliar de F1(F) | `>=0.40` |
| Pior fold F1(F) | `>=0.30` |
| Percentil 10 fold×seed F1(F) | `>=0.40` |
| SD entre seeds de F1(F) | `<=0.05` |
| Suporte por fold | `>=2` pacientes F e `>=20` beats F |
| Calibration intercept | `[-0.10, 0.10]` |
| Calibration slope | `[0.80, 1.20]` |
| ECE adaptativo por classe | `<=0.05` |
| Gap de subgroup | `<=0.10` |

Ativação exige ratificação clínica, estatística e de risco, análise de viabilidade por paciente e
nova versão major/minor da policy. Evidência insuficiente nunca pode ser convertida em pass.

## 5. Evidência e decisão

Hashes válidos não provam semântica válida. `APPROVED_FOR_AUDIT` exige simultaneamente:

- backend criptográfico e evidence evaluator explicitamente fornecidos;
- quorum independente;
- subject, policy e todos os evidence digests recalculados;
- evidence evaluator confiável retornando `PASS` para todos os gates obrigatórios;
- nenhum waiver;
- validade/replay aprovados.

Nesta etapa somente evaluators de fixture podem retornar `PASS`; por isso o relatório continua
não operacional. Claims e reason codes assinados são validados contra vocabulário controlado, mas
o resultado usa apenas reason codes calculados pelo verifier.

## 6. Provenance e referências

A estrutura segue:

- [DSSE protocol v1.0.2](https://github.com/secure-systems-lab/dsse/blob/v1.0.2/protocol.md);
- [in-toto Statement v1](https://github.com/in-toto/attestation/blob/v1.0/spec/v1.0/statement.md);
- [Sigstore verification](https://docs.sigstore.dev/cosign/verifying/verify/);
- [Sigstore bundle format v0.3.2](https://docs.sigstore.dev/about/bundle/);
- [SLSA v1.2 artifact verification](https://slsa.dev/spec/v1.2/verifying-artifacts);
- [NIST SSDF SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final);
- [NIST AI RMF 1.0](https://doi.org/10.6028/NIST.AI.100-1);
- [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378);
- [PROBAST+AI](https://doi.org/10.1136/bmj-2024-082505);
- [FDA PCCP final guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence).

## 7. Pontos que exigem ratificação humana

- implementação e trust root do backend Sigstore;
- allowlist produtiva de issuer/identity e proteção dos authorization records;
- evidence adapters que rederivem preflight/smoke/matrix/leakage/testes;
- ledger SQLite transacional e política de replay/revogação;
- duração, clock skew, retenção e supersession;
- significado jurídico do nome `REJECTED_AUTHENTICATED` quando a entrada não autentica;
- tratamento LGPD de identidades presentes em certificados/transparency logs;
- todos os thresholds da seção 4;
- qualquer transição além de `APPROVED_FOR_AUDIT` shadow.
