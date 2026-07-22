# 09 — Schema de Attestation Autenticada

**Status:** PROPOSTO — aguardando ratificação humana
**Data:** 2026-07-18
**Entregável:** `authenticated_attestation_schema` (12)
**Estende:** `docs/policies/authenticated_research_decision_v1.md` (v1, shadow/fail-closed),
`src/security/authenticated_decision_contracts.py`

---

## 1. Posicionamento

A política v1 existente cobre decisões de pesquisa em modo shadow. Este schema v2 cobre
**attestation de bundles de artefatos** (08): promoção, candidatos de calibração e ativação.
Permanece **fail-closed**: backend Sigstore ausente, trusted time ausente ou falha de
infraestrutura resultam em rejeição, nunca em aprovação por omissão.

## 2. Envelope e statement

- **Envelope DSSE**: `payloadType = application/vnd.in-toto+json`; exatamente as assinaturas
  exigidas pelo perfil de quorum (§5); payload UTF-8 limitado em tamanho; rejeição pré-Pydantic
  de chaves duplicadas, `NaN`/`Infinity`, tipos ambíguos ou campos desconhecidos.
- **Statement in-toto v1**: `_type = https://in-toto.io/Statement/v1`;
  `predicateType = https://project-lewis.dev/attestations/artifact-bundle/v2`.
- **Subject composto**: `H_bundle` (08 §3) como digest principal, com a lista nomeada de
  componentes (nome → sha256 lowercase). Nenhum path do payload é aberto: arquivos chegam por
  bindings confiáveis e root-constrained (mesma regra da v1).

## 3. Predicate versionado (v2, campos mínimos)

```json
{
  "bundleDigest": "sha256…",
  "components": {"model": "sha256…", "scaler": "sha256…", "calibrator": "sha256…",
                 "threshold": "sha256…", "ontology": "sha256…", "preprocessing": "sha256…",
                 "split": "sha256…", "metrics": "sha256…", "data": "sha256…"},
  "trainingRunId": "…", "sourceRevision": "…", "environmentHash": "sha256…",
  "decisionId": "…", "nonce": "…", "sequence": 0,
  "validFromUtc": "…", "validUntilUtc": "…",
  "policyId": "project-lewis/artifact-bundle/v2", "policyVersion": "2.0.0",
  "shadow": true, "operational": false,
  "outcome": "APPROVED_FOR_AUDIT|REVIEW_REQUIRED|INSUFFICIENT_EVIDENCE|REJECTED_AUTHENTICATED",
  "waivers": []
}
```

## 4. Identidade e transparência

- **Sigstore/cosign** como backend injetável (interface, nunca implementação embutida);
  identidade OIDC e issuer **exatos** declarados na política; certificado verificado contra
  Fulcio; prova de inclusão no log de transparência (Rekor) verificada; timestamp assinado
  (TSA) ou `trusted_time` injetado; mídia type
  `application/vnd.dev.sigstore.bundle.v0.3+json`.
- **Antirreplay**: `nonce` + `decisionId` + `sequence` monotônica por escopo + validade UTC
  (`validFromUtc`/`validUntilUtc`); consumo de estado antirreplay apenas por componente
  autorizado; shadow não consome estado produtivo (herdado da v1).

## 5. Quorum e resultados

- Perfis de quorum declarados por operação: promoção de bundle (evidence bot + aprovador
  científico humano independente); ativação de calibração (idem + revisor clínico).
- Resultados possíveis (mesmo conjunto da v1):
  `APPROVED_FOR_AUDIT`, `REVIEW_REQUIRED`, `INSUFFICIENT_EVIDENCE`, `REJECTED_AUTHENTICATED`.
  Todo relatório declara `shadow=true` e `operational=false` enquanto o backend produtivo não
  for ratificado.

## 6. Regras de verificação (hard reject em qualquer falha)

1. Assinatura inválida, identidade/issuer divergentes, certificado ou inclusão ausentes.
2. Qualquer hash de componente divergente do manifest; `H_bundle` não recomputa.
3. `trainingRunId` inconsistente entre componentes (modelo × scaler × calibrador × threshold).
4. `sequence` não monotônica, `nonce`/`decisionId` reutilizados, validade expirada.
5. Bundle parcial; `passes_qg5=false` em métricas; waiver sem revisão humana registrada.
6. Payload fora do contrato estrito (strict=True, extra=forbid, modelos congelados).

## 7. Critérios de aceite

1. Contratos Pydantic v2 estritos publicados (alinhados ao módulo `src/security/` existente).
2. Testes: aceitação de bundle válido (fixture), rejeição por cada regra da §6, antirreplay.
3. Backend Sigstore continua interface injetável até ratificação humana de um backend real —
   ausência de backend ⇒ apenas resultados shadow, nunca operacionais.
