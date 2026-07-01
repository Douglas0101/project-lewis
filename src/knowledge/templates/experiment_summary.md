# Experimento: {{ experiment_name }} | Estágio: {{ stage }} | Status: {{ status }}

## Resumo Executivo

{{ executive_summary }}

## Configuração

- **Run ID:** {{ run_id }}
- **Tipo de run:** {{ run_type }}
- **Início:** {{ start_time }}
- **Fim:** {{ end_time }}
- **Duração:** {{ duration_seconds }} s
- **Diretório de artefatos:** {{ artifact_dir }}
- **Git commit:** {{ git_commit }}
- **Caminho de config:** {{ config_path }}

## Métricas Finais vs Baseline

| Métrica | Valor | Baseline | Δ |
|---------|-------|----------|---|
{% for metric in metrics_table -%}
| {{ metric.name }} | {{ metric.value }} | {{ metric.baseline }} | {{ metric.delta }} |
{% endfor %}

## Alertas e Anomalias

{% if alerts %}
{% for alert in alerts -%}
- **[{{ alert.severity }}]** {{ alert.category }}: {{ alert.message }}{% if alert.metric_name %} ({{ alert.metric_name }}={{ alert.metric_value }}, threshold={{ alert.threshold }}){% endif %}
{% endfor %}
{% else %}
Nenhum alerta registrado.
{% endif %}

## Artefatos

{% if artifacts %}
{% for artifact in artifacts -%}
- `{{ artifact.artifact_type }}`: {{ artifact.path }} (checksum: {{ artifact.checksum }})
{% endfor %}
{% else %}
Nenhum artefato registrado.
{% endif %}

## Recomendações para Próximos Experimentos

{% if recommendations %}
{% for rec in recommendations -%}
- {{ rec }}
{% endfor %}
{% else %}
Nenhuma recomendação automática gerada.
{% endif %}

## Metadados para Indexação

- **experiment_id:** {{ experiment_id }}
- **run_id:** {{ run_id }}
- **stage:** {{ stage }}
- **status:** {{ status }}
- **health_status:** {{ health_status }}
- **tags:** {{ tags|join(', ') }}
