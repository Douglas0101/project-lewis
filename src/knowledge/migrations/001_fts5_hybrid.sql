CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    content,
    run_id UNINDEXED,
    stage UNINDEXED,
    status UNINDEXED,
    tokenize="porter"
);
