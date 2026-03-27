# Test Fixtures

Keep fixtures small and deliberate.

- `synthetic/` is for branch-targeted inputs with exact expected outputs.
- `sampled/` is for reviewed real rows promoted into regression fixtures.
- Prefer one idea per fixture.
- Do not snapshot large parquet outputs.
