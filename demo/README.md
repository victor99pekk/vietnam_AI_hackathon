# VietGraph demo

This is the first presentation surface for the knowledge-graph pipeline. It
leaves the legacy `Html/` pages untouched and talks to the same-origin API:

- `GET /healthz`
- `GET /api/demo/sample`
- `POST /api/pipeline/run`

## Local run

From repository root, with the project environment active:

```bash
uvicorn kg_generator.api:app --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080/>.

The API attempts the real offline pipeline by default. Set
`KG_DEMO_USE_FULL_PIPELINE=0` to use the dependency-free fallback while
debugging a minimal container.
