Deploying the qubo FastAPI service

Options:
- Docker (recommended for production): build and run the included Dockerfile.
- Vercel (for easy hosting): use `vercel` CLI with `vercel.json` (simple, but may require tweaks for long-running processes).

Docker example:

```bash
docker build -t qubo-service .
docker run -e QUBO_API_KEY="your-key" -p 8000:8000 qubo-service
```

Heroku/Render:
- Use the included `Procfile`.
- Set `QUBO_API_KEY` and `QUBO_ALLOW_INSTALL=false` in environment.

Security notes:
- The `/install` endpoint will be disabled by default unless `QUBO_ALLOW_INSTALL=true` in the environment. When enabled, require `QUBO_API_KEY` to avoid abuse.
