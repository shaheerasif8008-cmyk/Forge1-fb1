# Bootstrapping Checks

After the Docker Compose stack is running, capture baseline health information:

```bash
make health
```

The command stores `health.json` and the first 50 metric lines in this directory. Re-run it whenever you need a fresh snapshot of the runtime state.
