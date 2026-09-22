# Dependency Constraints

`constraints-py312.txt` records the direct dependency versions validated in the local Python 3.12
environment. It is a reviewable constraints baseline, not a substitute for checking transitive
security updates.

Install the development environment with the same constraint file:

```powershell
python -m pip install -e ".[training,tracking,dev]" -c requirements/constraints-py312.txt
```

When upgrading, update the constraints intentionally, run the complete CI gate, and record the
reason in the change. Do not silently regenerate versions during a production deployment.
