# Controlled retraining and model release

TRM-013 keeps model retraining separate from production promotion. Drift reports are a review
signal; they never start training or move an MLflow alias by themselves.

## Prepare an approved snapshot

Start the containerized tracking stack and make sure the registry has a validated `production` model
before running a controlled retrain. A first baseline can use the existing `make train-tracking`
command and an explicit `make promote-tracking`; later retrains must use `make retrain-tracking`.

Create a one-time manifest after the feature dataset passes the schema, duplicate-ID, and split
overlap checks:

```powershell
make prepare-retraining-manifest `
  APPROVED_BY="risk-reviewer" `
  APPROVAL_REASON="Reviewed the labeled snapshot and split period." `
  APPROVED_MANIFEST="artifacts/retraining/approved-manifest-2026-09.json"
```

The manifest records file SHA-256 values, split counts/time ranges, feature schema, model config,
and training-code fingerprints. The output path is create-only. Keep the approved file under
controlled versioned storage; its SHA-256 digest detects accidental edits. Review/approve it through
the normal source-control or data-governance process, since the digest is not a cryptographic
signature. Keep `data.features_path` repository-relative so the host and a scheduled container can
resolve the same logical path.

Run the approved retrain:

```powershell
make retrain-tracking `
  APPROVED_MANIFEST="artifacts/retraining/approved-manifest-2026-09.json"
```

The job recomputes the manifest before model fitting and exits if data, schema, config, or source
code differs. A source-code fix also changes the code fingerprint: review the updated snapshot and
create a newly approved manifest at a new path before retrying; do not edit or reuse the old approval.
It requires the active production alias and scores that incumbent on the candidate's
same held-out test split. Validation PR-AUC/precision/recall and test PR-AUC/precision/recall,
expected business cost per evaluated transaction, Brier score, expected calibration error, alert rate, sample size, and data-contract
checks are recorded in the quality-gate report. The policy digest and incumbent version are attached
to the run and registered version.

The starting values in `configs/retraining.yaml` are PaySim demonstration gates, not business-approved
production limits. A risk owner must calibrate those values with representative labeled data and
document the decision before using them for a real release.

On a rejected attempt, MLflow retains the run, report, manifest, and unregistered model artifact. The
command exits with status 2 for a quality-gate rejection. It does not create a model version or move
the `candidate`, `staging`, or `production` alias. Local artifacts are isolated under
`artifacts/retraining/runs/` and do not replace the serving bundle. Data/schema preflight failures
are also recorded with the approved manifest when the tracking server is available.

On a passing attempt, MLflow creates a registered candidate and moves only the `candidate` alias.
Production and staging still require a named reviewer and reason:

```powershell
make promote-tracking `
  MODEL_VERSION="7" `
  PROMOTION_STAGE="staging" `
  APPROVED_BY="risk-reviewer" `
  PROMOTION_REASON="Reviewed the quality-gate report and model card."
```

Promotion metadata records the previous version for that alias, the new version, approver, reason,
and timestamp. Promotion refuses rejected versions and versions without valid model signature and
serving-input checks.

## Roll out and recover

After the production alias is approved, export it as a create-only model bundle, then roll it out to
kind:

```powershell
make export-production-model
make kind-rollout MODEL_BUNDLE_PATH="artifacts/model-bundles/random_forest-v7"
```

Use the exact directory reported by the export command. The rollout derives a content-addressed image
tag, loads the Podman-built image into kind from a temporary archive, updates the model-bundle init
container and model artifact path in one Deployment revision, and
waits for Kubernetes readiness. If image build/load, model startup, or readiness fails, it restores
the previous Kubernetes revision and the previous MLflow production alias. Automatic release
rollback requires a prior production model version; a first-ever model has no rollback target.
`make export-production-model` only exports the version currently assigned to the `production`
alias; it does not move that alias. To return from a promoted version to its recorded predecessor,
run `make rollback-tracking ROLLBACK_STAGE=production APPROVED_BY="risk-reviewer"
ROLLBACK_REASON="Restore prior approved version"` first. An already exported bundle is create-only
and need not be exported again. On a Windows host, the export client downloads artifacts through
MLflow's proxy; the internal `minio:9000` hostname is not host-resolvable. Rolling back the registry
alias alone does not replace a model already loaded by the local API container or deployed to kind;
deploy the previously approved bundle separately if serving must also return to that version.

For an intentionally rejected candidate demo, copy `configs/retraining.yaml` to a separate policy
file, set an impossible but valid threshold such as `minimum_test_recall: 1.0`, and run the controlled
entry point against a snapshot whose recall is below 1.0. Confirm the run is `rejected`, no model
version was registered, and production/staging aliases are unchanged. Do not change the shared policy
for this demo.

For a drift demo, run the monitoring job against an isolated shifted sample and inspect the segmented
Evidently report. Treat drift as an investigation trigger; approve a new labeled dataset manifest
before retraining. Do not edit or overwrite the dataset snapshot referenced by an existing approved
manifest.

## Scheduled execution

`deployment/kubernetes/retraining-cronjob.yaml` defines a weekly CronJob with concurrency disabled,
no retries, a finite deadline, non-root/read-only settings, and a read-only mount for the dataset and
approved manifest. Writable Spark scratch paths are explicitly mounted despite the read-only root
filesystem. It is deliberately suspended by default. Before enabling it, provision the
`transaction-risk-datasets` PVC, create the `transaction-risk-approved-dataset-manifest` ConfigMap
from the reviewed manifest, create `transaction-risk-retraining-secrets` with a cluster-reachable
HTTPS `tracking-uri` and optional `tracking-username`, `tracking-password`, or `tracking-token` keys,
and build/load `transaction-risk-training:local`. The MLflow endpoint must be
reachable from the cluster and must use appropriate TLS/authentication outside a local demo. Update
the manifest ConfigMap when a new dataset snapshot has been approved, then unsuspend the CronJob.

The CronJob reuses only the currently mounted approved manifest; it never creates approval or
promotes a model. Run it on demand with `kubectl create job --from=cronjob/transaction-risk-retraining`
after the prerequisites are in place. Keep the scheduler suspended until representative data,
tracking connectivity, and resource requirements have been validated.
