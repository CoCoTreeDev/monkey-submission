# Optional workflow examples

These six files are inactive references. They are outside `.github/workflows`
and use the suffix `.yml.disabled`; GitHub Actions does not load them.

They illustrate frontend deployment, edge benchmarking, historical backfill,
optional tunneling, model evaluation and model promotion. Local development and
`make ci` do not require any of them.

To adopt an example, first review every command and configure an isolated
runner with the example's `submission-edge` or `submission-deploy` label.
Provide your own service endpoints, credentials and data paths. The examples
use the checked-out workspace rather than another installed project clone.

Some examples install services, upload media, replace model files, or push a
model update. The promotion example also dispatches a deployment workflow.
Review those operations and the target branch before activating it. Configure
all credentials and repository variables explicitly for the intended environment;
none are supplied here. The default branch name may differ from the example's
`main` branch and must be adjusted to the repository's policy.

Retain the `.disabled` suffix and location unless you intend to enable an
integration. The application has no configured GitHub dispatch target by default.
