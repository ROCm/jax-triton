`amd-main` is the `main` development branch of `jax-triton` based on
the upstream (https://github.com/jax-ml/jax-triton/). It contains additional
commits that might (or might not) eventually land into the upstream.

You're welcome to use it instead of the upstream to get additional functionality.

Treat it as a regular fork's `main` branch with the following important exceptions:

- **NEVER sync `amd-main` with the `upstream/main` manually**!
  There's a `.github/workflows/amd-sync-upstream2.yml` workflow implemented that
  performs daily branch synchronization (also runnable manually) and creates
  stacked PRs into `amd-main` should a diff introduce changes. It automatically
  strips away upstream's `.github` directory protecting our CI and lets maintainers
  to keep track on upstream changes and ensure the fork is always in a workable state.
  See the details in the workflow file.
  **Reminder:** ALWAYS merge these automation PRs with "Create a merge commit".

- Don't commit to the branch directly, use a PR instead.

