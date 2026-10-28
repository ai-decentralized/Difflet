# Security Policy

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

Report them privately through GitHub's vulnerability reporting instead:
**[Report a vulnerability](https://github.com/ai-decentralized/Difflet/security/advisories/new)**
(Security tab → *Report a vulnerability*).

Please include:

- the affected component (for example `difflet serve`, the artifact store, or the CLI)
  and the version or commit;
- steps to reproduce, or a proof of concept;
- the impact as you understand it.

We aim to acknowledge reports within 5 business days, and to agree on a disclosure
timeline with you once the issue is confirmed.

## Supported versions

Difflet is pre-alpha. Security fixes land on `main` only. There are no maintained
release branches yet.

## Scope notes

`difflet serve` exposes an HTTP API with optional single-key authentication
(`--api-key` / `DIFFLET_API_KEY`), and can upload outputs to S3-compatible storage with
the credentials it is given. See the README's [Hardening](README.md#hardening) section.
In scope: bypassing the API key on `/v1` routes, leaking credentials or presigned URLs,
reading or writing outside the configured bucket or output paths, and code execution
through request payloads. Running the server unauthenticated on a public port is a
deployment choice, not a vulnerability.
