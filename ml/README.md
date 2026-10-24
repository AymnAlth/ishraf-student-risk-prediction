# ML Data Preparation

This directory contains repeatable, offline preparation for the Ishraf student-risk learning project. It is deliberately isolated from API code, migrations, and production analytics so that experimentation cannot alter operational records.

## Commands

- `pnpm ml:profile` creates a schema catalogue and aggregate quality/readiness profile in `ml/artifacts/` using a repeatable, read-only PostgreSQL transaction.
- `pnpm ml:export:student-risk` creates `student-risk-training.csv` and metadata in `ml/artifacts/student-risk/`.
- `pnpm ml:generate:student-risk-synthetic` creates a separate 300,000-row synthetic training dataset by default: 10,000 fictional students with 30 historical snapshots each.

The scripts prefer `DATABASE_URL_MIGRATIONS` for a direct PostgreSQL connection and otherwise use `DATABASE_URL`. They never execute a data-modification statement.

Set `ML_EXPORT_PSEUDONYMIZATION_SALT` to a unique secret with at least 32 characters before exporting. Store it only in the approved secret manager or an untracked local environment. Do not reuse JWT, database, Firebase, or other application secrets as this salt.

## Privacy Boundary

Artifacts are ignored by Git. The feature export excludes names, academic numbers, contact details, birth dates, addresses, text notes, device data, and location coordinates. It uses an HMAC pseudonymous `student_key` only to keep historical rows of the same student together during train/test splitting.

## Current Scope

`student-risk` prepares a supervised dataset for an **offline** early-warning experiment: predict whether recorded presence will fall below the chosen threshold during the subsequent 14 days. It is not wired to the application and cannot make production decisions. See [the dataset contract](student-risk/DATASET_CONTRACT.md) before training.

## Synthetic Dataset

The synthetic generator has no database connection and never inserts into Neon. It creates internally consistent attendance, assessment, homework, behavior, and transport aggregates over three fictional academic years. The default size can be changed with `ML_SYNTHETIC_STUDENT_COUNT` and `ML_SYNTHETIC_SNAPSHOTS_PER_STUDENT`; use `ML_SYNTHETIC_SEED` to reproduce the same file.

Synthetic data is for coursework, pipeline validation, and safe model experimentation only. Keep it separate from real exports and never use synthetic-only metrics as evidence of production model performance.
