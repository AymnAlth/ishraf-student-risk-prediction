# ML Data Profile

Generated: 2026-10-23T19:05:32.335Z

## Schema Inventory

- Schema: `public`
- Base tables: 55
- Views: 27
- Columns requiring exclusion or review: 160

## Operational Coverage

- Active students: 8
- Attendance: 57 records across 22 sessions (2026-03-19 to 2026-04-25)
- Assessments: 9 student records (2026-03-16 to 2026-04-13)
- Homework submissions: 10
- Behavior records: 9 (2026-03-19 to 2026-04-25)
- Transport events: 34 (2026-03-31 to 2026-10-08)
- Active academic years outside their configured date range: 1

## Student Risk Snapshot Readiness

- Snapshot rows: 23
- Unique students: 8
- Positive labels: 22
- Negative labels: 1
- Contract: 28-day lookback, 14-day future horizon, at least 3 future observations, at-risk when future present attendance is below 85%.

This profile is descriptive only. It does not establish that the dataset is sufficient for production training. Review class balance, temporal coverage, data freshness, and student-level train/test separation before model experimentation.
