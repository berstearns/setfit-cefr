# `data/`

This directory is intentionally empty.

Place (or symlink) the following 5 CSVs here. Each CSV must have columns:
`writing_id,l1,cefr_level,text`.

| Filename                       | Role                    |
|--------------------------------|-------------------------|
| `norm-EFCAMDAT-train.csv`      | training data           |
| `norm-EFCAMDAT-remainder.csv`  | training data (extra)   |
| `norm-EFCAMDAT-test.csv`       | external test set       |
| `norm-KUPA-KEYS.csv`           | external test set       |
| `norm-CELVA-SP.csv`            | external test set       |

Automated setup:

```bash
export DATA_SRC=/path/to/your/csvs
just link-data
```
