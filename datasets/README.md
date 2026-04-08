# Dataset Placeholder

Place your raw datasets here following the structure documented in `README.md`.

```
datasets/
├── CWRU/
│   └── 12k_DE/
│       ├── 0hp/
│       ├── 1hp/
│       ├── 2hp/
│       └── 3hp/
├── MFPT/
│   ├── 1 - Three Baseline Conditions/
│   ├── 2 - Three Outer Race Fault Conditions/
│   ├── 3 - Seven More Outer Race Fault Conditions/
│   └── 4 - Seven Inner Race Fault Conditions/
├── JNU/
│   └── *.csv  (e.g., n600_1.csv, ib800_2.csv)
└── PU/
    ├── K001/
    ├── KA01/
    ├── KI01/
    └── ...
```

Processed caches will be saved to `datasets/cache/` automatically on first run.
