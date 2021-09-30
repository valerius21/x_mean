# Practical Couse on Data Fusion Summer 2021


## Structure

```
.
├── CITATION.cff
├── data_fusion
│   ├── definitions.py # definitions and constants
│   ├── entities
│   │   ├── annotations.py # annotation objects / instances
│   │   └── Vehicle.py # Wrapper class for annotated, moving vehicles
│   ├── kalman
│   │   ├── basic_kalman.py # Kalman Filter Implementation
│   │   ├── extended_kalman.py # Extended Kalman Filter Implementation
│   │   └── uncented_kalman.py # Uncented Kalman Filter Implementation
│   └── utils
│       ├── binary_files.py # Helper Methods for parsing the binary data files
│       ├── data_parsing.py # Helper Methods to wrap the included data and access
│       ├── data.py # Wrapper for the binary files and frequently acessed data
│       ├── helpers.py # other helpers
│       └── Watcher.py # Plotting helpers
├── notebooks
│   └── prep.ipynb # main notebook
├── README.md
└── requirements.txt

```

