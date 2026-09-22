# Build notes

This repository was designed around the SALVO AI Club first-year requirement: learn one fundamental ML/DL concept and demonstrate it through a working implementation.

The main concept is Random Forest Regression.

Important design choices:

- No feature scaling because Random Forest decision trees do not need it.
- Engine-wise validation split to avoid putting the same engine trajectory in train and validation.
- Raw NASA data is downloaded at runtime rather than committed.
- Test benchmark metrics are calculated on each engine's final observed cycle.
- The Streamlit threshold labels are educational UI categories, not operational aviation limits.
