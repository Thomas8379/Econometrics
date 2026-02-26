# run-tests

Run the full econtools test suite.

```bash
cd C:/Econometrics && python -m pytest tests/ -v
```

To run only the data layer tests:

```bash
cd C:/Econometrics && python -m pytest tests/data/ -v
```

To run a specific test file:

```bash
cd C:/Econometrics && python -m pytest tests/data/test_io.py -v
```

To run with coverage:

```bash
cd C:/Econometrics && python -m pytest tests/ -v --cov=econtools --cov-report=term-missing
```
