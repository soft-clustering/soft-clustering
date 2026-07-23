# Contributing to Soft Clustering

Thank you for considering contributing to **Soft Clustering**! 🎉

All contributions are welcome, including:

- Bug reports and fixes
- New features or algorithm implementations
- Documentation improvements
- Example code or tutorials
- Performance optimizations

## How to Contribute

1. **Fork** the repository and create a new branch for your changes.
2. Make your changes following the existing code style.
3. Write or update tests if applicable.
4. Ensure all tests pass (`pytest`).
5. Submit a **Pull Request** with a clear description of your changes.

## Development Setup

```bash
git clone https://github.com/soft-clustering/soft-clustering.git
cd soft-clustering
pip install -e ".[deep,dev]"   # or without [deep] if you don't need it
```

## Running Tests

The project has a comprehensive test suite located in the tests/ directory.

To run the tests:
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_fcm.py -v

# Run tests with coverage
pytest --cov=soft_clustering
```

See tests/HOW_TO_RUN.txt for additional details and tips.

Important notes:

- We use pytest as the test runner.
- Many tests use fixtures defined in tests/conftest.py.
- Some tests may require the [deep] extra (PyTorch) if they involve deep clustering models.

## Code Style

- We use black for code formatting.
- Please run black . before submitting a PR.
- Follow PEP 8 guidelines where possible.

## Reporting Bugs

Please open an issue with:

- A clear title
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment details (if relevant)


---


Happy coding! We look forward to your contributions.
