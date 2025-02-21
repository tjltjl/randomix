# Changelog

## [0.1.1] - 2025-02-21

### Fixed
- Correctly calculate `num_keys` in `Keyer.__call__` method.
- Convert `shape` to JAX array in `Keyer.__call__` method.
- Remove unnecessary `ravel` in tests.
- Update `testpaths` configuration in `pyproject.toml`.
- Ensure `__call__` returns a JAX array with specified shape in `Keyer`.
- Add documentation to `Keyer` class and `__call__` method.
- Add GitHub CI flow.
- Add section on new typed keys in `README.md`.

## [0.1.0] - Initial Release

- Initial implementation of `Keyer` class.
- Basic tests for `Keyer` class.
- Documentation setup.
