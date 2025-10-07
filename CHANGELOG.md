# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

## [0.2.0] - 2025-10-07

### Fixed

- `RDKit` backend raising `NameError` because `Mol` object was not defined if `rdkit` was not installed. With deferred evaluation it now correctly raises the `ModuleNotFoundError` when top-level functions try to use `rdkit` if it's not installed. [#6](https://github.com/coltonbh/qcinf/pull/6)

### Changed

- 🚨 Dropped python 3.9 support. Minimum supported version is now 3.10.

### Added

- Python 3.10-3.14 test matrix for GitHub Workflows.

## [0.1.1] - 2025-06-01

### Added

- `filter_conformers_indices` function to better serve the existing requirements of `qcio.view` setup.

## [0.1.0] - 2025-06-01

### Added

- Setup all DevOps workflows and basic package setup.
- Copied over all cheminformatics functions (e.g., `rmsd`, `align`, `filter_conformers` (formerly `ConformerSearchResults.conformers_filtered()`), `Structure.from_smiles()`, `Structure.to_smiles()`, etc.) from `qcio` into this repo.

[unreleased]: https://github.com/coltonbh/qcinf/compare/0.2.0...HEAD
[0.2.0]: https://github.com/coltonbh/qcinf/releases/tag/0.2.0
[0.1.1]: https://github.com/coltonbh/qcinf/releases/tag/0.1.1
[0.1.0]: https://github.com/coltonbh/qcinf/releases/tag/0.1.0
