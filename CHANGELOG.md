# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Changed

- Default `backend` for `rmsd` function changed from `rdkit` to `qcinf`.
- Default `backend` for `align` function changed from `rdkit` to `qcinf`.

## [0.1.1] - 2025-06-01

### Added

- `filter_conformers_indices` function to better serve the existing requirements of `qcio.view` setup.

## [0.1.0] - 2025-06-01

### Added

- Setup all DevOps workflows and basic package setup.
- Copied over all cheminformatics functions (e.g., `rmsd`, `align`, `filter_conformers` (formerly `ConformerSearchResults.conformers_filtered()`), `Structure.from_smiles()`, `Structure.to_smiles()`, etc.) from `qcio` into this repo.

[unreleased]: https://github.com/coltonbh/qcinf/compare/0.1.1...HEAD
[0.1.1]: https://github.com/coltonbh/qcinf/releases/tag/0.1.1
[0.1.0]: https://github.com/coltonbh/qcinf/releases/tag/0.1.0
