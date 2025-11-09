# GraphMERT Documentation

This folder contains organized documentation for the GraphMERT project.

## Quick Links

**Start here:** [../README.md](../README.md) - Main project README
**Setup:** [../INSTALL.md](../INSTALL.md) - Installation instructions
**Getting Started:** [../GETTING_STARTED.md](../GETTING_STARTED.md) - Quick start guide

## Documentation Structure

### 📊 Data Processing (`data/`)

Documentation for data extraction, formatting, and triple generation:

- [DATA_DOCUMENTATION_INDEX.md](data/DATA_DOCUMENTATION_INDEX.md) - Main index for data docs
- [DATA_FORMAT_QUICK_REFERENCE.md](data/DATA_FORMAT_QUICK_REFERENCE.md) - Quick reference for data formats
- [DATA_STRUCTURE_VISUALIZATION.md](data/DATA_STRUCTURE_VISUALIZATION.md) - Visual guide to data structures
- [README_DATA_FORMAT.md](data/README_DATA_FORMAT.md) - Detailed data format specifications
- [TRAINING_DATA_ANALYSIS.md](data/TRAINING_DATA_ANALYSIS.md) - Analysis of training data
- [EXTRACTION_GUIDE.md](data/EXTRACTION_GUIDE.md) - Guide for extracting code triples
- [EXTRACTOR_GUIDE.md](data/EXTRACTOR_GUIDE.md) - Triple extractor documentation

### ☁️ Deployment (`deployment/`)

Guides for deploying and training on cloud platforms:

- [LAMBDA_README.md](deployment/LAMBDA_README.md) - Lambda Labs deployment overview
- [LAMBDA_BARE_BONES_SETUP.md](deployment/LAMBDA_BARE_BONES_SETUP.md) - Minimal Lambda setup guide
- [QUICK_START_LAMBDA.md](deployment/QUICK_START_LAMBDA.md) - Quick start for Lambda training

### 🔗 Chain Graphs (`chain-graphs/`)

Documentation for the chain graph builder system:

- [QUICK_REFERENCE.md](chain-graphs/QUICK_REFERENCE.md) - Command reference for chain graph builder
- [USAGE_EXAMPLES.md](chain-graphs/USAGE_EXAMPLES.md) - Detailed usage examples and tutorials

## Core Documentation (Root)

These files remain in the project root for easy access:

- [../README.md](../README.md) - Main project README
- [../GETTING_STARTED.md](../GETTING_STARTED.md) - Getting started guide
- [../INSTALL.md](../INSTALL.md) - Installation instructions
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - Architecture overview
- [../TESTING.md](../TESTING.md) - Testing guide
- [../PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - Project summary

## Archive

Historical documentation and completed work summaries are in `../archive/`:

- Implementation summaries
- Refactoring progress notes
- Comparison documents
- Session notes
- Migration reports

## Contributing

When adding new documentation:

1. Place it in the appropriate subfolder (`data/`, `deployment/`, or `chain-graphs/`)
2. Update this index file
3. Add cross-references where relevant
4. Keep the root directory clean - only essential docs at root level
