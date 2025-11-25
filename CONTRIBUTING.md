# Contributing to vLLM vs TGI Benchmarking Study

Thank you for your interest in contributing to this benchmarking project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Benchmark Contributions](#benchmark-contributions)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and encourage diverse perspectives
- Focus on constructive feedback
- Prioritize the community and scientific integrity

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Publishing others' private information
- Cherry-picking data or manipulating results
- Other unprofessional conduct

## How Can I Contribute?

### Reporting Bugs

If you find a bug in the benchmarking code:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Environment details (GPU, CUDA version, framework versions)
   - Relevant logs or error messages

**Template:**
```markdown
**Bug Description:**
[Clear description]

**Steps to Reproduce:**
1. Run command: `python ...`
2. With config: `...`
3. Observe error: `...`

**Environment:**
- GPU: NVIDIA A100 80GB
- CUDA: 12.1
- vLLM: v0.6.1
- TGI: v2.3.0
- OS: Ubuntu 22.04

**Logs:**
[Paste relevant logs]
```

### Suggesting Enhancements

We welcome suggestions for:

- New metrics to measure
- Additional frameworks to benchmark (e.g., TensorRT-LLM, llama.cpp)
- New models to test
- Improved analysis methods
- Better visualizations

**Enhancement Template:**
```markdown
**Enhancement Title:**
[Brief description]

**Motivation:**
Why is this enhancement valuable?

**Proposed Solution:**
How would you implement it?

**Alternatives Considered:**
What other approaches did you think about?

**Impact:**
How does this improve the benchmarks?
```

### Improving Documentation

Documentation improvements are always welcome:

- Fix typos or clarify explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation (if applicable)

### Adding New Benchmarks

See [Benchmark Contributions](#benchmark-contributions) section below.

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/vllm-tgi-benchmark.git
cd vllm-tgi-benchmark

# Add upstream remote
git remote add upstream https://github.com/saicharan0810/vllm-tgi-benchmark.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks (if configured)
pre-commit install
```

### Verify Installation

```bash
# Run a quick test benchmark
python benchmarks/run_vllm_benchmark.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 10 \
    --concurrency 1
```

## Development Workflow

### Branch Naming Convention

- `feature/description`: New features or enhancements
- `bugfix/description`: Bug fixes
- `docs/description`: Documentation improvements
- `benchmark/model-name`: New benchmark for specific model
- `analysis/metric-name`: New analysis or metric

**Examples:**
- `feature/add-tensorrt-benchmark`
- `bugfix/fix-memory-leak`
- `docs/improve-readme`
- `benchmark/mistral-7b`

### Making Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Stage and commit
git add .
git commit -m "Add feature: description of changes"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(benchmark): add support for Mistral-7B model

- Add configuration for Mistral-7B
- Update benchmark runner to support new model
- Add results processing for Mistral architecture

Closes #42
```

```
fix(analysis): correct TPOT calculation for edge cases

The TPOT calculation was incorrect when generation length was 1.
Added check to avoid division by zero.

Fixes #67
```

## Benchmark Contributions

### Adding a New Model

1. **Create configuration** in `benchmarks/config.py`:
```python
MODEL_CONFIGS = {
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
        # ... other config
    }
}
```

2. **Add model-specific parameters** if needed
3. **Run benchmarks** following existing methodology
4. **Document results** in appropriate format
5. **Add visualization** if applicable

### Adding a New Framework

1. **Create new benchmark script**: `benchmarks/run_FRAMEWORK_benchmark.py`
2. **Follow existing structure**:
   - Initialization
   - Warm-up phase
   - Measurement phase
   - Metrics collection
   - Results output

3. **Update comparison script**: `analysis/compare_results.py`
4. **Document framework-specific configurations**
5. **Add to README** with installation instructions

### Adding New Metrics

1. **Define metric** in `benchmarks/utils.py`:
```python
def calculate_new_metric(data):
    """
    Calculate [metric name].

    Args:
        data: Raw benchmark data

    Returns:
        Computed metric value
    """
    # Implementation
    return metric_value
```

2. **Integrate into benchmark runner**
3. **Add to results output**
4. **Update analysis and visualization**
5. **Document in METHODOLOGY.md**

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Maximum line length: **100 characters**
- Use **docstrings** for all public functions/classes

**Example:**
```python
from typing import List, Dict, Optional

def calculate_throughput(
    total_tokens: int,
    elapsed_time: float,
    concurrency: int
) -> Dict[str, float]:
    """
    Calculate throughput metrics.

    Args:
        total_tokens: Total number of tokens generated
        elapsed_time: Wall-clock time in seconds
        concurrency: Number of concurrent requests

    Returns:
        Dictionary containing tokens/sec and requests/sec
    """
    tokens_per_sec = total_tokens / elapsed_time
    requests_per_sec = concurrency / elapsed_time

    return {
        "tokens_per_sec": tokens_per_sec,
        "requests_per_sec": requests_per_sec
    }
```

### Code Organization

- Keep functions focused and single-purpose
- Group related functionality in modules
- Use meaningful variable names
- Add comments for complex logic

### Error Handling

```python
# Good: Specific exception handling
try:
    result = run_benchmark(config)
except GPUOutOfMemoryError as e:
    logger.error(f"GPU OOM during benchmark: {e}")
    cleanup_gpu_memory()
    raise
except BenchmarkTimeoutError as e:
    logger.warning(f"Benchmark timeout: {e}")
    return partial_results
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=benchmarks tests/
```

### Writing Tests

```python
import pytest
from benchmarks.utils import calculate_throughput

def test_throughput_calculation():
    """Test basic throughput calculation."""
    result = calculate_throughput(
        total_tokens=1000,
        elapsed_time=10.0,
        concurrency=5
    )

    assert result["tokens_per_sec"] == 100.0
    assert result["requests_per_sec"] == 0.5

def test_throughput_zero_time():
    """Test throughput calculation with zero elapsed time."""
    with pytest.raises(ZeroDivisionError):
        calculate_throughput(
            total_tokens=1000,
            elapsed_time=0.0,
            concurrency=5
        )
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def run_benchmark(
    model: str,
    concurrency: int,
    num_prompts: int = 1000
) -> BenchmarkResults:
    """
    Run a complete benchmark for the specified model.

    This function executes a full benchmark including warm-up,
    measurement, and cool-down phases.

    Args:
        model: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        concurrency: Number of concurrent requests
        num_prompts: Total number of prompts to process (default: 1000)

    Returns:
        BenchmarkResults object containing all metrics

    Raises:
        GPUOutOfMemoryError: If GPU runs out of memory
        ModelNotFoundError: If model is not available

    Example:
        >>> results = run_benchmark(
        ...     model="meta-llama/Llama-2-7b-hf",
        ...     concurrency=50
        ... )
        >>> print(results.throughput)
        15243.5
    """
    # Implementation
```

### README Updates

When adding new features, update the README:

- Installation instructions (if dependencies change)
- Usage examples
- Configuration options
- Results summary (if benchmark added)

## Pull Request Process

### Before Submitting

1. **Sync with upstream:**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests:**
```bash
pytest tests/
```

3. **Check code style:**
```bash
# If using black and flake8
black benchmarks/ analysis/
flake8 benchmarks/ analysis/
```

4. **Update documentation** if needed

### PR Template

```markdown
## Description
[Brief description of changes]

## Motivation
Why is this change needed?

## Changes Made
- [ ] Added/modified benchmarks
- [ ] Updated analysis scripts
- [ ] Improved documentation
- [ ] Fixed bugs
- [ ] Added tests

## Testing
How were these changes tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commits are well-formatted
- [ ] No breaking changes (or documented)

## Related Issues
Closes #[issue_number]
```

### Review Process

1. Maintainer will review your PR
2. Address any feedback or requested changes
3. Once approved, PR will be merged
4. Your contribution will be acknowledged!

### After Merge

```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## Recognition

All contributors will be acknowledged in:
- README.md (Contributors section)
- Release notes
- Academic acknowledgments (for significant contributions)

## Questions?

- Open an issue for general questions
- Email: kscharan1608@gmail.com for private inquiries
- Discussion forum: [GitHub Discussions]

## License

By contributing, you agree that your contributions will be licensed under the same CC BY 4.0 license that covers this project.

---

Thank you for contributing to advancing LLM inference benchmarking! 🚀
