# AGENTS.md

## General philosophy

This is a research codebase, not an enterprise software project.

The main priorities are:

1. Readability
2. Correctness
3. Clear code structure
4. Ease of modification during experiments and modularity for further experiments.
5. Performance where it matters

Prefer code that is easy to understand and change over code that is maximally generic or abstract.

Do not over-engineer solutions.

A straightforward implementation is usually preferable to introducing a new abstraction layer, framework, registry, factory, inheritance hierarchy, or configuration system.

## Code structure

Keep the structure of the code clear and easy to follow.

* Prefer explicit data flow.
* Prefer simple functions and small modules.
* Split code into separate files when this makes responsibilities clearer.
* Avoid very large files when functionality can naturally be separated.
* Avoid excessive fragmentation into many tiny files with little value.
* Keep related functionality together.
* Use descriptive names for functions, classes, and variables.
* Keep control flow simple when possible.
* Avoid deeply nested logic.
* Avoid unnecessary indirection.

When adding new functionality, first inspect how the surrounding code is organized and follow the existing structure when it is reasonable.

Do not rewrite working code simply to make it fit a new abstraction.

## Abstractions

Use abstractions only when they make the code easier to understand.

Avoid:

* unnecessary base classes;
* generic factories for a small number of cases;
* registries when a simple mapping or explicit conditional is clearer;
* highly configurable classes that mix several responsibilities;
* wrappers that only hide one simple function call;
* abstractions introduced for hypothetical future use.

If two implementations are slightly different, some small amount of duplication is acceptable if it keeps both implementations clear.

Prefer a few readable classes or functions over one complicated generic system.

## Error handling and input assumptions

Do not write defensive code for every hypothetical invalid input.

This codebase is primarily used by developers and researchers who control the inputs. Prefer clear assumptions and fail-fast behavior over large amounts of fallback logic.

In particular:

* Do not surround normal code paths with broad `try/except` blocks.
* Do not catch exceptions unless there is a concrete, expected failure mode that can be handled meaningfully.
* Never use `except Exception` just to keep execution going.
* Do not silently fall back to another behavior when an input is invalid.
* Do not add chains of `if/elif/else` to support undocumented or hypothetical input formats.
* Do not try to automatically guess what the caller intended.
* Prefer one clearly documented input format over supporting many loosely equivalent ones.
* If an argument is required, require it.
* If a tensor is expected to have a specific shape, assume or assert that shape rather than adding many branches to reinterpret it.
* If a configuration is invalid, raise an informative error instead of inventing a fallback.
* Let Python or PyTorch errors propagate when they already clearly identify a programming mistake.

Validation is useful when it catches a likely mistake early and makes the error substantially clearer. It should not dominate the implementation.

Prefer:

```python
assert audio.ndim == 3, "Expected audio with shape [B, C, T]"
```

over logic that tries to reinterpret several unrelated shapes automatically.

Prefer:

```python
if mode not in {"fast", "slow"}:
    raise ValueError(f"Unknown mode: {mode}")
```

over silently selecting a default mode.

Error handling should make the intended code path clearer, not make every possible misuse work.

## Research-code considerations

The code should remain easy to modify during experiments.

In particular:

* Make tensor shapes clear when they are not obvious.
* Keep device transfers explicit.
* Keep important preprocessing and sampling behavior explicit.
* Avoid hidden state and surprising side effects.
* Avoid silently changing existing behavior.
* Preserve existing APIs and semantics unless the task explicitly requires a change.
* Do not add dependencies unless they provide a clear benefit.

Comments should explain non-obvious decisions or constraints rather than restating the code.

## Scope of changes

Keep changes focused on the requested task.

* Do not refactor unrelated code.
* Do not rename unrelated functions or variables.
* Do not reformat entire files unnecessarily.
* Do not introduce compatibility layers for behavior that does not need to be preserved.
* Remove obsolete code when the change clearly replaces it, rather than keeping multiple unused paths "just in case".

If a requested change conflicts substantially with the existing architecture, prefer the smallest clear change rather than redesigning the whole subsystem.

## Testing philosophy

Testing should be proportional to the change.

Do not test everything in the repository after every small modification.

For simple changes:

* Run the smallest relevant test or sanity check.
* Do not run large integration suites unnecessarily.

For tricky implementations:

* Test the important functions or components individually while implementing them.
* Use small synthetic inputs when possible.
* Check shapes, values, edge cases, and expected failure modes locally.

For substantial changes:

* After component-level testing, perform one end-to-end test of the affected pipeline.
* The final test should verify that the changed components work together.

Avoid expensive tests unless they are specifically needed to validate the behavior.

In particular:

* Do not run full-scale training jobs as a routine test.
* Do not run large CUDA benchmarks just to check correctness.
* Do not test with 100 batches when one or a few small batches are sufficient.
* Prefer tiny batch sizes and short inputs for CUDA sanity checks.
* Do not occupy GPUs unnecessarily.

Performance benchmarking should only be done when performance is relevant to the requested change.

## Test environment

Use the existing `after` Conda environment for testing.

For example:

```bash
conda run -n after python ...
```

or, when working interactively:

```bash
conda activate after
```

Do not create a new environment unless the task explicitly requires it.

Before installing or changing dependencies, inspect the existing `after` environment and project requirements first.

## Before finishing a change

Before considering a task complete:

* Review the changed code for readability.
* Remove unnecessary complexity introduced during implementation.
* Check that responsibilities are clearly separated.
* Run the smallest meaningful tests for the modified components.
* For substantial changes, run one small end-to-end test of the affected pipeline.
* Report what was tested and any important limitations.
