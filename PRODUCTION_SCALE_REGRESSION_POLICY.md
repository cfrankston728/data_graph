# DataGraph Production-Scale Regression Policy

Resolved: 2026-08-28  
Governing benchmark established by: DG-OPT-266 through DG-OPT-276

## Governing production-scale fixture

Use the exact historical DataGraph fixture with:

- Nodes: **2,242,200**
- Governing neighborhood size: **k=200**
- Expected CSR nnz for semantics-preserving changes: **542,487,708**
- Expected connected components: **1**
- Current reference build wall time: **518.703 s (8.645 min)**
- Current reference peak RSS: **64.292 GiB**
- Standard benchmark allocation: **32 CPUs**
- Thread configuration: NUMBA / OMP / MKL / OPENBLAS = allocated CPU count

The historical graph directory currently ends in `k400`, but this label is
misleading. A current k=200 run reproduced its CSR nnz exactly on the exact
historical feature fixture. A correction note is stored inside that graph
directory.

## Performance thresholds

For a comparable k=200 production-scale run:

- **Reference:** approximately 8.65 minutes.
- **Warning threshold:** build wall time > **12 minutes**.
- **Hard performance failure:** build wall time > **15 minutes**.
- **Process kill cap:** **20 minutes** from benchmark-process start.

Queue wait is not part of the cap.

A run exceeding 15 minutes on comparable hardware is treated as a regression
until explained. A run reaching the 20-minute process cap should be terminated
rather than consuming the full scheduler wall allocation.

If hardware, software environment, CPU allocation, filesystem conditions, or
fixture semantics materially differ, classify the result as non-comparable
before declaring a product regression.

## Structural validity

For changes intended to preserve graph semantics, the large-scale run should
also preserve:

- graph shape: 2,242,200 x 2,242,200
- connected components: 1
- CSR nnz: 542,487,708

An intentional algorithmic/topological change may legitimately change nnz,
but that difference must be explicitly identified as intended rather than
silently accepted as a performance-only optimization.

## Benchmark cadence

Run the production-scale k=200 benchmark:

1. **Immediately after any structural change** affecting graph traversal,
   batching, memory layout, parallelism, KNN construction, MST construction,
   pruning, polishing, refinement, or graph representation.
2. **After at most three accepted micro/JIT/cache optimizations** without an
   intervening production-scale checkpoint.
3. **Before declaring an optimization phase complete.**
4. Whenever a smaller fixture suggests a large speedup or regression that could
   plausibly behave differently at multi-million-node scale.

The 7,200-node fixture remains useful for mechanism isolation, but it is not a
substitute for this production-scale checkpoint.

## Failure and rollback protocol

If a candidate breaches the hard production-scale threshold:

1. Freeze further live optimization changes.
2. Preserve the failed candidate, benchmark result, source hashes, environment,
   timing breakdown, and rollback path.
3. Compare against the last known-good production-scale source state.
4. If one unvalidated change is responsible, test its rollback directly.
5. If several developments accumulated, perform ablation in scratch copies:
   - for up to four plausible changes, use full combinatorial subset ablation
     when computationally practical;
   - for larger sets, first use grouped/binary or fractional ablation, then
     perform focused combinatorial subsets around the implicated changes.
6. Do not reinstall or retain a suspect combination merely because its
   small-fixture benchmark looked favorable.
7. Restore/promote only a combination that passes the governing large-scale
   checkpoint or whose changed semantics have been explicitly reviewed.

The purpose is to detect interactions between individually reasonable
optimizations rather than assuming regressions are attributable to the most
recent change.

## k=400 role

k=400 is **not** the governing historical regression configuration.

The completed k=400 stress run produced:

- wall time: **998.665 s (16.644 min)**
- peak RSS: **128.035 GiB**
- CSR nnz: **1,101,315,958**

Use k=400 intermittently as a higher-density stress/scaling test, or when a
scientific requirement justifies the larger neighborhood.

## Current optimization priority

On the resolved k=200 production run, the dominant stages were:

1. Step 1 — KNN + MST: ~331.1 s
2. Step 2 — refinement: ~109.4 s
3. Step 3 — analysis: ~77.8 s
4. graph-distance calculation: ~14.3 s

Optimization work should be reconciled against these real-scale costs rather
than prioritizing cold-JIT microbenchmarks in isolation.
