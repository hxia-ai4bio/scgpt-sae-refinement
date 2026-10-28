# Reproducibility

Run from the original task root with the interpreter in ENVIRONMENT.json:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=4
    export MPLCONFIGDIR="$PWD/work/sae_complementarity/mplcache"
    python work/sae_complementarity/run.py preflight
    python work/sae_complementarity/run.py develop
    python work/sae_complementarity/run.py lock
    python work/sae_complementarity/run.py evaluate
    python work/sae_complementarity/regularization_diagnostic.py
    python work/sae_complementarity/report.py
    python work/sae_complementarity/verify.py

Input paths/checksums are in frozen_sources.json. The original source datasets and 18 pretrained SAE checkpoint/cache sets are external prerequisites, not bundled. The full frozen downstream models, validation grids, crossfit targets, selections, locked evaluation predictions, scripts and reports are bundled. Re-running at a different root requires adjusting input roots in a NEW experiment directory and regenerating the preflight hash binding; never overwrite sealed results. No neural model is trained here.

Time: O(n*p^2+p^3) per dense Ridge decomposition, O(p^2*C) per subsequent alpha, p<=2560,C=8; memory O(n*p+p^2). Sequential CPU, float64 SciPy BLAS and LAPACK, avoids the previously observed NumPy @ warnings. Inner label-crossfit baseline work is shared across SAE seeds.
