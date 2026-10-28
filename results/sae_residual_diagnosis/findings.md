# Research findings — frozen SAE residual diagnosis

- Experiment integrity: WARN; no unresolved material computation error found in independent bounded checks. Same-family/provisional.
- Result-to-claim: partial. Historical rejection stage and optimizer movement are established; exclusive cause, SAE intrinsic uselessness, and new generalization gains are not.
- Gate rejection funnel: epoch0 13/18, fresh Ridge 3/18, retained 2/18. Beta0 also selects epoch0 in 13/18 cases.
- In seed42 joint Gate replay, three folds contain a better probed Ridge VALID checkpoint before old stopping, but all comparisons reuse historically explored VALID labels.
- More training/frozen head yields small selected VALID maxima; learned decoder is not consistently better than rotation. Expanded learned and random TopK readouts both show selected gains under a more flexible classifier.
- Keep the current experiment immutable. Do not retune the same six VALID donors to claim generalization. Do not merge expanded-head diagnostics with the original residual's performance protocol.
- Recorded matmul warnings require preservation. All selected trajectory/identity readout selections were independently reproduced with dgemm/Cholesky; this bounded success is not certification of the environment for future experiments.
- Routing: finish this explanatory task with qualified claims. Further experiments are optional future work, not automatically authorized by the partial verdict. See CLAIMS_FROM_RESULTS.md for the precise evidence gap.
