# Prospective task-information ceiling and retention benchmark

Frozen before new outcomes, 8 October 2026. Scientific contract: [config/ceiling_benchmark.json](config/ceiling_benchmark.json). This implements [INFORMATION_CEILING_SCOPE.md](INFORMATION_CEILING_SCOPE.md) without changing any earlier experiment or failed method gate.

## Primary question and estimands

At a fixed 64-feature budget, can development-only information calculations predict which spatial/temporal allocation preserves a task's information on independent acquisition/source conditions? Report prediction error, within-candidate selection regret, retained bits and Bayes risk. The benchmark is conditional on the declared balanced binary target, 250-ms observation, source law, noise and available metadata. It supplies a prerequisite for model design; no pretrained foundation model is evaluated.

Sensor information and transformed information use the same hidden-state prior and zero subject calibration. Known-head/source/background oracles are separate optimistic references. Fitted models see the same observation or representation as its Bayes reference, but estimate parameters from development examples; the gap includes distribution estimation and domain shift. Accuracy does not estimate representation MI.

All integration is conditional on declared finite banks. Pointwise bounded-draw intervals describe numerical integration uncertainty. Cross-head and cross-bank variation is separate. No continuous anatomy population coverage is claimed. Existing population certificates remain attributed to their original experiments, not transferred into this richer model.

## Fixed simulation design

- Six positive, non-antipodal targets: three spatial patch distinctions and three temporal event distinctions, with no designed target-removal basis.
- 64 samples at256Hz. Source peak20nAm;8nAm is a known-state acquisition sensitivity. Both class means include persistent subject background. Independent epoch Gaussian source background is propagated through each state's lead field, producing a head-dependent rank-one covariance update.
- Four sensor-noise regimes cross spatial correlation0/.55 and AR(1) color0/.75, always6µV marginal SD. Spatial and temporal covariance are pushed through every physical transform.
- Two fresh spherical banks and two anatomical BEM banks (fsaverage and the public sample subject), each8 development,4 validation and24 check operators. Conductivity and mesh resolution are declared. Subject/cap/source/static-background states stay fixed across their observations. Check source states are independent of development and include an explicit broader shift domain.
- Nested19/64/native sensor coverage is fixed by nominal coverage geometry, before targets are evaluated. BEM uses the same64 named contacts and matched ico3 discretization across its two anatomies; resolution sensitivity is retained separately. Two geometries do not establish human-population generalization.
- Representation comparisons use the fixed19-channel montage, then18 independent CAR coordinates. Budgets32/64/128 compare four spatial/time allocations each, and three development-only spatial designs: generic multitask PCA, task-matched PCA, and noise-aware task Fisher directions. Temporal features are ordinary contiguous arithmetic block averages in raw time coordinates. A feature means one independent scalar output, not an entire channel-time patch token.

## Predictions and decisions

Before check evaluation, save each candidate's matrices, development MI/loss prediction, nominal known-state Gaussian prediction, variance-retention score and prospective selections. Generic/task PCA designs use the declared development second moment; Fisher directions use development task contrasts and the declared noise/background model. These are simulator-informed design priors, not priors estimated from real EEG.

Implementation definitions fixed before outcomes: generic PCA centres the pooled task mixture once, including between-task mean variation. The Fisher encoder is an approximate separable spatial design: after temporal whitening, average the rank-one background's spatial covariance, whiten spatial task contrasts with that effective covariance, then retain leading spatial directions. It is not an exact optimum of the full hidden-state experiment. The nominal-information comparison is explicitly a development mean-prototype Gaussian with averaged means/background, not a revealed nominal-head operator.

Primary prediction is development finite-bank MI loss. Mean absolute prediction error≤.05bits and mean selection regret≤.02bits are prospective descriptive usefulness criteria. Show both check domains and each anatomy/bank individually; success of a grid mean is not population certification. Compare development-information selection with fixed4-spatial×16-time allocation, variance-retention selection, and nominal-information selection. Include strong task-matched designs. Selection regret is relative to the frozen candidate pool, not every conceivable encoder.

For pure anatomy holdout, fsaverage-fitted19-contact matrices and predictions are evaluated on sample check states, with no sample states used to change them. Sample-local developmental designs are separately labelled as a stronger anatomy-informed baseline. No sample check labels or means are used for fitting or selection.

## Learning and precision checks

Two prespecified tasks per bank in joint noise compare full observations, fixed generic PCA, predicted task PCA and predicted Fisher representations. Ridge LDA and a24-unit MLP use development-only normalization, independent validation for selection and untouched check epochs. Both ordinary and shifted check states are retained. Persist all selected model parameters and per-head risk/log-loss results. Estimate accessible task information only as a declared clipped-predictor cross-entropy lower bound, not as exact MI; report the oracle-law advantage.

Independent high-precision65536-draw runs cover the two decoder tasks at64 features in joint noise on ordinary check states for every bank. They diagnose Monte Carlo error without changing a selection. Expected data-processing/risk inequalities are checked with integration uncertainty, not by demanding every noisy point estimate be monotone.

## Verification and outputs

Mathematical audit and independent toys must verify arbitrary class midpoints, full affine-mean/background sufficiency, raw block-average covariance including between-block color, reference rank and invertible-coordinate invariance. Numerical code tests cover invalid/rank-deficient inputs and paired estimates. Physical assets must pass hashes, units, source orientation and contact-surface checks; seed/split separation is mandatory.

Freeze protocol/config/source hashes before the main run. Log any implementation repairs, retain earlier outputs and never tune scientific parameters to results. Run all existing/new tests. An independent verifier recomputes transform covariance, known-state separations, prospective choices, prediction/selection scores, check count and artifact hashes. Save machine-readable tables, frozen transforms, numerical checks, a complete report and four readable figures: sensor-information map; spatial/temporal budget retention; prediction/held-out-anatomy validation; and representation-versus-learning gaps.

The prior scope clarification was documentation-only. This protocol starts a distinct prospective simulation study with new outputs; its outcomes may support or reject the proposed design predictions.
