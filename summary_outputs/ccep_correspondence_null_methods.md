# CCEP validation-strengthening package

Generated: 2026-07-03T03:26:26.338221+00:00

## Data availability boundary

The current project does not contain same-subject resting-state fMRI and
individual stimulation-response recordings. The available CCEP comparison uses
aggregate F-TRACT summaries in the MNI-HCP-MMP1 parcellation and fixed
`reproduce_0866` EC matrices from six held-out HCP model subjects. Therefore the
analysis below strengthens transparency and null-model control for an already
weak external correspondence result; it is not same-subject perturbation
validation and is not biological ground truth.

## Inputs

- EC inputs: `validation_results/{subject}/reproduce_0866_{subject}/ec.npy`
- CCEP amplitude input: `data-all\CCEP(F-TRACT)\MNI-HCP-MMP1\amplitude__median.txt.gz`
- CCEP probability input: `data-all\CCEP(F-TRACT)\MNI-HCP-MMP1\probability.txt.gz`
- Region labels: `data-all\CCEP(F-TRACT)\MNI-HCP-MMP1\MNI-HCP-MMP1.txt`
- Centroid source: `data-all\HCP_S1200_Atlas_Z4_pkXDZ\Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii` with S1200
  midthickness surfaces

## Coverage

Amplitude CCEP coverage is 20807 of
129240 directed off-diagonal edges
(16.100%). Probability CCEP coverage is
51931 of
129240 directed off-diagonal edges
(40.182%).

Coverage figure: `ccep_validation_strengthening_package\figures\supp_ccep_coverage_strengthening.png`

## Null controls

The observed primary-rule group mean was
`r = 0.032162` across
6 held-out model subjects. Three null
controls were computed with 1000 permutations:

- Edge-value permutation: null mean `0.000149`, empirical
  two-sided `p = 0.0010`.
- Synchronous region-label permutation: null mean `-0.000331`,
  empirical two-sided `p = 0.0010`.
- Distance-bin-matched edge permutation: null mean `0.001480`,
  empirical two-sided `p = 0.0010`.

Observed-versus-null figure: `ccep_validation_strengthening_package\figures\supp_ccep_nulls_distance_matched.png`

## Interpretation rule

These controls show whether the low fixed CCEP correspondence is greater than
several random mappings, including a distance-bin-matched null. They do not
change the absolute effect size and must not be described as strong external
validation. The correct interpretation is weak but label/edge-specific
correspondence under a fixed post-exploratory CCEP rule.
