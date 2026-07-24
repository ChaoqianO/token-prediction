# Stage 4 completion development summary

- Development artifacts: 4
- Final holdout (summary-reader scope): final artifacts, source data, and final labels were not opened by this reader; development results attest final target values were not used for fit, calibration, or scoring.
- Upstream loader audit: development artifact production parsed mixed raw source payloads before applying the task filter.

## Call-pre MLP vs LightGBM

| Source | Condition | Target | Per-seed paired MAE result |
|---|---|---|---|
| bagen_sokoban | `condition:effa60eb1d4380d124bf` | `call_billable_output_tokens` | 20260719: delta=-3.276, CI=[-10.792,4.148], inconclusive<br>20260720: delta=-5.138, CI=[-11.456,1.083], inconclusive<br>20260721: delta=-5.032, CI=[-11.010,0.985], inconclusive |
| bagen_sokoban | `condition:effa60eb1d4380d124bf` | `call_billable_total_tokens` | 20260719: delta=505.742, CI=[447.520,562.825], reference_supported<br>20260720: delta=544.275, CI=[481.948,607.075], reference_supported<br>20260721: delta=537.665, CI=[477.643,596.998], reference_supported |
| bagen_sokoban | `condition:effa60eb1d4380d124bf` | `call_final_response_output_tokens` | 20260719: delta=-3.276, CI=[-10.747,4.231], inconclusive<br>20260720: delta=-5.138, CI=[-11.375,0.789], inconclusive<br>20260721: delta=-5.032, CI=[-10.920,0.889], inconclusive |
| bagen_swebench | `condition:54cb50fce273f0aa2d74` | `call_billable_output_tokens` | 20260719: delta=11.675, CI=[1.157,30.155], reference_supported<br>20260720: delta=7.719, CI=[1.266,18.928], reference_supported<br>20260721: delta=7.221, CI=[1.174,17.777], reference_supported |
| bagen_swebench | `condition:54cb50fce273f0aa2d74` | `call_billable_total_tokens` | 20260719: delta=11144.234, CI=[9211.140,13187.552], reference_supported<br>20260720: delta=11326.960, CI=[9361.809,13439.561], reference_supported<br>20260721: delta=11263.143, CI=[9276.946,13507.974], reference_supported |
| bagen_swebench | `condition:54cb50fce273f0aa2d74` | `call_final_response_output_tokens` | 20260719: delta=11.675, CI=[1.063,30.673], reference_supported<br>20260720: delta=7.719, CI=[1.265,18.733], reference_supported<br>20260721: delta=7.221, CI=[1.176,17.632], reference_supported |
| bagen_swebench | `condition:949ac3b7a342718cd505` | `call_billable_output_tokens` | 20260719: delta=1.498, CI=[-0.983,4.189], inconclusive<br>20260720: delta=1.863, CI=[0.216,3.639], reference_supported<br>20260721: delta=0.412, CI=[-1.364,2.139], inconclusive |
| bagen_swebench | `condition:949ac3b7a342718cd505` | `call_billable_total_tokens` | 20260719: delta=4280.137, CI=[3690.019,4894.648], reference_supported<br>20260720: delta=4369.678, CI=[3792.619,4980.105], reference_supported<br>20260721: delta=4235.646, CI=[3678.892,4813.433], reference_supported |
| bagen_swebench | `condition:949ac3b7a342718cd505` | `call_final_response_output_tokens` | 20260719: delta=1.498, CI=[-1.032,4.250], inconclusive<br>20260720: delta=1.863, CI=[0.183,3.651], reference_supported<br>20260721: delta=0.412, CI=[-1.357,2.109], inconclusive |
| bagen_swebench | `condition:d94078c05d91b0d58aee` | `call_billable_output_tokens` | 20260719: delta=8.147, CI=[1.715,14.981], reference_supported<br>20260720: delta=7.255, CI=[2.018,13.494], reference_supported<br>20260721: delta=10.362, CI=[4.384,17.196], reference_supported |
| bagen_swebench | `condition:d94078c05d91b0d58aee` | `call_billable_total_tokens` | 20260719: delta=4489.744, CI=[3587.710,5450.997], reference_supported<br>20260720: delta=4744.481, CI=[3826.816,5775.013], reference_supported<br>20260721: delta=4481.456, CI=[3582.320,5435.671], reference_supported |
| bagen_swebench | `condition:d94078c05d91b0d58aee` | `call_final_response_output_tokens` | 20260719: delta=8.147, CI=[1.679,14.962], reference_supported<br>20260720: delta=7.255, CI=[1.888,13.361], reference_supported<br>20260721: delta=10.362, CI=[4.302,17.228], reference_supported |
| bagen_swebench | `condition:dce86ced00dc11c77205` | `call_billable_output_tokens` | 20260719: delta=4.508, CI=[1.002,8.634], reference_supported<br>20260720: delta=5.433, CI=[2.200,9.749], reference_supported<br>20260721: delta=4.976, CI=[0.676,10.113], reference_supported |
| bagen_swebench | `condition:dce86ced00dc11c77205` | `call_billable_total_tokens` | 20260719: delta=5346.932, CI=[4076.669,6737.382], reference_supported<br>20260720: delta=5069.653, CI=[3618.902,6595.460], reference_supported<br>20260721: delta=4756.047, CI=[3598.833,6045.767], reference_supported |
| bagen_swebench | `condition:dce86ced00dc11c77205` | `call_final_response_output_tokens` | 20260719: delta=4.508, CI=[1.074,8.638], reference_supported<br>20260720: delta=5.433, CI=[2.172,9.630], reference_supported<br>20260721: delta=4.976, CI=[0.761,10.169], reference_supported |
| bagen_swebench | `condition:f95ae2a5e11682f6b7fc` | `call_billable_output_tokens` | 20260719: delta=2.353, CI=[0.075,4.681], reference_supported<br>20260720: delta=9.167, CI=[4.075,15.480], reference_supported<br>20260721: delta=0.987, CI=[-1.906,3.868], inconclusive |
| bagen_swebench | `condition:f95ae2a5e11682f6b7fc` | `call_billable_total_tokens` | 20260719: delta=3753.144, CI=[2958.507,4619.310], reference_supported<br>20260720: delta=4206.071, CI=[3211.608,5351.622], reference_supported<br>20260721: delta=3679.909, CI=[2923.475,4473.906], reference_supported |
| bagen_swebench | `condition:f95ae2a5e11682f6b7fc` | `call_final_response_output_tokens` | 20260719: delta=2.353, CI=[0.043,4.648], reference_supported<br>20260720: delta=9.167, CI=[4.111,15.715], reference_supported<br>20260721: delta=0.987, CI=[-1.838,3.903], inconclusive |
| spend_openhands | `condition:b407e0d1ec34f386ebc4` | `call_billable_output_tokens` | 20260719: delta=30.298, CI=[28.787,31.769], reference_supported<br>20260720: delta=29.400, CI=[27.917,30.893], reference_supported<br>20260721: delta=31.007, CI=[29.528,32.505], reference_supported |
| spend_openhands | `condition:b407e0d1ec34f386ebc4` | `call_billable_total_tokens` | 20260719: delta=21856.341, CI=[21132.686,22606.883], reference_supported<br>20260720: delta=21806.835, CI=[21071.072,22582.492], reference_supported<br>20260721: delta=21909.542, CI=[21169.865,22676.443], reference_supported |
| spend_openhands | `condition:b407e0d1ec34f386ebc4` | `call_final_response_output_tokens` | 20260719: delta=30.298, CI=[28.832,31.806], reference_supported<br>20260720: delta=29.400, CI=[27.898,30.909], reference_supported<br>20260721: delta=31.007, CI=[29.485,32.531], reference_supported |

## Seed policy

Coverage guard: equal-weight `task_simultaneous_coverage`; a task is covered only when all of its positive-weight points are covered.

| Source | Condition | Per-seed WIS + task-coverage rule | Prospective rule |
|---|---|---|---|
| bagen_sokoban | `condition:effa60eb1d4380d124bf` | 20260719: WIS delta=-725.048, CI=[-2050.360,861.678], task simultaneous coverage delta=-0.010, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-218.754, CI=[-1096.580,844.221], task simultaneous coverage delta=-0.051, fail; MAE delta=0.000 (parity only)<br>20260721: WIS delta=1000.142, CI=[-780.018,3027.156], task simultaneous coverage delta=-0.071, fail; MAE delta=0.000 (parity only) | retain reference |
| bagen_swebench | `condition:54cb50fce273f0aa2d74` | 20260719: WIS delta=-516852.551, CI=[-887967.937,-228971.877], task simultaneous coverage delta=0.091, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-384203.723, CI=[-662216.701,-173212.755], task simultaneous coverage delta=0.000, pass; MAE delta=0.000 (parity only)<br>20260721: WIS delta=-479806.536, CI=[-1027690.323,-78669.831], task simultaneous coverage delta=-0.045, fail; MAE delta=0.000 (parity only) | retain reference |
| bagen_swebench | `condition:949ac3b7a342718cd505` | 20260719: WIS delta=12022.884, CI=[-15761.308,45442.336], task simultaneous coverage delta=-0.022, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=21.579, CI=[-33399.814,34394.459], task simultaneous coverage delta=-0.089, fail; MAE delta=0.000 (parity only)<br>20260721: WIS delta=38676.789, CI=[-33844.375,128804.018], task simultaneous coverage delta=-0.089, fail; MAE delta=0.000 (parity only) | retain reference |
| bagen_swebench | `condition:d94078c05d91b0d58aee` | 20260719: WIS delta=-20545.100, CI=[-78397.207,36787.024], task simultaneous coverage delta=-0.023, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-89174.195, CI=[-174572.912,-35189.301], task simultaneous coverage delta=0.047, fail; MAE delta=0.000 (parity only)<br>20260721: WIS delta=-31487.581, CI=[-56642.203,-10469.781], task simultaneous coverage delta=-0.047, fail; MAE delta=0.000 (parity only) | retain reference |
| bagen_swebench | `condition:dce86ced00dc11c77205` | 20260719: WIS delta=-284422.914, CI=[-463873.792,-143927.917], task simultaneous coverage delta=0.023, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-155307.368, CI=[-425472.894,120465.925], task simultaneous coverage delta=-0.068, fail; MAE delta=0.000 (parity only)<br>20260721: WIS delta=-277266.400, CI=[-439284.799,-148699.225], task simultaneous coverage delta=0.023, fail; MAE delta=0.000 (parity only) | retain reference |
| bagen_swebench | `condition:f95ae2a5e11682f6b7fc` | 20260719: WIS delta=-89914.555, CI=[-151356.345,-18044.708], task simultaneous coverage delta=-0.116, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-140234.747, CI=[-224159.334,-70836.249], task simultaneous coverage delta=-0.070, fail; MAE delta=0.000 (parity only)<br>20260721: WIS delta=-53121.228, CI=[-130233.376,40031.353], task simultaneous coverage delta=-0.116, fail; MAE delta=0.000 (parity only) | retain reference |
| spend_openhands | `condition:b407e0d1ec34f386ebc4` | 20260719: WIS delta=-469278.485, CI=[-612415.753,-338877.129], task simultaneous coverage delta=-0.010, fail; MAE delta=0.000 (parity only)<br>20260720: WIS delta=-489443.013, CI=[-631047.822,-362362.895], task simultaneous coverage delta=0.000, pass; MAE delta=0.000 (parity only)<br>20260721: WIS delta=-532384.017, CI=[-731261.743,-359536.406], task simultaneous coverage delta=0.003, pass; MAE delta=0.000 (parity only) | retain reference |

Prospective replacement rule: `retain_raw_repaired_reference_for_prospective_runs` (0/7 required conditions passed); the parent final selection is unchanged.

## Metric-field coverage

| Group | Complete / expected | Status |
|---|---:|---|
| Interval/reserve | 477 / 477 | complete |
| Repeated-run dispersion (lifecycle only) | 0 / 42 | declared unavailable |

