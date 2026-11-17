# Results (mean ± std over seeds; threshold chosen on validation)

Parameter count, single CNN: 2415926
Parameter count, learned fusion: 7246241

## iccad1 (6 seeds)

| Method | BA | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|
| orig | 0.825 ± 0.009 | 0.999 ± 0.002 | 0.122 ± 0.006 | 0.650 ± 0.018 | 0.217 ± 0.009 |
| edge | 0.831 ± 0.015 | 0.997 ± 0.002 | 0.126 ± 0.010 | 0.664 ± 0.030 | 0.224 ± 0.016 |
| dens | 0.838 ± 0.019 | 0.999 ± 0.002 | 0.131 ± 0.015 | 0.677 ± 0.040 | 0.232 ± 0.023 |
| pair_orig_edge | 0.829 ± 0.030 | 0.996 ± 0.011 | 0.128 ± 0.023 | 0.662 ± 0.059 | 0.226 ± 0.035 |
| pair_orig_dens | 0.829 ± 0.026 | 0.998 ± 0.004 | 0.126 ± 0.017 | 0.661 ± 0.050 | 0.224 ± 0.027 |
| pair_edge_dens | 0.847 ± 0.033 | 0.992 ± 0.013 | 0.146 ± 0.040 | 0.703 ± 0.074 | 0.253 ± 0.057 |
| concat3 | 0.824 ± 0.018 | 0.999 ± 0.004 | 0.122 ± 0.010 | 0.650 ± 0.033 | 0.217 ± 0.016 |
| learned_fusion | 0.832 ± 0.013 | 1.000 ± 0.000 | 0.126 ± 0.009 | 0.663 ± 0.027 | 0.224 ± 0.013 |
| late_fusion | 0.809 ± 0.009 | 1.000 ± 0.000 | 0.112 ± 0.004 | 0.618 ± 0.018 | 0.202 ± 0.007 |

Validation-selected single representation: BA 0.825 ± 0.009
- concat3 vs SEL: mean diff -0.0006, Wilcoxon exact two-sided p = 0.8438
- learned_fusion vs SEL: mean diff +0.0069, Wilcoxon exact two-sided p = 0.3125
- late_fusion vs SEL: mean diff -0.0160, Wilcoxon exact two-sided p = 0.0938

## iccad2 (6 seeds)

| Method | BA | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|
| orig | 0.974 ± 0.010 | 0.962 ± 0.024 | 0.457 ± 0.084 | 0.985 ± 0.005 | 0.615 ± 0.072 |
| edge | 0.950 ± 0.016 | 0.977 ± 0.026 | 0.155 ± 0.072 | 0.924 ± 0.031 | 0.262 ± 0.102 |
| dens | 0.943 ± 0.033 | 0.897 ± 0.071 | 0.520 ± 0.096 | 0.989 ± 0.005 | 0.650 ± 0.062 |
| pair_orig_edge | 0.969 ± 0.010 | 0.956 ± 0.028 | 0.434 ± 0.128 | 0.982 ± 0.011 | 0.585 ± 0.126 |
| pair_orig_dens | 0.955 ± 0.034 | 0.927 ± 0.074 | 0.427 ± 0.133 | 0.983 ± 0.009 | 0.572 ± 0.118 |
| pair_edge_dens | 0.957 ± 0.032 | 0.925 ± 0.068 | 0.497 ± 0.083 | 0.988 ± 0.004 | 0.640 ± 0.050 |
| concat3 | 0.957 ± 0.025 | 0.922 ± 0.053 | 0.596 ± 0.085 | 0.992 ± 0.003 | 0.719 ± 0.059 |
| learned_fusion | 0.960 ± 0.018 | 0.944 ± 0.018 | 0.424 ± 0.174 | 0.977 ± 0.025 | 0.566 ± 0.188 |
| late_fusion | 0.959 ± 0.030 | 0.933 ± 0.069 | 0.526 ± 0.213 | 0.985 ± 0.015 | 0.643 ± 0.175 |

Validation-selected single representation: BA 0.962 ± 0.015
- concat3 vs SEL: mean diff -0.0047, Wilcoxon exact two-sided p = 0.6875
- learned_fusion vs SEL: mean diff -0.0015, Wilcoxon exact two-sided p = 1.0000
- late_fusion vs SEL: mean diff -0.0026, Wilcoxon exact two-sided p = 0.4375

## iccad3 (6 seeds)

| Method | BA | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|
| orig | 0.962 ± 0.005 | 0.979 ± 0.003 | 0.417 ± 0.049 | 0.945 ± 0.011 | 0.583 ± 0.049 |
| edge | 0.959 ± 0.004 | 0.976 ± 0.007 | 0.400 ± 0.060 | 0.941 ± 0.014 | 0.565 ± 0.057 |
| dens | 0.961 ± 0.006 | 0.971 ± 0.011 | 0.457 ± 0.089 | 0.952 ± 0.021 | 0.617 ± 0.085 |
| pair_orig_edge | 0.956 ± 0.007 | 0.979 ± 0.003 | 0.372 ± 0.050 | 0.934 ± 0.016 | 0.537 ± 0.054 |
| pair_orig_dens | 0.955 ± 0.006 | 0.978 ± 0.003 | 0.363 ± 0.036 | 0.932 ± 0.011 | 0.529 ± 0.038 |
| pair_edge_dens | 0.953 ± 0.010 | 0.979 ± 0.002 | 0.352 ± 0.053 | 0.927 ± 0.021 | 0.515 ± 0.060 |
| concat3 | 0.957 ± 0.008 | 0.978 ± 0.005 | 0.381 ± 0.068 | 0.935 ± 0.019 | 0.546 ± 0.070 |
| learned_fusion | 0.958 ± 0.006 | 0.979 ± 0.002 | 0.385 ± 0.046 | 0.938 ± 0.013 | 0.551 ± 0.047 |
| late_fusion | 0.963 ± 0.005 | 0.978 ± 0.004 | 0.434 ± 0.069 | 0.948 ± 0.014 | 0.599 ± 0.064 |

Validation-selected single representation: BA 0.963 ± 0.004
- concat3 vs SEL: mean diff -0.0060, Wilcoxon exact two-sided p = 0.0312
- learned_fusion vs SEL: mean diff -0.0043, Wilcoxon exact two-sided p = 0.3125
- late_fusion vs SEL: mean diff +0.0006, Wilcoxon exact two-sided p = 0.6875

## iccad4 (6 seeds)

| Method | BA | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|
| orig | 0.907 ± 0.033 | 0.851 ± 0.079 | 0.140 ± 0.056 | 0.964 ± 0.023 | 0.236 ± 0.084 |
| edge | 0.911 ± 0.037 | 0.856 ± 0.095 | 0.188 ± 0.142 | 0.967 ± 0.022 | 0.282 ± 0.156 |
| dens | 0.912 ± 0.045 | 0.857 ± 0.103 | 0.170 ± 0.119 | 0.968 ± 0.016 | 0.262 ± 0.123 |
| pair_orig_edge | 0.913 ± 0.031 | 0.882 ± 0.072 | 0.165 ± 0.100 | 0.944 ± 0.070 | 0.264 ± 0.146 |
| pair_orig_dens | 0.918 ± 0.011 | 0.901 ± 0.046 | 0.090 ± 0.060 | 0.934 ± 0.030 | 0.158 ± 0.089 |
| pair_edge_dens | 0.896 ± 0.054 | 0.829 ± 0.123 | 0.134 ± 0.063 | 0.963 ± 0.022 | 0.224 ± 0.086 |
| concat3 | 0.882 ± 0.053 | 0.799 ± 0.085 | 0.211 ± 0.144 | 0.965 ± 0.043 | 0.311 ± 0.167 |
| learned_fusion | 0.937 ± 0.018 | 0.912 ± 0.045 | 0.148 ± 0.079 | 0.961 ± 0.024 | 0.246 ± 0.109 |
| late_fusion | 0.926 ± 0.025 | 0.890 ± 0.062 | 0.145 ± 0.064 | 0.962 ± 0.025 | 0.242 ± 0.091 |

Validation-selected single representation: BA 0.918 ± 0.030
- concat3 vs SEL: mean diff -0.0359, Wilcoxon exact two-sided p = 0.1562
- learned_fusion vs SEL: mean diff +0.0187, Wilcoxon exact two-sided p = 0.1562
- late_fusion vs SEL: mean diff +0.0079, Wilcoxon exact two-sided p = 0.3125

## iccad5 (6 seeds)

| Method | BA | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|
| orig | 0.928 ± 0.022 | 0.882 ± 0.054 | 0.088 ± 0.039 | 0.975 ± 0.016 | 0.157 ± 0.064 |
| edge | 0.918 ± 0.039 | 0.846 ± 0.083 | 0.268 ± 0.116 | 0.991 ± 0.012 | 0.392 ± 0.150 |
| dens | 0.838 ± 0.093 | 0.691 ± 0.194 | 0.124 ± 0.058 | 0.985 ± 0.013 | 0.197 ± 0.077 |
| pair_orig_edge | 0.909 ± 0.071 | 0.821 ± 0.142 | 0.356 ± 0.125 | 0.996 ± 0.002 | 0.484 ± 0.129 |
| pair_orig_dens | 0.873 ± 0.027 | 0.772 ± 0.059 | 0.117 ± 0.067 | 0.974 ± 0.033 | 0.196 ± 0.104 |
| pair_edge_dens | 0.929 ± 0.013 | 0.874 ± 0.032 | 0.155 ± 0.109 | 0.984 ± 0.011 | 0.249 ± 0.150 |
| concat3 | 0.919 ± 0.063 | 0.854 ± 0.120 | 0.128 ± 0.072 | 0.985 ± 0.007 | 0.218 ± 0.106 |
| learned_fusion | 0.940 ± 0.039 | 0.902 ± 0.077 | 0.212 ± 0.157 | 0.978 ± 0.026 | 0.317 ± 0.212 |
| late_fusion | 0.946 ± 0.025 | 0.919 ± 0.061 | 0.132 ± 0.155 | 0.974 ± 0.016 | 0.202 ± 0.186 |

Validation-selected single representation: BA 0.940 ± 0.012
- concat3 vs SEL: mean diff -0.0209, Wilcoxon exact two-sided p = 0.5625
- learned_fusion vs SEL: mean diff +0.0001, Wilcoxon exact two-sided p = 1.0000
- late_fusion vs SEL: mean diff +0.0063, Wilcoxon exact two-sided p = 0.3125
