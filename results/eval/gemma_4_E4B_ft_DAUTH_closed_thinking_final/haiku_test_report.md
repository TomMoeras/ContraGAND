# Haiku gender classification — report

## Condition: zero_shot

- Examples: 1395
- Overall accuracy: **0.386** (539/1395)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine          9         0        456            0          0
feminine           0        65        400            0          0
ambiguous          0         0        465            0          0
```

Per-class precision/recall/F1:
```
    label  tp  fp  fn  precision  recall     f1
masculine   9   0 456      1.000  0.0194 0.0380
 feminine  65   0 400      1.000  0.1398 0.2453
ambiguous 465 856   0      0.352  1.0000 0.5207
    macro 539 856 856      0.784  0.3864 0.2680
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows — ambiguous)    source 465    1.0000
                adjective  feminine  62    0.2258
                adjective masculine  62    0.0484
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    0.5000
              combination  feminine   1    0.0000
              combination masculine   1    0.0000
         context_modifier  feminine  16    0.3125
         context_modifier masculine  16    0.1250
        pronoun_insertion  feminine  55    0.1091
        pronoun_insertion masculine  55    0.0000
             pronoun_swap  feminine  89    0.0899
             pronoun_swap masculine  89    0.0112
            referent_swap  feminine  21    0.5238
            referent_swap masculine  21    0.0000
     relational_insertion  feminine  25    0.0400
     relational_insertion masculine  25    0.0000
                 sir_maam  feminine  77    0.0000
                 sir_maam masculine  77    0.0000
          title_insertion  feminine 110    0.1364
          title_insertion masculine 110    0.0091
                  unknown  feminine   5    0.2000
                  unknown masculine   5    0.0000
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   33    0.3636
     True 1362    0.3869
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1368    0.3874
       True   27    0.3333
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1392    0.3865
                  True    3    0.3333
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1371    0.3873
                True   24    0.3333
```

Confidence calibration:
```
 correct   n  mean_confidence
    True 539              5.0
   False 856              5.0
```

Top 20 referents by frequency:
```
    referent  n  accuracy
      client 27    0.3333
       idiot 24    0.3333
       chief 24    0.3333
      virgin 24    0.3333
       clerk 24    0.3750
photographer 24    0.3750
      doctor 24    0.3333
     captain 24    0.3333
   assistant 21    0.3333
      driver 18    0.5000
       guard 18    0.5000
    minister 18    0.3889
       lover 18    0.3333
       coach 18    0.3333
      dancer 18    0.3333
      friend 18    0.3333
     soldier 18    0.4444
   colleague 16    0.3750
  vegetarian 15    0.3333
    opponent 15    0.4000
```
