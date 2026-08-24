# Haiku gender classification — report

## Condition: zero_shot_identify

- Examples: 1395
- Overall accuracy: **0.326** (455/1395)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine          0         0        443            0         22
feminine           0         9        434            0         22
ambiguous          0         0        446            0         19
```

Per-class precision/recall/F1:
```
    label  tp  fp  fn  precision  recall     f1
masculine   0   0 465     0.0000  0.0000 0.0000
 feminine   9   0 456     1.0000  0.0194 0.0380
ambiguous 446 877  19     0.3371  0.9591 0.4989
    macro 455 877 940     0.4457  0.3262 0.1790
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows — ambiguous)    source 465    0.9591
                adjective  feminine  62    0.0000
                adjective masculine  62    0.0000
          appositive_dash  feminine   4    0.0000
          appositive_dash masculine   4    0.0000
              combination  feminine   1    0.0000
              combination masculine   1    0.0000
         context_modifier  feminine  16    0.0000
         context_modifier masculine  16    0.0000
        pronoun_insertion  feminine  55    0.0000
        pronoun_insertion masculine  55    0.0000
             pronoun_swap  feminine  89    0.0000
             pronoun_swap masculine  89    0.0000
            referent_swap  feminine  21    0.4286
            referent_swap masculine  21    0.0000
     relational_insertion  feminine  25    0.0000
     relational_insertion masculine  25    0.0000
                 sir_maam  feminine  77    0.0000
                 sir_maam masculine  77    0.0000
          title_insertion  feminine 110    0.0000
          title_insertion masculine 110    0.0000
                  unknown  feminine   5    0.0000
                  unknown masculine   5    0.0000
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   33    0.3030
     True 1362    0.3267
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1368    0.3268
       True   27    0.2963
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1392    0.3261
                  True    3    0.3333
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1371    0.3268
                True   24    0.2917
```

Confidence calibration:
```
 correct   n  mean_confidence
    True 455              5.0
   False 940              5.0
```

Top 20 referents by frequency:
```
    referent  n  accuracy
      client 27    0.2593
       idiot 24    0.3333
       chief 24    0.2917
      virgin 24    0.3333
       clerk 24    0.3333
photographer 24    0.3333
      doctor 24    0.3333
     captain 24    0.3333
   assistant 21    0.3333
      driver 18    0.3333
       guard 18    0.2222
    minister 18    0.3333
       lover 18    0.3333
       coach 18    0.3333
      dancer 18    0.3333
      friend 18    0.3333
     soldier 18    0.2778
   colleague 16    0.3750
  vegetarian 15    0.3333
    opponent 15    0.2667
```
