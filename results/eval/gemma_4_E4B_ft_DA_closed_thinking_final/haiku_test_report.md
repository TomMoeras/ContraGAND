# Haiku gender classification — report

## Condition: zero_shot

- Examples: 1395
- Overall accuracy: **0.993** (1385/1395)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        461         0          4            0          0
feminine           0       461          4            0          0
ambiguous          1         1        463            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  461   1   4     0.9978  0.9914 0.9946
 feminine  461   1   4     0.9978  0.9914 0.9946
ambiguous  463   8   2     0.9830  0.9957 0.9893
    macro 1385  10  10     0.9929  0.9928 0.9928
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows — ambiguous)    source 465    0.9957
                adjective  feminine  62    1.0000
                adjective masculine  62    1.0000
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   1    1.0000
              combination masculine   1    1.0000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  55    1.0000
        pronoun_insertion masculine  55    0.9818
             pronoun_swap  feminine  89    0.9775
             pronoun_swap masculine  89    0.9888
            referent_swap  feminine  21    0.9524
            referent_swap masculine  21    0.9524
     relational_insertion  feminine  25    0.9600
     relational_insertion masculine  25    0.9600
                 sir_maam  feminine  77    1.0000
                 sir_maam masculine  77    1.0000
          title_insertion  feminine 110    1.0000
          title_insertion masculine 110    1.0000
                  unknown  feminine   5    1.0000
                  unknown masculine   5    1.0000
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   33    1.0000
     True 1362    0.9927
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1368    0.9934
       True   27    0.9630
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1392    0.9928
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1371    0.9934
                True   24    0.9583
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1385              5.0
   False   10              5.0
```

Top 20 referents by frequency:
```
    referent  n  accuracy
      client 27    1.0000
       idiot 24    1.0000
       chief 24    1.0000
      virgin 24    1.0000
       clerk 24    1.0000
photographer 24    1.0000
      doctor 24    1.0000
     captain 24    0.9583
   assistant 21    0.9524
      driver 18    1.0000
       guard 18    1.0000
    minister 18    1.0000
       lover 18    1.0000
       coach 18    1.0000
      dancer 18    1.0000
      friend 18    1.0000
     soldier 18    1.0000
   colleague 16    1.0000
  vegetarian 15    1.0000
    opponent 15    1.0000
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    fighter                    1                  1                 0         1.0
housekeeper                    1                  0                 1         0.0
```
