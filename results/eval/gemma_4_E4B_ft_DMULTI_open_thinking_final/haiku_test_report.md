# Haiku gender classification — report

## Condition: zero_shot_identify

- Examples: 1395
- Overall accuracy: **0.978** (1364/1395)
- Parse errors: 9

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        449         0         10            5          1
feminine           0       455          5            4          1
ambiguous          2         3        460            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  449   2  16     0.9956  0.9656 0.9803
 feminine  455   3  10     0.9934  0.9785 0.9859
ambiguous  460  15   5     0.9684  0.9892 0.9787
    macro 1364  20  31     0.9858  0.9778 0.9816
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows — ambiguous)    source 465    0.9892
                adjective  feminine  62    0.9839
                adjective masculine  62    1.0000
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   1    1.0000
              combination masculine   1    1.0000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    0.9375
        pronoun_insertion  feminine  55    1.0000
        pronoun_insertion masculine  55    0.9818
             pronoun_swap  feminine  89    0.9888
             pronoun_swap masculine  89    0.9775
            referent_swap  feminine  21    0.9524
            referent_swap masculine  21    0.7619
     relational_insertion  feminine  25    0.9200
     relational_insertion masculine  25    0.9600
                 sir_maam  feminine  77    1.0000
                 sir_maam masculine  77    1.0000
          title_insertion  feminine 110    0.9545
          title_insertion masculine 110    0.9455
                  unknown  feminine   5    1.0000
                  unknown masculine   5    1.0000
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   33    0.9697
     True 1362    0.9780
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1368    0.9781
       True   27    0.9630
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1392    0.9777
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1371    0.9781
                True   24    0.9583
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1364            4.999
   False   31            5.000
```

Top 20 referents by frequency:
```
    referent  n  accuracy
      client 27    1.0000
       idiot 24    1.0000
       chief 24    1.0000
      virgin 24    0.9583
       clerk 24    1.0000
photographer 24    1.0000
      doctor 24    1.0000
     captain 24    0.9167
   assistant 21    0.9524
      driver 18    1.0000
       guard 18    0.8889
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
    amateur                    1                  0                 1         0.0
   champion                    1                  1                 0         1.0
    fighter                    1                  1                 0         1.0
housekeeper                    1                  0                 1         0.0
salesperson                    1                  0                 1         0.0
```
