# Haiku gender classification — report

## Condition: zero_shot_identify

- Examples: 1395
- Overall accuracy: **0.966** (1347/1395)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        453         0          3            0          9
feminine           0       453          3            0          9
ambiguous          0         1        441            0         23
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  453   0  12     1.0000  0.9742 0.9869
 feminine  453   1  12     0.9978  0.9742 0.9859
ambiguous  441   6  24     0.9866  0.9484 0.9671
    macro 1347   7  48     0.9948  0.9656 0.9800
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows — ambiguous)    source 465    0.9484
                adjective  feminine  62    1.0000
                adjective masculine  62    1.0000
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   1    1.0000
              combination masculine   1    1.0000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  55    0.9455
        pronoun_insertion masculine  55    0.9273
             pronoun_swap  feminine  89    0.9438
             pronoun_swap masculine  89    0.9438
            referent_swap  feminine  21    0.9524
            referent_swap masculine  21    0.9524
     relational_insertion  feminine  25    1.0000
     relational_insertion masculine  25    1.0000
                 sir_maam  feminine  77    0.9870
                 sir_maam masculine  77    0.9870
          title_insertion  feminine 110    0.9818
          title_insertion masculine 110    0.9909
                  unknown  feminine   5    1.0000
                  unknown masculine   5    1.0000
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   33    0.9091
     True 1362    0.9670
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1368    0.9678
       True   27    0.8519
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1392    0.9655
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1371    0.9679
                True   24    0.8333
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1347              5.0
   False   48              5.0
```

Top 20 referents by frequency:
```
    referent  n  accuracy
      client 27    0.9259
       idiot 24    1.0000
       chief 24    0.8750
      virgin 24    1.0000
       clerk 24    1.0000
photographer 24    1.0000
      doctor 24    1.0000
     captain 24    1.0000
   assistant 21    0.9524
      driver 18    1.0000
       guard 18    0.8889
    minister 18    1.0000
       lover 18    0.8889
       coach 18    1.0000
      dancer 18    1.0000
      friend 18    1.0000
     soldier 18    0.8333
   colleague 16    1.0000
  vegetarian 15    1.0000
    opponent 15    0.9333
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
housekeeper                    1                  0                 1         0.0
```
