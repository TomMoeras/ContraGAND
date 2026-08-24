# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.972** (1472/1515)
- Parse errors: 1

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        483         1         20            1          0
feminine           0       494         11            0          0
ambiguous          6         4        495            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  483   6  22     0.9877  0.9564 0.9718
 feminine  494   5  11     0.9900  0.9782 0.9841
ambiguous  495  31  10     0.9411  0.9802 0.9602
    macro 1472  42  43     0.9729  0.9716 0.9720
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9802
                adjective  feminine  62    0.9839
                adjective masculine  62    0.9516
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    0.5000
              combination masculine   2    0.5000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  59    0.9831
        pronoun_insertion masculine  59    0.9831
             pronoun_swap  feminine  91    0.9780
             pronoun_swap masculine  91    0.9780
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.5926
     relational_insertion  feminine  29    1.0000
     relational_insertion masculine  29    1.0000
                 sir_maam  feminine  83    0.9880
                 sir_maam masculine  83    0.9880
          title_insertion  feminine 125    0.9760
          title_insertion masculine 125    0.9840
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.8222
     True 1470    0.9762
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9730
       True   33    0.9091
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9716
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9731
                True   30    0.9000
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1472            4.997
   False   43            4.905
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.9333
      doctor 27    1.0000
      client 27    1.0000
photographer 27    1.0000
     captain 24    1.0000
       idiot 24    1.0000
      virgin 24    1.0000
       clerk 24    0.9583
    opponent 21    0.9524
   assistant 21    1.0000
      friend 21    0.8571
       lover 21    1.0000
       buddy 21    0.9524
   detective 18    1.0000
       nanny 18    0.8889
       guard 18    1.0000
    employee 18    1.0000
       coach 18    0.8889
     soldier 18    0.9444
      dancer 18    1.0000
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
 referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    nanny                    2                  0                 2         0.0
 champion                    1                  1                 0         1.0
    chief                    1                  1                 0         1.0
    clerk                    1                  1                 0         1.0
   friend                    1                  1                 0         1.0
 neighbor                    1                  0                 1         0.0
    nurse                    1                  0                 1         0.0
scientist                    1                  1                 0         1.0
  soldier                    1                  1                 0         1.0
```

## Condition: zero_shot_identify

- Examples: 1515
- Overall accuracy: **0.943** (1428/1515)
- Parse errors: 3

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        456         1         45            2          1
feminine           0       474         29            1          1
ambiguous          2         4        498            0          1
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  456   2  49     0.9956  0.9030 0.9470
 feminine  474   5  31     0.9896  0.9386 0.9634
ambiguous  498  74   7     0.8706  0.9861 0.9248
    macro 1428  81  87     0.9519  0.9426 0.9451
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9861
                adjective  feminine  62    0.9839
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    0.5000
              combination masculine   2    0.5000
         context_modifier  feminine  16    0.9375
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  59    0.9831
        pronoun_insertion masculine  59    0.9661
             pronoun_swap  feminine  91    0.9890
             pronoun_swap masculine  91    1.0000
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.6296
     relational_insertion  feminine  29    0.7241
     relational_insertion masculine  29    0.7241
                 sir_maam  feminine  83    0.9398
                 sir_maam masculine  83    0.8434
          title_insertion  feminine 125    0.9120
          title_insertion masculine 125    0.8960
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.7556
     True 1470    0.9483
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9426
       True   33    0.9394
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9425
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9428
                True   30    0.9333
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1428            4.996
   False   87            4.938
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.9667
      doctor 27    0.9630
      client 27    0.9630
photographer 27    0.9630
     captain 24    1.0000
       idiot 24    0.9583
      virgin 24    0.9583
       clerk 24    1.0000
    opponent 21    0.9524
   assistant 21    0.9524
      friend 21    0.8095
       lover 21    0.9048
       buddy 21    0.6667
   detective 18    1.0000
       nanny 18    0.8889
       guard 18    1.0000
    employee 18    0.9444
       coach 18    0.8333
     soldier 18    1.0000
      dancer 18    1.0000
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
 referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    nanny                    2                  0                 2         0.0
 champion                    1                  1                 0         1.0
 neighbor                    1                  0                 1         0.0
    nurse                    1                  0                 1         0.0
scientist                    1                  1                 0         1.0
```

## All-conditions headline summary

```
          metric  zero_shot  zero_shot_identify
         overall     0.9716              0.9426
        macro_f1     0.9720              0.9451
ambiguity_recall     0.9802              0.9861
masculine_recall     0.9564              0.9030
 feminine_recall     0.9782              0.9386
```

## zero_shot  vs  zero_shot_identify

- Fixed by zero_shot_identify (wrong in zero_shot): 12
- Broken by zero_shot_identify (correct in zero_shot): 56
- Net gain (zero_shot_identify − zero_shot): -44 examples
- Agreement rate: 1444/1515 = 0.953

Confusion-matrix diff (zero_shot_identify − zero_shot):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        -27         0         25            1          1
feminine           0       -20         18            1          1
ambiguous         -4         0          3            0          1
```
