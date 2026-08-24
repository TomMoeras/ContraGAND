# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.914** (1385/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        440         3         62            0          0
feminine           2       451         52            0          0
ambiguous          6         5        494            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  440   8  65     0.9821  0.8713 0.9234
 feminine  451   8  54     0.9826  0.8931 0.9357
ambiguous  494 114  11     0.8125  0.9782 0.8877
    macro 1385 130 130     0.9257  0.9142 0.9156
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9782
                adjective  feminine  62    1.0000
                adjective masculine  62    0.9677
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    1.0000
              combination masculine   2    1.0000
         context_modifier  feminine  16    0.9375
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  59    0.9492
        pronoun_insertion masculine  59    0.9153
             pronoun_swap  feminine  91    0.9451
             pronoun_swap masculine  91    0.9670
            referent_swap  feminine  27    0.9630
            referent_swap masculine  27    0.5926
     relational_insertion  feminine  29    0.7241
     relational_insertion masculine  29    0.7586
                 sir_maam  feminine  83    0.6988
                 sir_maam masculine  83    0.6627
          title_insertion  feminine 125    0.9120
          title_insertion masculine 125    0.9360
                  unknown  feminine   7    1.0000
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.8222
     True 1470    0.9170
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9170
       True   33    0.7879
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512     0.914
                  True    3     1.000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9172
                True   30    0.7667
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1385            4.987
   False  130            4.838
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.8667
      doctor 27    0.9630
      client 27    1.0000
photographer 27    0.9630
     captain 24    0.9167
       idiot 24    0.8750
      virgin 24    0.8333
       clerk 24    1.0000
    opponent 21    0.9048
   assistant 21    0.9524
      friend 21    0.7143
       lover 21    0.8095
       buddy 21    0.6190
   detective 18    0.8333
       nanny 18    0.8889
       guard 18    0.7778
    employee 18    1.0000
       coach 18    1.0000
     soldier 18    0.8333
      dancer 18    0.8889
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
      chief                    2                  2                 0         1.0
   champion                    1                  1                 0         1.0
    fighter                    1                  1                 0         1.0
housekeeper                    1                  0                 1         0.0
      nanny                    1                  0                 1         0.0
   neighbor                    1                  0                 1         0.0
      nurse                    1                  0                 1         0.0
  scientist                    1                  1                 0         1.0
     tailor                    1                  1                 0         1.0
     virgin                    1                  0                 1         0.0
```

## Condition: zero_shot_identify

- Examples: 1515
- Overall accuracy: **0.733** (1111/1515)
- Parse errors: 14

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        290         1        187            1         26
feminine           5       349        120            2         29
ambiguous          3         3        472           11         16
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  290   8 215     0.9732  0.5743 0.7223
 feminine  349   4 156     0.9887  0.6911 0.8135
ambiguous  472 307  33     0.6059  0.9347 0.7352
    macro 1111 319 404     0.8559  0.7334 0.7570
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9347
                adjective  feminine  62    0.9677
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    0.2500
          appositive_dash masculine   4    0.5000
              combination  feminine   2    0.5000
              combination masculine   2    0.5000
         context_modifier  feminine  16    0.6875
         context_modifier masculine  16    0.5000
        pronoun_insertion  feminine  59    0.8983
        pronoun_insertion masculine  59    0.7797
             pronoun_swap  feminine  91    0.7143
             pronoun_swap masculine  91    0.6374
            referent_swap  feminine  27    0.9630
            referent_swap masculine  27    0.5926
     relational_insertion  feminine  29    0.2069
     relational_insertion masculine  29    0.0690
                 sir_maam  feminine  83    0.3976
                 sir_maam masculine  83    0.1928
          title_insertion  feminine 125    0.7040
          title_insertion masculine 125    0.6080
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.5714
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.5778
     True 1470    0.7381
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.7355
       True   33    0.6364
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.7335
                  True    3    0.6667
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.7354
                True   30    0.6333
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1111            4.971
   False  404            4.777
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.6667
      doctor 27    0.7778
      client 27    0.7407
photographer 27    0.9259
     captain 24    0.7917
       idiot 24    0.1667
      virgin 24    0.3750
       clerk 24    0.8333
    opponent 21    0.6667
   assistant 21    0.7619
      friend 21    0.5714
       lover 21    0.5238
       buddy 21    0.5238
   detective 18    0.7222
       nanny 18    0.7222
       guard 18    0.6111
    employee 18    0.7778
       coach 18    0.7778
     soldier 18    0.6111
      dancer 18    0.9444
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
   champion                    1                  1                 0         1.0
housekeeper                    1                  0                 1         0.0
   neighbor                    1                  0                 1         0.0
  scientist                    1                  1                 0         1.0
     tailor                    1                  1                 0         1.0
     virgin                    1                  0                 1         0.0
```

## All-conditions headline summary

```
          metric  zero_shot  zero_shot_identify
         overall     0.9142              0.7333
        macro_f1     0.9156              0.7570
ambiguity_recall     0.9782              0.9347
masculine_recall     0.8713              0.5743
 feminine_recall     0.8931              0.6911
```

## zero_shot  vs  zero_shot_identify

- Fixed by zero_shot_identify (wrong in zero_shot): 11
- Broken by zero_shot_identify (correct in zero_shot): 285
- Net gain (zero_shot_identify − zero_shot): -274 examples
- Agreement rate: 1203/1515 = 0.794

Confusion-matrix diff (zero_shot_identify − zero_shot):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine       -150        -2        125            1         26
feminine           3      -102         68            2         29
ambiguous         -3        -2        -22           11         16
```
