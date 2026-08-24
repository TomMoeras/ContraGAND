# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.963** (1459/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        482         1         22            0          0
feminine           2       485         18            0          0
ambiguous          6         7        492            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  482   8  23     0.9837  0.9545 0.9688
 feminine  485   8  20     0.9838  0.9604 0.9719
ambiguous  492  40  13     0.9248  0.9743 0.9489
    macro 1459  56  56     0.9641  0.9631 0.9632
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9743
                adjective  feminine  62    0.9839
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    1.0000
              combination masculine   2    1.0000
         context_modifier  feminine  16    0.8750
         context_modifier masculine  16    0.9375
        pronoun_insertion  feminine  59    0.9661
        pronoun_insertion masculine  59    0.9661
             pronoun_swap  feminine  91    0.9560
             pronoun_swap masculine  91    0.9780
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.5926
     relational_insertion  feminine  29    1.0000
     relational_insertion masculine  29    1.0000
                 sir_maam  feminine  83    0.9157
                 sir_maam masculine  83    0.9639
          title_insertion  feminine 125    0.9840
          title_insertion masculine 125    0.9840
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.8667
     True 1470    0.9660
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9642
       True   33    0.9091
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512     0.963
                  True    3     1.000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9643
                True   30    0.9000
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1459            4.997
   False   56            4.911
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.9333
      doctor 27    0.9630
      client 27    1.0000
photographer 27    1.0000
     captain 24    1.0000
       idiot 24    1.0000
      virgin 24    0.9167
       clerk 24    1.0000
    opponent 21    1.0000
   assistant 21    0.9524
      friend 21    0.9048
       lover 21    1.0000
       buddy 21    0.8095
   detective 18    1.0000
       nanny 18    0.8889
       guard 18    0.9444
    employee 18    1.0000
       coach 18    0.8889
     soldier 18    0.8333
      dancer 18    0.9444
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
 referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    nanny                    2                  0                 2         0.0
  soldier                    2                  2                 0         1.0
 champion                    1                  1                 0         1.0
   dancer                    1                  0                 1         0.0
    chief                    1                  1                 0         1.0
 neighbor                    1                  0                 1         0.0
homemaker                    1                  0                 1         0.0
    nurse                    1                  0                 1         0.0
scientist                    1                  1                 0         1.0
   tailor                    1                  1                 0         1.0
   virgin                    1                  0                 1         0.0
```

## Condition: zero_shot_identify

- Examples: 1515
- Overall accuracy: **0.935** (1416/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        451         1         51            0          2
feminine           2       473         28            0          2
ambiguous          5         5        492            0          3
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  451   7  54     0.9847  0.8931 0.9367
 feminine  473   6  32     0.9875  0.9366 0.9614
ambiguous  492  79  13     0.8616  0.9743 0.9145
    macro 1416  92  99     0.9446  0.9347 0.9375
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9743
                adjective  feminine  62    0.9677
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    1.0000
              combination masculine   2    1.0000
         context_modifier  feminine  16    0.8125
         context_modifier masculine  16    0.8750
        pronoun_insertion  feminine  59    0.9831
        pronoun_insertion masculine  59    0.9661
             pronoun_swap  feminine  91    0.9780
             pronoun_swap masculine  91    1.0000
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.6296
     relational_insertion  feminine  29    0.7241
     relational_insertion masculine  29    0.5517
                 sir_maam  feminine  83    0.8675
                 sir_maam masculine  83    0.7349
          title_insertion  feminine 125    0.9760
          title_insertion masculine 125    0.9760
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.7778
     True 1470    0.9395
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9359
       True   33    0.8788
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9345
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9360
                True   30    0.8667
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1416            4.994
   False   99            4.880
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.9333
      doctor 27    1.0000
      client 27    0.9630
photographer 27    1.0000
     captain 24    1.0000
       idiot 24    0.9167
      virgin 24    0.8333
       clerk 24    1.0000
    opponent 21    0.9048
   assistant 21    0.9524
      friend 21    0.7143
       lover 21    0.8095
       buddy 21    0.6190
   detective 18    1.0000
       nanny 18    0.8889
       guard 18    0.8889
    employee 18    0.9444
       coach 18    0.8889
     soldier 18    0.8889
      dancer 18    0.9444
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
   champion                    1                  1                 0         1.0
      chief                    1                  1                 0         1.0
housekeeper                    1                  0                 1         0.0
      nanny                    1                  0                 1         0.0
   neighbor                    1                  0                 1         0.0
      nurse                    1                  0                 1         0.0
  scientist                    1                  1                 0         1.0
    soldier                    1                  1                 0         1.0
     tailor                    1                  1                 0         1.0
     virgin                    1                  0                 1         0.0
```

## All-conditions headline summary

```
          metric  zero_shot  zero_shot_identify
         overall     0.9630              0.9347
        macro_f1     0.9632              0.9375
ambiguity_recall     0.9743              0.9743
masculine_recall     0.9545              0.8931
 feminine_recall     0.9604              0.9366
```

## zero_shot  vs  zero_shot_identify

- Fixed by zero_shot_identify (wrong in zero_shot): 10
- Broken by zero_shot_identify (correct in zero_shot): 53
- Net gain (zero_shot_identify − zero_shot): -43 examples
- Agreement rate: 1451/1515 = 0.958

Confusion-matrix diff (zero_shot_identify − zero_shot):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        -31         0         29            0          2
feminine           0       -12         10            0          2
ambiguous         -1        -2          0            0          3
```
