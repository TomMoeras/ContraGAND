# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.963** (1459/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error
masculine        482         1         22            0
feminine           2       485         18            0
ambiguous          6         7        492            0
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

## Condition: few_shot_full

- Examples: 1515
- Overall accuracy: **0.962** (1457/1515)
- Parse errors: 8

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error
masculine        487         1         13            4
feminine           3       493          5            4
ambiguous         16        12        477            0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  487  19  18     0.9625  0.9644 0.9634
 feminine  493  13  12     0.9743  0.9762 0.9753
ambiguous  477  18  28     0.9636  0.9446 0.9540
    macro 1457  50  58     0.9668  0.9617 0.9642
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9446
                adjective  feminine  62    0.9839
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    0.2500
          appositive_dash masculine   4    0.5000
              combination  feminine   2    1.0000
              combination masculine   2    1.0000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    0.9375
        pronoun_insertion  feminine  59    0.9831
        pronoun_insertion masculine  59    0.9661
             pronoun_swap  feminine  91    0.9780
             pronoun_swap masculine  91    1.0000
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.7407
     relational_insertion  feminine  29    1.0000
     relational_insertion masculine  29    1.0000
                 sir_maam  feminine  83    0.9759
                 sir_maam masculine  83    0.9759
          title_insertion  feminine 125    0.9920
          title_insertion masculine 125    0.9840
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.8222
     True 1470    0.9660
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9636
       True   33    0.8788
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9616
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9636
                True   30    0.8667
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1457             4.99
   False   58             4.62
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
      virgin 24    0.9583
       clerk 24    1.0000
    opponent 21    1.0000
   assistant 21    0.9524
      friend 21    0.8571
       lover 21    0.9524
       buddy 21    0.8095
   detective 18    1.0000
       nanny 18    0.8333
       guard 18    0.9444
    employee 18    1.0000
       coach 18    1.0000
     soldier 18    0.8889
      dancer 18    0.9444
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
      buddy                    4                  4                 0         1.0
     master                    3                  3                 0         1.0
      nanny                    3                  0                 3         0.0
       chef                    2                  1                 1         0.5
    soldier                    2                  2                 0         1.0
   champion                    1                  1                 0         1.0
  attendant                    1                  0                 1         0.0
   governor                    1                  1                 0         1.0
      chief                    1                  1                 0         1.0
     dancer                    1                  0                 1         0.0
     friend                    1                  1                 0         1.0
      lover                    1                  0                 1         0.0
housekeeper                    1                  0                 1         0.0
  homemaker                    1                  0                 1         0.0
   neighbor                    1                  0                 1         0.0
```

## All-conditions headline summary

```
          metric  zero_shot  few_shot_full
         overall     0.9630         0.9617
        macro_f1     0.9632         0.9642
ambiguity_recall     0.9743         0.9446
masculine_recall     0.9545         0.9644
 feminine_recall     0.9604         0.9762
```

## zero_shot  vs  few_shot_full

- Fixed by few_shot_full (wrong in zero_shot): 22
- Broken by few_shot_full (correct in zero_shot): 24
- Net gain (few_shot_full − zero_shot): -2 examples
- Agreement rate: 1467/1515 = 0.968

Confusion-matrix diff (few_shot_full − zero_shot):
```
           masculine  feminine  ambiguous  parse_error
masculine          5         0         -9            4
feminine           1         8        -13            4
ambiguous         10         5        -15            0
```
