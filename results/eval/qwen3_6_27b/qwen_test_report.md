# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.914** (1385/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error
masculine        440         3         62            0
feminine           2       451         52            0
ambiguous          6         5        494            0
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

## Condition: few_shot_full

- Examples: 1515
- Overall accuracy: **0.939** (1422/1515)
- Parse errors: 7

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error
masculine        470         4         31            0
feminine           3       481         15            6
ambiguous         18        15        471            1
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  470  21  35     0.9572  0.9307 0.9438
 feminine  481  19  24     0.9620  0.9525 0.9572
ambiguous  471  46  34     0.9110  0.9327 0.9217
    macro 1422  86  93     0.9434  0.9386 0.9409
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9327
                adjective  feminine  62    1.0000
                adjective masculine  62    0.9677
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    1.0000
              combination masculine   2    1.0000
         context_modifier  feminine  16    0.9375
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  59    1.0000
        pronoun_insertion masculine  59    0.9831
             pronoun_swap  feminine  91    0.9560
             pronoun_swap masculine  91    0.9451
            referent_swap  feminine  27    1.0000
            referent_swap masculine  27    0.6296
     relational_insertion  feminine  29    0.8966
     relational_insertion masculine  29    0.9655
                 sir_maam  feminine  83    0.8193
                 sir_maam masculine  83    0.8916
          title_insertion  feminine 125    0.9920
          title_insertion masculine 125    0.9520
                  unknown  feminine   7    1.0000
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.8000
     True 1470    0.9429
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9420
       True   33    0.7879
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9385
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9421
                True   30    0.7667
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1422            4.995
   False   93            4.860
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.8000
      doctor 27    1.0000
      client 27    1.0000
photographer 27    1.0000
     captain 24    0.8750
       idiot 24    0.9167
      virgin 24    0.8750
       clerk 24    0.9583
    opponent 21    0.9524
   assistant 21    0.9524
      friend 21    1.0000
       lover 21    0.9048
       buddy 21    0.6667
   detective 18    0.8889
       nanny 18    0.7778
       guard 18    1.0000
    employee 18    1.0000
       coach 18    1.0000
     soldier 18    0.8333
      dancer 18    0.8889
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
   referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
      buddy                    4                  4                 0         1.0
      nanny                    3                  0                 3         0.0
   champion                    2                  2                 0         1.0
      chief                    2                  2                 0         1.0
     dancer                    2                  1                 1         0.5
    captain                    2                  2                 0         1.0
     virgin                    2                  0                 2         0.0
      nurse                    2                  0                 2         0.0
    amateur                    1                  0                 1         0.0
  attendant                    1                  0                 1         0.0
  homemaker                    1                  0                 1         0.0
housekeeper                    1                  0                 1         0.0
   governor                    1                  0                 1         0.0
      clerk                    1                  1                 0         1.0
     master                    1                  1                 0         1.0
```

## All-conditions headline summary

```
          metric  zero_shot  few_shot_full
         overall     0.9142         0.9386
        macro_f1     0.9156         0.9409
ambiguity_recall     0.9782         0.9327
masculine_recall     0.8713         0.9307
 feminine_recall     0.8931         0.9525
```

## zero_shot  vs  few_shot_full

- Fixed by few_shot_full (wrong in zero_shot): 75
- Broken by few_shot_full (correct in zero_shot): 38
- Net gain (few_shot_full − zero_shot): 37 examples
- Agreement rate: 1395/1515 = 0.921

Confusion-matrix diff (few_shot_full − zero_shot):
```
           masculine  feminine  ambiguous  parse_error
masculine         30         1        -31            0
feminine           1        30        -37            6
ambiguous         12        10        -23            1
```
