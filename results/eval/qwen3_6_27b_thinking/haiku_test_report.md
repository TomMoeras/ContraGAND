# Haiku gender classification - report

## Condition: zero_shot

- Examples: 1515
- Overall accuracy: **0.954** (1446/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        476         1         28            0          0
feminine           3       475         27            0          0
ambiguous          6         4        495            0          0
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  476   9  29     0.9814  0.9426 0.9616
 feminine  475   5  30     0.9896  0.9406 0.9645
ambiguous  495  55  10     0.9000  0.9802 0.9384
    macro 1446  69  69     0.9570  0.9545 0.9548
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9802
                adjective  feminine  62    0.9839
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    1.0000
          appositive_dash masculine   4    1.0000
              combination  feminine   2    0.5000
              combination masculine   2    0.5000
         context_modifier  feminine  16    1.0000
         context_modifier masculine  16    1.0000
        pronoun_insertion  feminine  59    0.9831
        pronoun_insertion masculine  59    0.9661
             pronoun_swap  feminine  91    0.9560
             pronoun_swap masculine  91    0.9780
            referent_swap  feminine  27    0.9630
            referent_swap masculine  27    0.7407
     relational_insertion  feminine  29    0.9310
     relational_insertion masculine  29    0.9655
                 sir_maam  feminine  83    0.9036
                 sir_maam masculine  83    0.8916
          title_insertion  feminine 125    0.9280
          title_insertion masculine 125    0.9600
                  unknown  feminine   7    0.5714
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.7556
     True 1470    0.9605
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.9555
       True   33    0.9091
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.9544
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.9556
                True   30    0.9000
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1446            4.987
   False   69            4.754
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.9000
      doctor 27    0.9630
      client 27    1.0000
photographer 27    1.0000
     captain 24    1.0000
       idiot 24    0.9583
      virgin 24    0.9583
       clerk 24    1.0000
    opponent 21    0.9048
   assistant 21    0.9524
      friend 21    0.9048
       lover 21    0.9048
       buddy 21    0.7619
   detective 18    1.0000
       nanny 18    0.8333
       guard 18    0.8889
    employee 18    1.0000
       coach 18    0.8889
     soldier 18    0.9444
      dancer 18    1.0000
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
 referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    buddy                    1                  1                 0         1.0
 champion                    1                  1                 0         1.0
    chief                    1                  1                 0         1.0
 neighbor                    1                  0                 1         0.0
    nurse                    1                  0                 1         0.0
scientist                    1                  1                 0         1.0
  soldier                    1                  1                 0         1.0
```

## Condition: zero_shot_identify

- Examples: 1515
- Overall accuracy: **0.737** (1116/1515)
- Parse errors: 0

Confusion matrix (rows=expected, cols=predicted):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine        320         0        173            0         12
feminine           0       313        181            0         11
ambiguous          5         5        483            0         12
```

Per-class precision/recall/F1:
```
    label   tp  fp  fn  precision  recall     f1
masculine  320   5 185     0.9846  0.6337 0.7711
 feminine  313   5 192     0.9843  0.6198 0.7606
ambiguous  483 354  22     0.5771  0.9564 0.7198
    macro 1116 364 399     0.8487  0.7366 0.7505
```

Breakdown by strategy (masc/fem variants):
```
                 strategy   variant   n  accuracy
(source rows - ambiguous)    source 505    0.9564
                adjective  feminine  62    1.0000
                adjective masculine  62    0.9839
          appositive_dash  feminine   4    0.5000
          appositive_dash masculine   4    0.5000
              combination  feminine   2    0.5000
              combination masculine   2    0.5000
         context_modifier  feminine  16    0.5000
         context_modifier masculine  16    0.5000
        pronoun_insertion  feminine  59    0.9322
        pronoun_insertion masculine  59    0.9322
             pronoun_swap  feminine  91    0.7802
             pronoun_swap masculine  91    0.7582
            referent_swap  feminine  27    0.9630
            referent_swap masculine  27    0.6667
     relational_insertion  feminine  29    0.3793
     relational_insertion masculine  29    0.3793
                 sir_maam  feminine  83    0.2048
                 sir_maam masculine  83    0.0482
          title_insertion  feminine 125    0.4400
          title_insertion masculine 125    0.6800
                  unknown  feminine   7    0.7143
                  unknown masculine   7    0.8571
```

Breakdown by pipeline acceptance:
```
 accepted    n  accuracy
    False   45    0.6667
     True 1470    0.7388
```

Breakdown by filter_any:
```
 filter_any    n  accuracy
      False 1482    0.7355
       True   33    0.7879
```

Breakdown by filter_multi_sentence:
```
 filter_multi_sentence    n  accuracy
                 False 1512    0.7361
                  True    3    1.0000
```

Breakdown by filter_pre_gendered:
```
 filter_pre_gendered    n  accuracy
               False 1485    0.7360
                True   30    0.7667
```

Confidence calibration:
```
 correct    n  mean_confidence
    True 1116            4.956
   False  399            4.736
```

Top 20 referents by frequency:
```
    referent  n  accuracy
       chief 30    0.6667
      doctor 27    0.8889
      client 27    0.7407
photographer 27    0.8889
     captain 24    0.7500
       idiot 24    0.2083
      virgin 24    0.5417
       clerk 24    0.7500
    opponent 21    0.8571
   assistant 21    0.8095
      friend 21    0.5238
       lover 21    0.6190
       buddy 21    0.4762
   detective 18    0.5556
       nanny 18    0.7222
       guard 18    0.5556
    employee 18    0.6111
       coach 18    0.7222
     soldier 18    0.8889
      dancer 18    0.9444
```

Stereotypical bias on ambiguous sources (confidence ≥ 4):
```
 referent  n_confident_guesses  guessed_masculine  guessed_feminine  male_share
    nanny                    3                  0                 3         0.0
    buddy                    1                  1                 0         1.0
    chief                    1                  1                 0         1.0
 champion                    1                  1                 0         1.0
   master                    1                  1                 0         1.0
 neighbor                    1                  0                 1         0.0
scientist                    1                  1                 0         1.0
   virgin                    1                  0                 1         0.0
```

## All-conditions headline summary

```
          metric  zero_shot  zero_shot_identify
         overall     0.9545              0.7366
        macro_f1     0.9548              0.7505
ambiguity_recall     0.9802              0.9564
masculine_recall     0.9426              0.6337
 feminine_recall     0.9406              0.6198
```

## zero_shot  vs  zero_shot_identify

- Fixed by zero_shot_identify (wrong in zero_shot): 6
- Broken by zero_shot_identify (correct in zero_shot): 336
- Net gain (zero_shot_identify − zero_shot): -330 examples
- Agreement rate: 1167/1515 = 0.770

Confusion-matrix diff (zero_shot_identify − zero_shot):
```
           masculine  feminine  ambiguous  parse_error  not_found
masculine       -156        -1        145            0         12
feminine          -3      -162        154            0         11
ambiguous         -1         1        -12            0         12
```
