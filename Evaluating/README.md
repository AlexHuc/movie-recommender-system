# Quickstart

1. **Prepare the data**
Place the MovieLens “latest-small” CSV files under `ml-latest-small/`

2. **Run the evaluation**
`python Evaluating/TestMetrics.py`

OUTPUT:
```bash
Loading movie ratings...

Computing movie popularity ranks so we can measure novelty later...

Computing item similarities so we can measure diversity later...
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.

Building recommendation model...

Computing recommendations...

Evaluating accuracy of model...
RMSE:  0.9033701087151802
MAE:  0.6977882196132263

Evaluating top-10 recommendations...
Computing recommendations with leave-one-out...
Predict ratings for left-out set...
Predict all missing ratings...
Compute top 10 recs per user...

Hit Rate:  0.029806259314456036

rHR (Hit Rate by Rating value): 
3.5 0.017241379310344827
4.0 0.0425531914893617
4.5 0.020833333333333332
5.0 0.06802721088435375

cHR (Cumulative Hit Rate, rating >= 4):  0.04960835509138381

ARHR (Average Reciprocal Hit Rank):  0.0111560570576964

Computing complete recommendations, no hold outs...

User coverage:  0.9552906110283159
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.

Diversity:  0.9665208258150911

Novelty (average popularity rank):  491.5767777960256
```

3. Interpretation of metrics
- **RMSE / MAE**: Lower is better—raw prediction accuracy.
- **HR (Hit Rate)**: Fraction of held-out items that appear in each user’s top-10.
- **cHR**: Same as HR but only for actual ratings ≥ 4.0 (“do they like it?”).
- **ARHR**: Weighted hit rate that rewards higher-ranked hits.
- **Coverage**: Percentage of users who get at least one recommendation ≥ 4.0.
- **Diversity**: 1 − s, where s is average item-item similarity in the top-10. Higher = more varied.
- **Novelty**: Average popularity rank of recommended movies (higher means more obscure).