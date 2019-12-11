# Clustering Experiments

Spectral Clustering seems to work, but need to experiment with hyper parameters, e.g. `n_components`

Notes:
- usually one large cluster for left, one for right subs
- `n_components` should be the same as `n_clusters` (otherwise left and right subs get mixed)

Example result:

0. accidentallycommunist, anarchism, anarchocommunism, anarchy101, ani_communism, antifascistsofreddit, antiwork, askaliberal, bannedfromthe_donald, beto2020, breadtube, centerleftpolitics, chapotraphouse, chapotraphouse2, chomsky, circlebroke, cringeanarchy, paleoconservative
1. conservatives_only
2. bluemidterm2018
3. againsthatesubreddits, antifastonetoss
4. askaconservative, benshapiro, conservative, conservativelounge, conservatives, jordanpeterson, louderwithcrowder, metacanada, newpatriotism, republican, rightwinglgbt, shitpoliticssays, the_donald, thenewright, tuesday, walkaway


## Clustering algorithms that I couldn't get to work

- Problem: one cluster has almost all instances, the other clusters only have one instance
    - K-Means
    - Affinity Propagation (euclidean affinity)
    - Mean Shift
    - Agglomerative Clustering (Hierarchical clustering)
    - Birch
- Problem: There is only one cluster
    - DBSCAN
        - some instances may be in no cluster
    - OPTICS
        - may work with more experimentation, but probably not worth it


## TODO


[read this for evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
