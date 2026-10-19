# equate.evaluate.suites

Loaders for the standard matching benchmarks, fetched on first use into a store.

Registered in `benchmarks` (research doc 14, §6 and §8.3):

- `abt_buy`, `amazon_google`, `dblp_acm` — the product and bibliographic tables the
  Magellan / DeepMatcher suite made standard. `source='leipzig'` (default) loads the
  original tables of the Leipzig database group with their **complete** gold mapping
  (CC BY 4.0). `source='deepmatcher'` (named `<suite>_deepmatcher`, so a report keeps
  the two apart) loads DeepMatcher’s preprocessed tables with its
  canonical labelled train/valid/test pairs in `Benchmark.pairs`; its gold is the
  labelled positives only, which misses true matches its blocking dropped (`meta`
  says how many), so it is the source for comparing pair classifiers with published
  numbers, not for scoring whole-table matching.
- `wdc_products` — WDC Products (Peeters, Der & Bizer 2024), the difficulty-controlled
  product benchmark: `corner_cases` (80/50/20 %), `unseen` (0/50/100 % of test
  entities absent from training) and `dev_size` (small/medium/large). Built from its
  multi-class files, with its train/valid/test sets as `Benchmark.splits`.

Every loader takes `store=` (any `MutableMapping`; default
[`default_store()`](equate.evaluate.store.html.md#equate.evaluate.store.default_store)), `download=`, `get=` (the downloader)
and `verify=` (check the pinned SHA-256, on every read — default on; off only for a
trusted mirror), puts the publisher’s record ids in `Benchmark.ids`, and fills `meta` with
`source`, `license`, a `difficulty` sentence and a measured `difficulty_profile`
([`difficulty_profile()`](equate.evaluate.difficulty.html.md#equate.evaluate.difficulty.difficulty_profile), cached in the store).

### Module Attributes

| [`MAGELLAN_SOURCES`](#equate.evaluate.suites.MAGELLAN_SOURCES)   | the selectable sources of the three Magellan/DeepMatcher tables   |
|---------------------------------------------------------------------|-------------------------------------------------------------------|

### Functions

| [`abt_buy`](#equate.evaluate.suites.abt_buy)(\*\*config)                           | Abt-Buy: 1081 × 1092 product offers, 1097 matches (Leipzig; textual, hard).      |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`amazon_google`](#equate.evaluate.suites.amazon_google)(\*\*config)                     | Amazon-Google: 1363 × 3226 software products, 1300 matches (Leipzig; hard).      |
| [`dblp_acm`](#equate.evaluate.suites.dblp_acm)(\*\*config)                          | DBLP-ACM: 2616 × 2294 publications, 2224 matches (Leipzig; easy).                |
| [`wdc_products`](#equate.evaluate.suites.wdc_products)(\*[, corner_cases, unseen, ...]) | WDC Products as a two-table benchmark with its canonical train/valid/test split. |

### equate.evaluate.suites.MAGELLAN_SOURCES *= ('leipzig', 'deepmatcher')*

the selectable sources of the three Magellan/DeepMatcher tables

### equate.evaluate.suites.abt_buy(\*\*config)

Abt-Buy: 1081 × 1092 product offers, 1097 matches (Leipzig; textual, hard).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.suites.amazon_google(\*\*config)

Amazon-Google: 1363 × 3226 software products, 1300 matches (Leipzig; hard).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.suites.dblp_acm(\*\*config)

DBLP-ACM: 2616 × 2294 publications, 2224 matches (Leipzig; easy).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.suites.wdc_products(, corner_cases=80, unseen=100, dev_size='large', seed=0, store=None, download=True, get=None, verify=True, profile=True)

WDC Products as a two-table benchmark with its canonical train/valid/test split.

The defaults pick the hardest generalization variant — 80 % corner cases, every test
entity unseen in training (research doc 14, §7.6) — with the large development set.
Gold pairs every `a`/`b` offer of the same product, across parts too (a seen test
product matches its training offers); only the test-part gold is scored. Row order
within each part is shuffled with `seed`. `meta['leakage']` counts test offers
that also appear, identically, in train or valid. WDC’s own files have such overlap:
hundreds in the 50 %-unseen variants, a few in valid for 0 %. A warning says so.

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)
