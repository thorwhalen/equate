# Equate

This is a package with tools for matching things. 

Dirty things like language, files in your file system, socks and whistles.

# Install

```
pip install equate
```

Moving on...

# Little peep

Merging/joining tables is very common instance, yet only a small part of what is
possible, and often needed. Consider the following use cases:

- Find the columns to match (join keys) by comparing how well the values of the column
match.

- Comparing the values of the columns with something more flexible than hard equality.
For example, correlation, similarity, etc.

- Find near duplicate columns

- Find rows to align, based on flexible comparison of fuzzily matched cells

## Simple case

Say you have two sets of strings, and all you do is want to is match each element of
the "keys" set to an element of the "values" set (never reusing the same value for a 
different key), and say you know that a matching value string will differ from its 
key only by a few characters. In that case just do this:

```python
keys = ['apple', 'banana', 'carrot']
values = ['car', 'app', 'carob', 'cabana']
dict(match_greedily(keys, values))
# == {'apple': 'app', 'banana': 'cabana', 'carrot': 'carob'}
```

Note that by default, `match_greedily` uses the edit-distance as its `score_func`, 
and you can specify your own custom `score_func`. But the matching is done "greedily". 
This means that at every step, the highest-score value will be taken as the match. 
Some times this is not ideal (typically, your matches validity will decline along with 
the number of keys matched). 

## More involved methods

Here's a more involved entry point:

```python
from equate import similarity_matrix

keys = ['apple pie', 'apple crumble', 'banana split']
values = ['american pie', 'big apple', 'american girl', 'banana republic']
dict(match_keys_to_values(keys, values))
# == {'apple pie': 'american pie',
# 'apple crumble': 'big apple',
# 'banana split': 'banana republic'}
```

The algorithm that gets you these matches is a bit more involved, and parametrizable.
This is how it works.

First, it computes a similarity matrix, and then applies a "matcher" that will 
search through this similarity matrix for an optimal matching, using any 
optimal search function you want (and of course, we provide a few standard ones).

The similarity matrix, if full, contains scores for every possible combination of 
keys and values. 
You can specify how to compute this similarity matrix, which means that if you 
don't want to compute the similarity for every combination, you don't need to. 
Usually, your similarity matrix will be a "sparse matrix", that is, only specify a 
few non-zero entries.

The default similarity matrix function uses a `obj_to_vect` function along with a 
vector similarity function `similarty_func` to compute what the `score_func` 
did in our first `match_greedily` function.
By default here, though, `similarity_matrix` will use methods that are more powerful 
than just an edit-distance. It will learn and use a "TfIdf" vectorization 
(a.k.a. embedding) and a cosine similarity function. 
As a result, you should get finer matchings.

```python
from equate import similarity_matrix

keys = ['apple pie', 'apple crumble', 'banana split']
values = ['american pie', 'big apple', 'american girl', 'banana republic']
m = similarity_matrix(keys, values)
m.round(2).tolist()
# == [[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
```

The `equate.util` module has a few optimal matching function you can use from 
there to extract the matching pairs from the matrix. 
At the time of writing this, we've implemented: 
`greedy_matching`, 
`hungarian_matching`, 
`maximal_matching`,
`stable_marriage_matching`, and 
`kuhn_munkres_matching`.


# An example: In search of a import-to-package name matcher

Here, we'll go through an actual practical example of when you might want to match 
things: "Guessing" the pip install name from a pip package name, 
and other related analyses.

## The problem

Ever got an import error and wondered what the pip install package name was.

Say... 
```
ImportError: No module named skimage
```

But it ain't `pip install skimage` is it (well, it USED to not to, but you get the point...).
What you actually need to do to install (with `pip`) is:
```
pip install scikit-image
```

I would have guessed that!

So no, it's annoying. It shouldn't be allowed. And since it is, there should be an index out there to help out, right?

```
pip install --just-find-it-for-me skimage
```

Instead of just complaining, I thought I'd throw some code at it.
(I'll still complain though.)

Here's a solution: Ask the world (of semantic clouds -- otherwise known as "Google") about it...

## A (fun) solution


```python
import requests
import re
from collections import Counter

search_re = re.compile('(?<=pip install\W)[-\w]+')

def pkg_name_options(query):
    r = requests.get('https://www.google.com/search', params={'q': f'python "pip install" {query}'})
    if r.status_code == 200:
        return Counter(filter(lambda x: x != query, p.findall(r.content.decode('latin-1')))).most_common()
    
def best_guess(query):
    t = pkg_name_options(query)
    if t:
        return t[0][0]
        
```


```python
>>> pkg_name_options('skimage')
[('scikit-image', 5),
 ('-e', 2),
 ('virtualenv', 1),
 ('scikit', 1),
 ('scikit-', 1),
 ('pillow', 1)]
```









```python
>>> best_guess('skimage')
'scikit-image'
```


Yay, it works!
With a sample of one!
Let's try two...


```python
>>> pkg_name_options('sklearn')
[('numpy', 3), ('scikit-learn', 2), ('-U', 2), ('scikit-', 1), ('scipy', 1)]
```




Okay, so it already fails. 

Sure, I could parse more carefully. I could dig into the webpages and get more scope. 

That'd be fun. 

But that's not very nice too Google (and probably is illegal, if anyone cares). 

What you'll find next is an attempt to look at the man in the mirror instead. Looking locally, where the packages actually are: In the site-packages folders...



## Extract, analyze and compare site-packages info names


```python
import pandas as pd
import numpy as np

from equate.examples.site_names import (
    DFLT_SITE_PKG_DIR,    
    site_packages_info_df,
    print_n_null_elements_in_each_column_containing_at_least_one,
    Lidx,
)
```


```python
>>> DFLT_SITE_PKG_DIR
'~/.virtualenvs/382/lib/python3.8/site-packages'
```



```python
>>> data = site_packages_info_df()
>>> print(f"{data.shape}")
(303, 8)
>>> data
```






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dist_info_dirname</th>
      <th>info_kind</th>
      <th>dist_name</th>
      <th>most_frequent_record_dirname</th>
      <th>first_line_of_top_level_txt</th>
      <th>installer</th>
      <th>metadata_name</th>
      <th>pypi_url_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xlrd-1.2.0.dist-info</td>
      <td>dist-info</td>
      <td>xlrd</td>
      <td>xlrd</td>
      <td>xlrd</td>
      <td>pip</td>
      <td>xlrd</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>boltons-20.2.0.dist-info</td>
      <td>dist-info</td>
      <td>boltons</td>
      <td>boltons</td>
      <td>boltons</td>
      <td>pip</td>
      <td>boltons</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>appdirs-1.4.3.dist-info</td>
      <td>dist-info</td>
      <td>appdirs</td>
      <td>appdirs</td>
      <td>appdirs</td>
      <td>pip</td>
      <td>appdirs</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yapf-0.29.0.dist-info</td>
      <td>dist-info</td>
      <td>yapf</td>
      <td>yapftests</td>
      <td>yapf</td>
      <td>pip</td>
      <td>yapf</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cmudict-0.4.4.dist-info</td>
      <td>dist-info</td>
      <td>cmudict</td>
      <td>cmudict</td>
      <td>cmudict</td>
      <td>pip</td>
      <td>cmudict</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>simplegeneric-0.8.1.dist-info</td>
      <td>dist-info</td>
      <td>simplegeneric</td>
      <td>simplegeneric</td>
      <td>simplegeneric</td>
      <td>pip</td>
      <td>simplegeneric</td>
      <td>None</td>
    </tr>
    <tr>
      <th>299</th>
      <td>plotly-4.6.0.dist-info</td>
      <td>dist-info</td>
      <td>plotly</td>
      <td>plotly</td>
      <td>_plotly_future_</td>
      <td>pip</td>
      <td>plotly</td>
      <td>None</td>
    </tr>
    <tr>
      <th>300</th>
      <td>rsa-3.4.2.dist-info</td>
      <td>dist-info</td>
      <td>rsa</td>
      <td>rsa</td>
      <td>rsa</td>
      <td>pip</td>
      <td>rsa</td>
      <td>None</td>
    </tr>
    <tr>
      <th>301</th>
      <td>backcall-0.1.0.dist-info</td>
      <td>dist-info</td>
      <td>backcall</td>
      <td>backcall</td>
      <td>backcall</td>
      <td>pip</td>
      <td>backcall</td>
      <td>None</td>
    </tr>
    <tr>
      <th>302</th>
      <td>cantools-33.1.1.dist-info</td>
      <td>dist-info</td>
      <td>cantools</td>
      <td>cantools</td>
      <td>cantools</td>
      <td>pip</td>
      <td>cantools</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 8 columns</p>
</div>



```python
>>> print_n_null_elements_in_each_column_containing_at_least_one(data)
most_frequent_record_dirname:	1 null values
first_line_of_top_level_txt:	6 null values
installer:	32 null values
metadata_name:	1 null values
pypi_url_name:	255 null values
```


```python
>>> lidx = Lidx(data)
>>> df = data[lidx.no_nans]
>>> print(f"no nan df: {len(df)=}")
no_nans: 302
equal: 187
dash_underscore_eq: 220
('equal', 'dash_underscore_eq'): 186
```


```python
>>> lidx = Lidx(df)
>>> lidx.print_diagnosis()
no_nans: 302
equal: 187
dash_underscore_eq: 220
('equal', 'dash_underscore_eq'): 186
```





```python
>>> lidx = Lidx(df, 'first_line_of_top_level_txt')
>>> lidx.print_diagnosis()
no_nans: 297
equal: 182
dash_underscore_eq: 214
('equal', 'dash_underscore_eq'): 181
```



```python
>>> t = Lidx(df, 'most_frequent_record_dirname')
>>> tt = Lidx(df, 'first_line_of_top_level_txt')
>>> sum(t.equal | tt.equal)
199
```


```python
>> sum(t.dash_underscore_eq | tt.dash_underscore_eq)
233
```



