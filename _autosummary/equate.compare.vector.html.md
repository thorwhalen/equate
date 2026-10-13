# equate.compare.vector

Vector comparators (stage ②): pairwise similarity/distance between two vectors.

### Functions

| [`cosine`](#equate.compare.vector.cosine)(u, v)           | Cosine similarity between two vectors (0 if either is a zero vector).      |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`dot`](#equate.compare.vector.dot)(u, v)              | Dot product of two vectors (an unbounded similarity).                      |
| [`angular_distance`](#equate.compare.vector.angular_distance)(u, v) | Angular distance in [0, 1] — a true metric derived from cosine similarity. |

### equate.compare.vector.angular_distance(u, v)

Angular distance in [0, 1] — a true metric derived from cosine similarity.

### equate.compare.vector.cosine(u, v)

Cosine similarity between two vectors (0 if either is a zero vector).

#### NOTE
cosine is a *similarity*, not a metric — do not use it with a
triangle-inequality index (use [`angular_distance()`](#equate.compare.vector.angular_distance) when a metric is needed).

### equate.compare.vector.dot(u, v)

Dot product of two vectors (an unbounded similarity).
