# equate.compare.numeric_geo

Numeric, date, and geospatial comparators (stage ②): turn a domain distance into a
[0, 1] similarity via a decay function. Pure standard library.

### Functions

| [`exp_decay`](#equate.compare.numeric_geo.exp_decay)([scale, distance])           | Comparator mapping (non-negative) distance `d` to `exp(-d / scale)` in `(0, 1]`.   |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`linear_decay`](#equate.compare.numeric_geo.linear_decay)([max_distance, distance]) | Comparator mapping distance `d` to `1 - d / max_distance`, clamped to [0, 1].      |
| [`gaussian_decay`](#equate.compare.numeric_geo.gaussian_decay)([scale, distance])      | Comparator mapping distance `d` to `exp(-(d / scale)^2 / 2)` in `(0, 1]`.          |
| [`haversine`](#equate.compare.numeric_geo.haversine)(a, b)                        | Great-circle distance in km between two `(lat, lon)` points (degrees).             |
| [`geo_similarity`](#equate.compare.numeric_geo.geo_similarity)([scale_km])             | Geospatial similarity comparator: exponential decay of haversine distance (km).    |

### equate.compare.numeric_geo.exp_decay(scale=1.0, \*, distance=<function \_default_distance>)

Comparator mapping (non-negative) distance `d` to `exp(-d / scale)` in `(0, 1]`.

### equate.compare.numeric_geo.gaussian_decay(scale=1.0, \*, distance=<function \_default_distance>)

Comparator mapping distance `d` to `exp(-(d / scale)^2 / 2)` in `(0, 1]`.

### equate.compare.numeric_geo.geo_similarity(scale_km=10.0)

Geospatial similarity comparator: exponential decay of haversine distance (km).

### equate.compare.numeric_geo.haversine(a, b)

Great-circle distance in km between two `(lat, lon)` points (degrees).

```pycon
>>> round(haversine((0.0, 0.0), (0.0, 1.0)), 1)   # 1 deg lon at the equator
111.2
```

### equate.compare.numeric_geo.linear_decay(max_distance=1.0, \*, distance=<function \_default_distance>)

Comparator mapping distance `d` to `1 - d / max_distance`, clamped to [0, 1].
