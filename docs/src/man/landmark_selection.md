# Landmark Selection
In order to avoid the construction of a complete similarity matrix some spectral clustering
methods compute the simmilarity function between a subset of patterns. This module provides an
interface to sample points from diferentes data structures.

Methods availaible:
- `Random` . This selection method samples $k$ random points from a dataset
- `EvenlySpaced`. This selection method samples spaced evenly acorrding ther index.
## Detailed Description

### Random Landmark Selection
```@example
using SpectralClustering
number_of_points = 20
dimension = 5
data = rand(dimension,number_of_points)
selector = RandomLandmarkSelection()
number_of_landmarks = 7
select_landmarks(selector, number_of_landmarks, data )
```
### Evenly Spaced Landmark Selection

```@example
using SpectralClustering
number_of_points = 20
dimension = 5
data = rand(dimension,number_of_points)
selector = EvenlySpacedLandmarkSelection()
number_of_landmarks = 5
select_landmarks(selector, number_of_landmarks, data )

```

## Index
```@index
Modules=[SpectralClustering]
Pages=["man/landmark_selection.md"]
```
## Content
```@autodocs
Modules=[SpectralClustering]
Pages=["src/LandmarkSelection.jl"]
```

