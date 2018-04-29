# Data Access

In order to establish how the data is going to be accessed, the module `DataAccess` provides an unified interface to access to the data for the underlaying algorithms. Every `DataAccessor` must implement this two methods:
1. `get_element(d::T, X, i::Integer)`. This function must return the i-th pattern of `X`.
2. `number_of_patterns(d::T,X)`. This function must return the numer of patterns of `X`

# Reference Index
```@index
Modules = [SpectralClustering]
Pages=["man/data_access.md"]
```

# Members Documentation

```@autodocs
Modules = [SpectralClustering]
Pages=["Utils/DataAccess.jl"]
```
