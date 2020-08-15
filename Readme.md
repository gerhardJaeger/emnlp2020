# Readme

This repository contains data and code accompanying the paper "Imputing typological values via phylogenetic inference" (submission to the SIGTYP 2020 Shared Task).

All scripts were developed with Julia v1.5.0. You need an R installation (I used version 3.6.3) with the libraries `phytools` and `ape` installed.

## Workflow

In the directory `code`:

- `julia extractCLDF.jl train`: convert the training data to CLDF format
- `julia extractCLDF.jl dev` convert the development data to CLDF format
- `julia extractCLDF.jl test` convert the test data to CLDF format
- `julia imputeASJP.jl`: merging the WALS data with ASJP data and identifying proxies for languages without a corresponding ASJP doculect
- 