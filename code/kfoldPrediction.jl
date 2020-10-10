cd(@__DIR__)
using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

#---

using CSV
using DataFrames
using Pipe
using Plots
using StatsBase
using StatsPlots
using Distances
using ProgressMeter
using DataStructures
using RCall
using Base.Threads
using Random
using MLBase
Random.seed!(12345)
plotly()

#---

train = CSV.File("../data/trainImputed.cldf.csv") |> DataFrame!
dev = CSV.File("../data/devImputed.cldf.csv") |> DataFrame!
test = CSV.File("../data/testImputed.cldf.csv") |> DataFrame!
test[!,:value] = replace.(test.value, "?" => "0")
test[!,:value] = parse.(Int, test.value)

train[:,:source] .= "train"
dev[:,:source] .= "dev"
test[:,:source] .= "test"

d = vcat(train, dev, test)
#d = dropmissing(d,:ASJP)
#filter!(x -> x.ASJP == x.imputation, d)

#---

n = nrow(d)
languages = unique(d.wals_code)
features = unique(d.feature)
families =
    @pipe d |> dropmissing(_, :glotFam) |> unique(_, :glotFam) |> _.glotFam

d.languageIndex = indexin(d.wals_code, languages)
d.featureIndex = indexin(d.feature, features)
d.familyIndex = indexin(d.glotFam, families)

#---

lfvDict = Dict(zip(zip(d.wals_code, d.feature), d.value))

fMatrix = zeros(Int, length(languages), length(features))
for (j, f) in enumerate(features), (i, l) in enumerate(languages)
    if (l, f) in keys(lfvDict)
        fMatrix[i, j] = lfvDict[l, f]
    end
end

#---

function mostFrequent(x)
    xu = unique(x)
    xu[argmax(map(y -> sum(x .== y), xu))]
end


wd = @pipe d[:, [:wals_code, :imputation]] |>
      dropmissing |>
      unique |>
      zip(_.wals_code, _.imputation)
language2Doculect = DefaultDict(missing)
for (l, d) in wd
    language2Doculect[l] = d
end

#---

@rlibrary phytools

pth = "../data/"

worldTree = midpoint_root(read_newick(pth * "RAxML_bestTree.world_sc_ccGlot"))

doculects = convert(Vector, worldTree[Symbol("tip.label")])
#---

include("phyloLL.jl")
pt = rimport("phytools")

#---

taxa = unique(d.imputation)
toBeDropped = [l for l in doculects if l ∉ taxa]

tree = R"library(ape);drop.tip($worldTree, $toBeDropped)"

l2d = Dict(zip(d.wals_code, d.imputation));

taxa = [l2d[l] for l in languages]

#---

model = findall(d.source .== "train")


mMatrix = zeros(Int, size(fMatrix))
mIndices = Matrix{Int}(d[model, [:languageIndex, :featureIndex]])
for v = 1:size(mIndices, 1)
    a, b = mIndices[v, :]
    mMatrix[a, b] = fMatrix[a, b]
end
ft = unique(d.featureIndex[model])
#---


function getLLF(r)
    llf = 0.0
    for fi in ft
        drop_i = taxa[mMatrix[:, fi].==0]
        ftreeS = convert(
            String,
            R"library(ape);write.tree(drop.tip($tree, $drop_i))",
        )
        ftree = readTree(ftreeS)
        ftaxa = taxa[mMatrix[:, fi].!=0]
        fchar = mMatrix[mMatrix[:, fi].!=0, fi]
        fstates = unique(fchar)
        charN = hcat([Int.(fchar .== s) for s in fstates]...)
        char = DataFrame(
            hcat([ftaxa, charN]...),
            pushfirst!(Symbol.(fstates), :taxa),
        )
        llf += loglik(ftree, char, r)
    end
    llf
end


#---

r = optimize(x -> -getLLF(x), 0.001, 100.0).minimizer

#---

function predict(test, model, r = r)
    mMatrix = zeros(Int, size(fMatrix))
    mIndices = Matrix{Int}(d[model, [:languageIndex, :featureIndex]])
    for v = 1:size(mIndices, 1)
        a, b = mIndices[v, :]
        mMatrix[a, b] = fMatrix[a, b]
    end
    ft = unique(d.featureIndex[model])

    predicted = Int[]
    for i in test
        fi = d.featureIndex[i]
        li = d.languageIndex[i]
        drop_i = taxa[mMatrix[:, fi].==0]
        filter!(x -> x != taxa[li], drop_i)
        ftreeS = convert(
            String,
            R"library(ape);write.tree(drop.tip($tree, $drop_i))",
        )
        ftree = readTree(ftreeS)
        ftaxa = taxa[mMatrix[:, fi].!=0]
        fchar = mMatrix[mMatrix[:, fi].!=0, fi]
        fstates = unique(fchar)
        charN = hcat([Int.(fchar .== s) for s in fstates]...)
        char = DataFrame(
            hcat([ftaxa, charN]...),
            pushfirst!(Symbol.(fstates), :taxa),
        )
        rtree = reroot(ftree, indexmap(ftree.tip_label)[taxa[li]])
        itree = getSubtree(rtree, getDaughters(rtree, getRootNode(rtree))[2])
        push!(predicted, fstates[argmax(asr(itree, char, r))])
    end
    predicted
end

#---


candidates = findall(
    (d.value .!= 0) .* (d.ASJP .=== d.imputation)
) |> shuffle

#---

nSamples = length(candidates)
k = 20

kf = Kfold(nSamples, k)

#---
d[:,:prediction] .= 0


@showprogress for ii in kf
    train_indices = candidates[ii]
    test_indices = [i for i in candidates if i ∉ train_indices]
    d[test_indices,:prediction] = predict(test_indices, train_indices)
end

#---
CSV.write("../data/kfoldPredictions.csv", d[candidates,:])
