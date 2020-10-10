## CTMC
using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

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


n = nrow(d)
languages = unique(d.wals_code)
features = unique(d.feature)
families =
    @pipe d |> dropmissing(_, :glotFam) |> unique(_, :glotFam) |> _.glotFam

d.languageIndex = indexin(d.wals_code, languages)
d.featureIndex = indexin(d.feature, features)
d.familyIndex = indexin(d.glotFam, families)


lfvDict = Dict(zip(zip(d.wals_code, d.feature), d.value))

fMatrix = zeros(Int, length(languages), length(features))
for (j, f) in enumerate(features), (i, l) in enumerate(languages)
    if (l, f) in keys(lfvDict)
        fMatrix[i, j] = lfvDict[l, f]
    end
end


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

@rlibrary phytools

pth = "../data/"

worldTree = midpoint_root(read_newick(pth * "RAxML_bestTree.world_sc_ccGlot"))

doculects = convert(Vector, worldTree[Symbol("tip.label")])


include("phyloLL.jl")
pt = rimport("phytools")

taxa = unique(d.imputation)
toBeDropped = [l for l in doculects if l ∉ taxa]

tree = R"library(ape);drop.tip($worldTree, $toBeDropped)"

l2d = Dict(zip(d.wals_code, d.imputation));

taxa = [l2d[l] for l in languages]

model = findall(d.source .== "train")


mMatrix = zeros(Int, size(fMatrix))
mIndices = Matrix{Int}(d[model, [:languageIndex, :featureIndex]])
for v = 1:size(mIndices, 1)
    a, b = mIndices[v, :]
    mMatrix[a, b] = fMatrix[a, b]
end
ft = unique(d.featureIndex[model])

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

rr = 0.1:0.1:10.

lll = zeros(length(rr))

#---

@showprogress for i in 1:length(rr)
    lll[i] = getLLF(rr[i])
end

rr[argmax(lll)]

#---

theme(:vibrant)
ft1 = Plots.font(18)
ft2 = Plots.font(15)
p = plot(rr, lll, legend=false,
    guidefont=ft1, tickfont=ft2,
)
xlabel!(p, "rate")
ylabel!(p, "log-likelihood")
title!(" ")

savefig(p, "ctmcOptimization.pdf")



#---

@time r = optimize(x -> -getLLF(x), 0.001, 100.0).minimizer



function predict(test, model, r = r)
    mMatrix = zeros(Int, size(fMatrix))
    mIndices = Matrix{Int}(d[model, [:languageIndex, :featureIndex]])
    for v = 1:size(mIndices, 1)
        a, b = mIndices[v, :]
        mMatrix[a, b] = fMatrix[a, b]
    end
    ft = unique(d.featureIndex[model])

    predicted = Int[]
    @showprogress for i in test
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



###########

test = findall(
    (d.value .== 0) .* (d.ASJP .=== d.imputation) .* (d.source .== "test"),
)
model = findall(d.value .!= 0)

d[!,:prediction] .= 0
d[test,:prediction] = predict(test, model)


## backup

function mostFrequent(x)
    xu = unique(x)
    xu[argmax(map(y -> sum(x .== y), xu))]
end

meanEarthRadius = 6372.0

function getCoordinates(languages = languages, d = d)
    coorDict = Dict{String,Tuple{Float64,Float64}}()
    for l in languages
        lon, lat = first(d[d.wals_code.==l, [:longitude, :latitude]])
        coorDict[l] = (lon, lat)
    end
    coorDict
end
coorDict = getCoordinates()

function getDistances(
    languages = languages,
    d = d,
    meanEarthRadius = meanEarthRadius,
)
    gdist = Dict{Tuple{String,String},Float64}()
    for l1 in languages
        for l2 in languages
            if l1 <= l2
                gdist[l1, l2] =
                    haversine(coorDict[l1], coorDict[l2], meanEarthRadius)
                gdist[l2, l1] = gdist[l1, l2]
            end
        end
    end
    gdist
end
gdist = getDistances()

function sortNeighbors(languages = languages, d = d, gdist = gdist)
    snDict = Dict{String,Vector{String}}()
    for l in languages
        ln = languages[sortperm([gdist[l, l1] for l1 in languages])][2:end]
        snDict[l] = ln
    end
    snDict
end
snDict = sortNeighbors()

dstMtx = zeros(length(languages), length(languages))
for (i, l1) in enumerate(languages)
    for (j, l2) in enumerate(languages)
        dstMtx[i, j] = gdist[l1, l2]
    end
end

function geoKNN(testing, model, k=8)
    mMatrix = zeros(Int, size(fMatrix))
    mIndices = Matrix{Int}(d[model, [:languageIndex, :featureIndex]])
    for v = 1:size(mIndices, 1)
        a, b = mIndices[v, :]
        mMatrix[a, b] = fMatrix[a, b]
    end
    predicted = zeros(Int, size(testing, 1))
    for (v, i) in enumerate(testing)
        li = d.languageIndex[i]
        fi = d.featureIndex[i]
        miLangs = findall(mMatrix[:, fi] .> 0)
        o = minimum([length(miLangs), k])
        nn = miLangs[sortperm(dstMtx[li, miLangs])[1:o]]
        predicted[v] = mostFrequent(mMatrix[nn, fi])
    end
    predicted
end

testG = findall((d.value .== 0) .* (d.ASJP .!== d.imputation))

d[testG,:prediction] = geoKNN(testG, model)

CSV.write(
    "testpredictions.csv",
    filter(x -> x.source == "test", d[:, Not(:familyIndex)]),
)
