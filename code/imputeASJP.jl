using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

using CSV, Distances, DataFrames, Pipe, RCall

train = CSV.File("../data/train.cldf.csv") |> DataFrame
train[!, :source] .= "train"

dev = CSV.File("../data/dev.cldf.csv") |> DataFrame
dev[!, :source] .= "dev"

test = CSV.File("../data/test.cldf.csv") |> DataFrame
test[!, :source] .= "test"

taxaTrain = unique(train.wals_code)
taxaDev = unique(dev.wals_code)
taxaTest = unique(test.wals_code)

d = vcat(train, dev, test)

@rlibrary phytools

pth = "../data/"

worldTree = midpoint_root(read_newick(pth * "RAxML_bestTree.world_sc_ccGlot"))

doculects = convert(Vector, worldTree[Symbol("tip.label")])

languages = @pipe d |>
            dropmissing(_,:ASJP) |>
            filter(x -> x.ASJP in doculects, _) |>
            select(_,:wals_code) |>
            unique |>
            _[:,1]


wals2ASJP = @pipe d |> dropmissing(_, :ASJP) |> zip(_.wals_code, _.ASJP) |> Dict

taxa = [wals2ASJP[l] for l in languages]

meanEarthRadius = 6372.0


allLanguages = unique(d.wals_code)

function getCoordinates(allLanguages = allLanguages, d = d)
    coorDict = Dict{String,Tuple{Float64,Float64}}()
    for l in allLanguages
        lon, lat = first(d[d.wals_code.==l, [:longitude, :latitude]])
        coorDict[l] = (lon, lat)
    end
    coorDict
end

coorDict = getCoordinates()

function getDistances(
    allLanguages = allLanguages,
    d = d,
    meanEarthRadius = meanEarthRadius,
)
    gdist = Dict{Tuple{String,String},Float64}()
    for l1 in allLanguages
        for l2 in allLanguages
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

dstMtx = zeros(length(allLanguages), length(allLanguages))
for (i, l1) in enumerate(allLanguages)
    for (j, l2) in enumerate(allLanguages)
        dstMtx[i, j] = gdist[l1, l2]
    end
end

doculects = unique(d[
    :,
    [:wals_code, :latitude, :longitude, :genus, :family, :ASJP, :glotFam],
])

defined = filter(x -> x.wals_code in languages, dropmissing(doculects, :ASJP))
defined[!, :languageIndex] = indexin(defined.wals_code, allLanguages)

undefined = [x for x in allLanguages if x ∉ languages]


function imputeLanguage(l)
    if l in languages
        return l
    end
    li = first(indexin([l], allLanguages))
    genus, family, glotFam = @pipe doculects |>
          filter(x -> x.wals_code == l, _) |>
          select(_, [:genus, :family, :glotFam]) |>
          first
    gneighbors = filter(x -> x.genus == genus, defined).wals_code
    if length(gneighbors) > 0
        gnIndices = indexin(gneighbors, allLanguages)
        return gneighbors[argmin(dstMtx[li, gnIndices])]
    end
    if !ismissing(glotFam)
        glotneighbors = filter(x -> x.glotFam == glotFam, defined).wals_code
        if length(glotneighbors) > 0
            glnIndices = indexin(glotneighbors, allLanguages)
            return glotneighbors[argmin(dstMtx[li, glnIndices])]
        end
    end
    fneighbors = filter(x -> x.family == family, defined).wals_code
    if length(fneighbors) > 0
        fnIndices = indexin(fneighbors, allLanguages)
        return fneighbors[argmin(dstMtx[li, fnIndices])]
    end
    return defined.wals_code[argmin(dstMtx[li, defined.languageIndex])]
end

doculects[!, :imputation] =
    map(x -> wals2ASJP[x], imputeLanguage.(doculects.wals_code))

wals2imputation = @pipe doculects |> zip(_.wals_code, _.imputation) |> Dict

train[!, :imputation] = [wals2imputation[l] for l in train.wals_code]
CSV.write("../data/trainImputed.cldf.csv", train)

dev[!, :imputation] = [wals2imputation[l] for l in dev.wals_code]
CSV.write("../data/devImputed.cldf.csv", dev)

test[!, :imputation] = [wals2imputation[l] for l in test.wals_code]
CSV.write("../data/testImputed.cldf.csv", test)
