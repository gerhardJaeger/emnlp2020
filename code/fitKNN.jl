using Pkg; Pkg.activate(pwd()); Pkg.instantiate()

using CSV,
    DataFrames,
    Pipe,
    Plots,
    Statistics,
    Random,
    StatsPlots,
    Distances,
    ProgressMeter,
    Distributions


kfold = 20
Random.seed!(12345);


#languages = readlines("languages.csv")
d = CSV.File("../data/trainImputed.cldf.csv") |> DataFrame!
d = dropmissing(d, :ASJP)
#d = filter(x -> x.wals_code ∈ languages, d)

n = nrow(d)
languages = unique(d.wals_code)
features = unique(d.feature)
families = @pipe d |>
                dropmissing(_,:glotFam) |>
                unique(_,:glotFam) |> _.glotFam

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



function createTestsets(n = n, kfold = kfold)
    testitems = randperm(n)
    ts = Base.reshape(testitems[1:(n-(n%kfold))], (kfold, n ÷ kfold))
    testsets = [ts[i, :] for i = 1:kfold]
    for i = 1:(n%kfold)
        push!(testsets[i], testitems[end+1-i])
    end
    testsets
end

subsets = createTestsets();

function mostFrequent(x)
    xu = unique(x)
    xu[argmax(map(y -> sum(x .== y), xu))]
end

test = sample(subsets)
model = (1:n)[Not(test)]
i = test[1]


meanEarthRadius = 6372.

function getCoordinates(languages=languages, d=d)
    coorDict = Dict{String,Tuple{Float64, Float64}}()
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


dstMtx = zeros(length(languages), length(languages))
for (i, l1) in enumerate(languages)
    for (j, l2) in enumerate(languages)
        dstMtx[i, j] = gdist[l1, l2]
    end
end

function predict(testing, model, k)
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




a,t, kk = Float64[], Int64[], Int64[]
@showprogress for k = 1:20
    acc = zeros(kfold)
    for i = 1:kfold
        local test = subsets[i]
        local model = (1:n)[Not(test)]
        try
            acc[i] = mean(predict(test, model, k) .== d.value[test])
        catch e
            println(i)
        end
        push!(a, acc[i])
        push!(t, i)
        push!(kk,k)
    end
    accuracy = mean(acc)
end


ev = DataFrame(accuracy=a, testset=t, k=kk)
evDF = combine(:accuracy => mean, groupby(ev, :k))

maxK = evDF.k[argmax(evDF.accuracy_mean)]
@show maxK


theme(:vibrant)

ft1 = Plots.font(18)
ft2 = Plots.font(15)

@df evDF plot(:k, :accuracy_mean, legend=false,
    guidefont=ft1, tickfont=ft2,
)
xlabel!("k")
ylabel!("accuracy")

title!(" ")

savefig("knnOptimization.pdf")
