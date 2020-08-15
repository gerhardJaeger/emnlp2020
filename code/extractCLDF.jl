using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

println("""
Usage: julia extractCLDF.jl <dt>
where <dt> is "train", "dev" or "test"
""")

dt = ARGS[1]

inputFile = "../data/"*dt*".csv"
outputFile = "../data/"*dt*".cldf.csv"

if dt=="test"
    inputFile = "../data/test_blinded.csv"
end


using CSV, DataFrames, DataStructures, Pipe

asjpLanguages = "../data/lexibank-asjp-0c18d44/cldf/languages.csv"
!isfile(asjpLanguages) && begin
    asjpZip = download(
        "https://zenodo.org/record/3843469/files/lexibank/asjp-v19.1.zip",
        "../data/asjp-v19.1.zip",
    )
    run(`unzip -o $asjpZip -d ../data/`)
end
asjp = CSV.File(asjpLanguages) |> DataFrame!



dropmissing!(asjp, :Glottocode);

asjpCC = CSV.File("../data/asjp18Clustered.csv") |> DataFrame!;

asjp = @pipe asjpCC |>
      unique(_, [:doculect, :concept]) |>
      groupby(_, :doculect) |>
      combine(nrow, _) |>
      rename(_, [:doculect, :nEntries]) |>
      sort(_, :nEntries, rev = true) |>
      innerjoin(_, asjp, on = :doculect => :Name) |>
      unique(_, :Glottocode)

walsL = "../data/cldf-datasets-wals-014143f/cldf/languages.csv"
!isfile(walsL) && begin
    walsZip = download(
        "https://zenodo.org/record/3731125/files/cldf-datasets/wals-v2020.zip",
        "../data/wals-v2020.zip",
    )
    run(`unzip -o $walsZip -d ../data/`)
end

walsLanguages = CSV.File(walsL) |> DataFrame!
dropmissing!(walsLanguages, :Glottocode)

glot2wals = DefaultDict(missing)
for (i, w) in zip(walsLanguages.Glottocode, walsLanguages.ID)
    for ii in split(i)
        glot2wals[ii] = w
    end
end

insertcols!(asjp, :wals_code=>[glot2wals[x] for x in asjp.Glottocode])

dropmissing!(asjp, :wals_code)

insertcols!(
    asjp,
    :longname => (@pipe asjp[:, [:classification_wals, :doculect]] |>
           eachrow |>
           Vector.(_) |>
           join.(_, ".") |>
           replace.(_, "-" => "_")),
)


######

raw = readlines(inputFile)

d = @pipe raw[2:end] |>
      split.(_, "\t") |>
      map(x -> x[1:7], _) |>
      hcat(_...) |>
      permutedims |>
      DataFrame |>
      rename(_, split(raw[1])[1:7])


insertcols!(
    d,
    :features => (@pipe raw[2:end] |>
           split.(_, "\t") |>
           map(x -> x[8:end], _) |>
           join.(_, " ")),
)

wals2glot = DefaultDict(missing)
for (w, l) in zip(walsLanguages.ID, walsLanguages.Glottocode)
    wals2glot[w] = l
end


insertcols!(d, :Glottocode => [wals2glot[w] for w in d.wals_code])


glot2ln = DefaultDict(missing)
for (w, l) in zip(asjp.Glottocode, asjp.longname)
    glot2ln[w] = l
end
insertcols!(d, :ASJP => [glot2ln[w] for w in d.Glottocode])


insertcols!(
    asjp,
    :glotFam =>
        (@pipe asjp.classification_glottolog |> split.(_, ",") |> first.(_)),
)

glot2fm = DefaultDict(missing)
for (w, l) in zip(asjp.Glottocode, asjp.glotFam)
    glot2fm[w] = l
end
insertcols!(d, :glotFam => [glot2fm[w] for w in d.Glottocode])


cldfA = []

nonFeatures = filter(x -> x != "features", names(d))

for i = 1:size(d, 1)
    rw = d[i, :]
    ln = Vector(rw[nonFeatures])
    for fvp in split(rw.features, "|")
        f, v = split(fvp, "=")
        v1 = split(v)[1]
        v2 = join(split(v)[2:end], " ")
        push!(cldfA, vcat([ln, [f, v1, v2]]...))
    end
end

cldf = @pipe cldfA |>
      hcat(_...) |>
      permutedims |>
      DataFrame |>
      rename(
          _,
          Symbol.(vcat([nonFeatures, [:feature, :value, :comment]]...)),
      )

CSV.write(outputFile, cldf)

rm("../data/lexibank-asjp-0c18d44", recursive=true)
rm("../data/cldf-datasets-wals-014143f", recursive=true)
rm("../data/wals-v2020.zip")
rm("../data/asjp-v19.1.zip")
