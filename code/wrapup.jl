using Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

using CSV, DataFrames, Pipe

testData = CSV.File("testpredictions.csv") |> DataFrame!


trainingData = CSV.File("../data/train.cldf.csv") |> DataFrame!
devData = CSV.File("../data/dev.cldf.csv") |> DataFrame!

d = vcat(trainingData, devData, testData[:, names(trainingData)])
filter!(x -> x.value != 0, d)

fvDF = @pipe d |>
      select(_, [:feature, :value, :comment]) |>
      unique

fvDict = Dict()

for rw in eachrow(fvDF)
      fvDict[rw.feature, rw.value] = rw.comment
end


for rw in eachrow(testData)
      if rw.prediction != 0
            rw.comment = fvDict[rw.feature, rw.prediction]
      end
end

predictions = filter(x -> x.value == 0, testData)

languages = unique(predictions.wals_code)

reformatted = String[]

for l in languages
      lp = filter(x -> x.wals_code == l, predictions)
      name = first(lp.name)
      family = first(lp.family)
      fvp = join(
            [rw.feature * "=" * string(rw.prediction) * " " * rw.comment
                  for rw in eachrow(lp)],   "|")
      push!(reformatted, join([name, family, fvp], "\t"))
end

open("crosslingference_unconstrained", "w") do io
      for ln in reformatted
            println(io, ln)
      end
end
