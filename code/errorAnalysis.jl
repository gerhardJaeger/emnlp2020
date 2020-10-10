cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#---

using CSV
using DataFrames
using StatsPlots
using Pipe
using ProgressMeter
using StatsFuns
using RCall
using Images
using Mamba

#---
d = CSV.read("../data/kfoldPredictions.csv", DataFrame)

families = unique(d.glotFam)
f2i = Dict(zip(families, 1:length(families)))
d[:, :familyIndex] = [f2i[fm] for fm in d.glotFam]

features = unique(d.feature)
ft2i = Dict(zip(features, 1:length(features)))
d[:, :featureIndex] = [ft2i[ft] for ft in d.feature]

d[:,:y] = convert.(Int, d.value .== d.prediction)

#---

famSizes = @pipe d |>
      select(_, [:name, :familyIndex]) |>
      unique |>
      groupby(_, :familyIndex) |>
      combine(_, nrow)



fam2Size = Dict(zip(famSizes[:,1], famSizes[:,2]))

d[:,:famSize] = [fam2Size[f] for f in d.familyIndex]

#---

featureFreq = combine(groupby(d, :featureIndex), nrow)

feature2Freq = Dict(zip(featureFreq[:,1], featureFreq[:,2]))

d[:,:featureFreq] = [feature2Freq[f] for f in d.featureIndex]

#---

famAccuracy = @pipe d |> groupby(_, :familyIndex) |> combine(_, :y => mean)

famSizeAccuracy = innerjoin(famSizes, famAccuracy, on=:familyIndex)

rename!(famSizeAccuracy, :nrow => :size, :y_mean => :accuracy)


#---
#
# R"
# library(ggplot2)
# dat <- $famSizeAccuracy
# ggplot(dat, aes(size, accuracy)) +
#       geom_point() +
#       scale_x_log10() +
#       geom_smooth(method='gam') +
#       theme(axis.text=element_text(size=10),
#         axis.title=element_text(size=14))
# ggsave('accuracyFamilySize.pdf')
# "
# load("accuracyFamilySize.pdf")
#
#---

featureAccuracy = @pipe d |> groupby(_, :featureIndex) |> combine(_, :y => mean)

featureFreqAccuracy = innerjoin(featureFreq, featureAccuracy, on=:featureIndex)

rename!(featureFreqAccuracy, :nrow => :freq, :y_mean => :accuracy)

#---
#
# R"
# library(ggplot2)
# dat <- $featureFreqAccuracy
# ggplot(dat, aes(freq, accuracy)) +
#       geom_point() +
#       scale_x_log10() +
#       geom_smooth(method='gam') +
#       theme(axis.text=element_text(size=10),
#         axis.title=element_text(size=14),
#         plot.margin = margin(10, 20, 10, 10))
# ggsave('accuracyFeatureFreq.pdf')
# "
# load("accuracyFeatureFreq.pdf")
#
#---

model = Model(
    y = Stochastic(
        1,
        (n, μ) -> UnivariateDistribution[Bernoulli(invlogit(μ[i])) for i = 1:n],
        false,
    ),
    μ = Logical(
        1,
        (α, β1, β2, r1, r2, fmi, fti, n, logFamilySize, logFeatureFreq) -> [
            α + β1 * logFamilySize[i] + β2 * logFeatureFreq[i] + r1[fmi[i]] + r2[fti[i]]
            for i = 1:n
        ],
        false,
    ),
    α = Stochastic(
        () -> Normal(0, 100)
    ),
    β1 = Stochastic(
        () -> Normal(0, 100)
    ),
    β2 = Stochastic(
        () -> Normal(0, 100)
    ),
    r1 = Stochastic(1,
        (s1, nFamilies) -> MvNormal(zeros(nFamilies), s1)
    ),
    r2 = Stochastic(1,
        (s2, nFeatures) -> MvNormal(zeros(nFeatures), s2)
    ),
    s1 = Stochastic(
        () -> InverseGamma(0.001, 0.001)
    ),
    s2 = Stochastic(
        () -> InverseGamma(0.001, 0.001)
    ),
)

#---

line = Dict{Symbol, Any}(
    :y => d.y,
    :n => size(d,1),
    :nFamilies => length(unique(d.familyIndex)),
    :nFeatures => length(unique(d.featureIndex)),
    :fmi => d.familyIndex,
    :fti => d.featureIndex,
    :logFamilySize => log.(d.famSize),
    :logFeatureFreq => log.(d.featureFreq)
)
#---

inits = [
    Dict{Symbol, Any}(
        :y => d.y,
        :n => size(d,1),
        :nFamilies => length(unique(d.familyIndex)),
        :nFeatures => length(unique(d.featureIndex)),
        :fmi => d.familyIndex,
        :fti => d.featureIndex,
        :logFamilySize => log.(d.famSize),
        :logFeatureFreq => log.(d.featureFreq),
        :α => 0.,
        :β1 => 0.,
        :β2 => 0.,
        :r1 => zeros(line[:nFamilies]),
        :r2 => zeros(line[:nFeatures]),
        :s1 => 1.,
        :s2 => 1.,
    )
    for c in 1:2
]

#---

scheme = [
    Slice([:α, :β1, :β2, :s1, :s2], 3.0),
    Slice([:r1, :r2], 3.0)
]

setsamplers!(model, scheme)

#---

sim = mcmc(model, line, inits, 21000, burnin=1000, thin=10, chains=2)

#---

@show gelmandiag(sim)

#---

p = Mamba.plot(sim)
draw(p, ask=false)
