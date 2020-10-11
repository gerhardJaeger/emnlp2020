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

R"
library(ggplot2)
dat <- $famSizeAccuracy
ggplot(dat, aes(size, accuracy)) +
      geom_point() +
      scale_x_log10() +
      geom_smooth(method='gam') +
      theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14))
ggsave('accuracyFamilySize.pdf')
"
load("accuracyFamilySize.pdf")
#
#---

featureAccuracy = @pipe d |> groupby(_, :featureIndex) |> combine(_, :y => mean)

featureFreqAccuracy = innerjoin(featureFreq, featureAccuracy, on=:featureIndex)

rename!(featureFreqAccuracy, :nrow => :freq, :y_mean => :accuracy)

#---

R"
library(ggplot2)
dat <- $featureFreqAccuracy
ggplot(dat, aes(freq, accuracy)) +
      geom_point() +
      scale_x_log10() +
      geom_smooth(method='gam') +
      theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14),
        plot.margin = margin(10, 20, 10, 10))
ggsave('accuracyFeatureFreq.pdf')
"
load("accuracyFeatureFreq.pdf")
#
#---

function featureEntropy(x)
    vals = unique(x)
    probs = [mean(x .== v) for v in vals]
    -probs' * log2.(probs)
end




#---

d = @pipe d |> select(_, [:feature, :value]) |>
    unique |>
    groupby(_, :feature) |>
    combine(_, :value => featureEntropy => :entropy) |>
    innerjoin(d, _ , on=:feature)


dat = @pipe d |>
      groupby(_, :featureIndex) |>
      combine(_, :y => mean, :entropy) |>
      rename!(_, :y_mean => :accuracy)

#---
R"
library(ggplot2)
dat <- $dat
ggplot(dat, aes(entropy, accuracy)) +
      geom_point() +
      geom_smooth(method='lm') +
      theme(axis.text=element_text(size=10),
        axis.title=element_text(size=14),
        plot.margin = margin(10, 20, 10, 10))
ggsave('accuracyNvalues.pdf')
"
load("accuracyNvalues.pdf")



#---

R"
library(rstan)
library(brms)
dat = $d
fit = brm(y ~ entropy + featureFreq + famSize + (1|glotFam) + (1|feature),
      data=dat, family=bernoulli)
summary(fit)
"
