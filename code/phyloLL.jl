using CSV, DataFrames, StatsFuns, Statistics, Optim

include("treeSurgery.jl")
include("io.jl")
include("utils.jl")

function asr(tree::phylo, char::DataFrame, r::Float64)
    nTaxa = length(tree.tip_label)
    nNodes = 2*nTaxa-1
    charI = convert(Matrix{Float64},
        char[indexin(tree.tip_label, char[:,1]),2:end])::Matrix{Float64}
    nStates = size(charI,2)::Int
    llTips = log.(charI)::Matrix{Float64}
    ll = Dict{Int, Vector{Float64}}()
    for nd = 1:nTaxa
        ll[nd] = llTips[nd,:]
    end

    for nd in nNodes:-1:(nTaxa + 1)
        daughters, branches = findnz(tree.edge[nd,:])
        llnd = zeros(Float64,nStates)
        for (i,d) in enumerate(daughters)
            lld = ll[d]
            lse_lld = logsumexp(lld)
            ci = log(1 - exp(-r * branches[i])) - log(nStates) + lse_lld
            for s in 1:nStates
                llnd[s] += logaddexp(ci, ll[d][s] - r*branches[i])
            end
        end
        ll[nd] = llnd
    end
    ll[nTaxa+1] - repeat([log(nStates)], nStates)
end

loglik(tree::phylo, char::DataFrame, r::Float64) = logsumexp(asr(tree, char, r))


function rhat(tree::phylo, char::DataFrame)
    exp(optimize(x -> -loglik(tree, char, exp(first(x))),
                 [0.], LBFGS()).minimizer[1])
end

function getPhiMatrix(tree::phylo)
    ntips = length(tree.tip_label)
    nnodes = size(tree.edge,1)
    el = Vector(mapslices(sum,tree.edge,dims=1)[1,:])
    topology = zeros(ntips, nnodes)
    for t in 1:ntips
        topology[t,getAncestors(tree.edge,t)] .= 1
    end
    phi = mapslices(x -> x .* sqrt.(el), topology, dims=2)[:,Not(ntips+1)]
    return sparse(phi)
end
