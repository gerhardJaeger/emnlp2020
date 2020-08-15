include("phytools.jl")


function getAncestors(tree::phylo, pivot::Int)
    getAncestors(tree.edge)
end


function getRootNode(x::phylo)
    nTips = length(x.tip_label)
    if nTips == 1
        1
    else
        nTips+1
    end
end

function getDaughters(tree::phylo, nd::Int)
    findnz(tree.edge[nd,:])[1]
end

function getMother(tree::phylo, node::Int)
    m = findnz(tree.edge[:,node])[1]
    if length(m) == 0
        return(nothing)
    end
    m[1]
end

function lca(tree::phylo, i::Int, j::Int)
    [x for x in getAncestors(tree.edge, i)
     if x in getAncestors(tree.edge, j)][1]
end
