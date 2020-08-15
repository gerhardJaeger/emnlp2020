include("utils.jl");
import Base.==
import Base.isless

struct phylo
    edge::SparseMatrixCSC{Float64,Int}
    tip_label::Vector{String}
    node_names::Vector{String}
    root_edge::Float64
end

function ==(x::phylo, y::phylo)
    (x.edge == y.edge) & (x.tip_label == y.tip_label) &
    (x.node_names == y.node_names) & (x.root_edge == y.root_edge)
end

function isless(t1::phylo, t2::phylo)
    if length(t1.tip_label) != length(t2.tip_label)
        return(length(t1.tip_label) < length(t2.tip_label))
    end
    isless(t1.tip_label, t2.tip_label)
end


function phylo(edge::SparseMatrixCSC{Float64,Int};
               tip_label=nothing,
               node_names=nothing,
               root_edge=1.0
               )
    r, c, v = findnz(edge)
    nodes = unique(vcat([r, c]...))
    tips = nodes[[!(nd in r) for nd in nodes]]
    internal = nodes[[(nd in r) for nd in nodes]]
    if tip_label==nothing
        tl = string.(tips)
    else
        tl = tip_label
    end
    if node_names==nothing
        ndnames = repeat([""], length(internal))
    else
        ndnames = node_names
    end
    phylo(edge, tl, ndnames, root_edge)
end



function phylo(tip; root_edge=1.0)
    phylo(spzeros(1, 1), tip_label=[tip], root_edge=root_edge)
end
