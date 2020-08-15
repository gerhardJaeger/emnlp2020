include("phytools.jl")
include("navigate.jl")

function combineTrees(t1, t2; root_edge=1.0::Float64, root_label=""::String)
    # add option for node name
    tl = vcat([t1.tip_label, t2.tip_label]...)
    nTips = length(tl)
    root = nTips+1
    function t1Shift(nd)
        if nd <= length(t1.tip_label)
            nd
        else
            nd + length(t2.tip_label) + 1
        end
    end
    function t2Shift(nd)
        if nd <= length(t2.tip_label)
            nd + length(t1.tip_label)
        else
            nd + size(t1.edge,1) + 1
        end
    end
    nNodes = size(t1.edge)[1] + size(t2.edge)[1] + 1
    e = spzeros(nNodes, nNodes)
    e[root, t1Shift(getRootNode(t1))] = t1.root_edge
    e[root, t2Shift(getRootNode(t2))] = t2.root_edge
    t1Branches = findnz(t1.edge)
    t2Branches = findnz(t2.edge)
    for (r,c,v) in zip(t1Branches[1], t1Branches[2], t1Branches[3])
        e[t1Shift(r), t1Shift(c)] = v
    end
    for (r,c,v) in zip(t2Branches[1], t2Branches[2], t2Branches[3])
        e[t2Shift(r), t2Shift(c)] = v
    end
    nl = vcat([[root_label], t1.node_names, t2.node_names]...)
    phylo(e, tip_label=tl, root_edge=root_edge, node_names=nl)
end

function getSubtree(tree::phylo, nd::Int)
    if nd == getRootNode(tree)
        return(tree)
    end
    re = findnz(tree.edge[:,nd])[2][1]
    accessible = preorder(tree.edge, nd)
    nTips = length(tree.tip_label)
    tips = sort([n for n in accessible if n <= nTips])
    if length(tips) == 1
        edge = spzeros(1,1)
        tip_label = tree.tip_label[tips]
        node_names = String[]
        return(phylo(edge,
                     tip_label=tip_label,
                     node_names=node_names,
                     root_edge=re))
    end
    internal = [n for n in accessible if n > nTips]
    tl = tree.tip_label[tips]
    nl = tree.node_names[internal .- nTips]
    newNodes = vcat([tips, internal]...)
    edge = copy(tree.edge[newNodes, newNodes])
    phylo(edge,
          tip_label=tl,
          node_names=nl,
          root_edge=re[1])
end


function reroot(tree::phylo, target::Int; d=(1-1e-10)::Float64)
    if target==getRootNode(tree)
        if d>0
            error("d must be 0 if rerooting at root node")
        else
            return(tree)
        end
    end
    mother = getAncestors(tree.edge, target)[2]
    newNode = size(tree.edge,1)+1
    edge = spzeros(newNode, newNode)
    edge[1:(end-1), 1:(end-1)] = tree.edge
    l = edge[mother, target]
    edge[mother, newNode] = d*l
    edge[newNode, target] = (1-d)*l
    edge[mother, target] = 0
    droptol!(edge, 0)
    anc = getAncestors(edge, newNode)
    r, c, v = findnz(edge)
    for (i,j) in zip(r,c)
        if [i,j] ⊆ anc
            edge[j,i] = edge[i,j]
            edge[i,j] = 0
        end
    end
    droptol!(edge, 0)
    oldRoot = getRootNode(tree)
    for i in findnz(edge[:,oldRoot])[1]
        for j in findnz(edge[oldRoot,:])[1]
            edge[i,j] = edge[i,oldRoot] + edge[oldRoot,j]
        end
    end
    edge[oldRoot,:] .= 0
    edge[:,oldRoot] .= 0
    droptol!(edge, 0)
    nodes = preorder(edge)
    tips = [n for n in nodes if n <= length(tree.tip_label)]
    nonTips = [n for n in nodes if n ∉ tips]
    nodes = vcat([tips, nonTips]...)
    tl = tree.tip_label[tips]
    nn = tree.node_names[nonTips .- length(tips) .- 1]
    edge = copy(edge[nodes, nodes])
    phylo(edge, tip_label=tl, node_names=nn, root_edge=tree.root_edge)
end

function ladderize(tree::phylo)
    if length(tree.tip_label)==1
        return(tree)
    end
    daughters = getDaughters(tree, getRootNode(tree))
    subtrees = sort([ladderize(getSubtree(tree, n)) for n in daughters])
    combineTrees(subtrees[1], subtrees[2])
end
