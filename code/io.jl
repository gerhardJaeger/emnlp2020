include("utils.jl")
include("phytools.jl")
include("navigate.jl")
include("treeSurgery.jl")


function readTree(text::String)
    text = string(split(text, ";")[1])::String
    text = replace(text, r"[ \t]" => "")::String
    text = replace(text, r"''" => "")::String
    textS = convert(Array{String}, split(text, ""))
    state = zeros(Int, length(textS))
    protected = zeros(Int, length(textS))
    counter = 0::Int
    p = 0::Int
    for (i, s) in enumerate(textS)
        if s == "("
            counter += 1
        end
        if s == ")"
            counter -= 1
        end
        state[i] = counter
        if (s == "'") | (s == "\"")
            p = 1-p
        end
        protected[i] = p
    end
    splitpoints = findall((state .== 1) .& (textS .== ",") .& (protected .== 0))
    if length(splitpoints) == 0
        if occursin(":", text)
            tl, rls = convert(Array{String}, split(text, ":"))
            rl = (if rls == "" 0.0 else parse(Float64, rls) end)
        else
            tl = text
            rl = 1.0
        end
        phylo(tl, root_edge=rl)
    else
        if length(splitpoints) > 1
            println("tree is not binary-branching")
            return nothing
        end
        text1 = join(textS[2:(splitpoints[1]-1)])
        closingBracket = findlast((state .== 0) .& (textS .== ")"))
        text2 = join(textS[(splitpoints[1]+1):(closingBracket-1)])
        trailing = join(textS[(closingBracket+1):end])
        if occursin(":", trailing)
            nodeLabel, bls = convert(Array{String}, split(trailing, ":"))
            bl = maximum([1e-5,parse(Float64, bls)])
        elseif trailing == ""
            nodeLabel = ""
            bl = 1.0
        else
            nodeLabel = trailing
            bl = 1.0
        end
        combineTrees(readTree(text1), readTree(text2), root_edge=bl,
        root_label=nodeLabel)
    end
end



function _writeTree(tree::phylo)
    if length(tree.tip_label) == 1
        return(tree.tip_label[1]*":"*string(tree.root_edge))
    end
    daughters = getDaughters(tree, getRootNode(tree))
    s = "("
    s *= join([_writeTree(getSubtree(tree, d)) for d in daughters],",")
    s *= ")"*tree.node_names[1]*":"*string(tree.root_edge)
end

function writeTree(x)
    _writeTree(x)*";"
end
