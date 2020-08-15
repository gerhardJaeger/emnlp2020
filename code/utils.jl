using SparseArrays

function preorder(am::SparseMatrixCSC{Float64, Int}, start::Int)
    succesors(n) = am[n,:].nzind
    current = start
    visited = [current]
    stack = succesors(current)
    while length(stack) > 0
        current = popfirst!(stack)
        stack = append!(succesors(current), stack)
        push!(visited, current)
    end
    visited
end

function preorder(am::SparseMatrixCSC{Float64, Int})
    r, c, v = findnz(am)
    root = unique([x for x in r if !(x in c)])[1]
    preorder(am, root)
end

function getAncestors(am::SparseMatrixCSC{Float64, Int}, pivot::Int)
    preorder(sparse(am'), pivot)
end
