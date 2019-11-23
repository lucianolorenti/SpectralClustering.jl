using LinearAlgebra,
      SparseArrays

function normalize_matrix(A::AbstractMatrix, dim::Integer; f=LinearAlgebra.norm)
    if (size(A, dim) == 1)
        return A
    end
    return A./mapslices(f, A, dims=[dim])
end

normalize_rows(A::AbstractMatrix; f=LinearAlgebra.norm) = normalize_matrix(A, 2, f=f)
normalize_cols(A::AbstractMatrix; f=LinearAlgebra.norm) = normalize_matrix(A, 1, f=f)

function normalize_matrix!(A::AbstractMatrix, dim::Integer; f=LinearAlgebra.norm)
    if (size(A, dim) == 1)
        return A
    end
    A ./= mapslices(f, A, dims=[dim])
end
normalize_rows!(A::AbstractMatrix; f=LinearAlgebra.norm) = normalize_matrix!(A, 2, f=f)
normalize_cols!(A::AbstractMatrix; f=LinearAlgebra.norm) = normalize_matrix!(A, 1, f=f)
#=
# get the column sums of A
S = vec(sum(A,1))

# get the nonzero entries in A. ei is row index, ej is col index, ev is the value in A
ei,ej,ev = findnz(A)

# get the number or rows and columns in A
m,n = size(A)

# create a new normalized matrix. For each nonzero index (ei,ej), its new value will be
# the old value divided by the sum of that column, which can be obtained by S[ej]
A_normalized = sparse(ei,ej,ev./S[ej],m,n)
http://stackoverflow.com/questions/24296856/in-julia-how-can-i-column-normalize-a-sparse-matrix
=#
function normalize_cols(A::SparseMatrixCSC)
    sums = sum(A, 1) + eps()
    I, J, V = findnz(A)
    for idx in 1:length(V)
        V[idx] /= sums[J[idx]]
    end
    sparse(I, J, V)
end

function svd_whiten(X)
    U, s, Vt = svd(X)
    return U * Vt
end
