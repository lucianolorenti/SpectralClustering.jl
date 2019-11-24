export bibliography,
       file
using BibTeX
using BibTeXFormat
using Mustache
read_file(path) = read(open(path), String)
function bibliography(cites::Vector{String})
    bibliography = Bibliography(read_file(joinpath(dirname(@__FILE__), "SpectralClustering.bib")))
    formatted_entries = format_entries(UNSRTAlphaStyle,bibliography)
    return write_to_string( HTMLBackend() ,formatted_entries,  cites)
end
function file(name::String)
    file = read_file(joinpath(dirname(@__FILE__),"src", "js",name))
    return file
end

function file(name::String, data::Dict{String, String})
    file = read_file(joinpath(dirname(@__FILE__),"src", "js",name))
    return render(file, data)
end
