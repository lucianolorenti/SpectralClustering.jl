module DocUtils
export bibliography,
       file
using BibTeX
using BibTeXFormat
using Mustache
function bibliography(cites::Vector{String})
    bibliography      = Bibliography(readstring(joinpath(dirname(@__FILE__), "SpectralClustering.bib")))
    formatted_entries = format_entries(UNSRTAlphaStyle,bibliography)
    return write_to_string( HTMLBackend() ,formatted_entries,  cites)
end
function file(name::String)
    local file = readstring(joinpath(dirname(@__FILE__),"src", "js",name))
    return file
end

function file(name::String, data::Dict{String, String})
    local file = readstring(joinpath(dirname(@__FILE__),"src", "js",name))
    return render(file, data)
end
end
