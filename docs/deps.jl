using Pkg
packages = ["Documenter","Formatting", "Images", "TestImages", 
"RDatasets", "InfoZIP", "ImageView", "ImageMagick","Mustache", 
"StringEncodings", "TextAnalysis", "Latexify", "IJulia"]
for p in packages
    try
        if Pkg.installed(p) == nothing
            Pkg.add(p)
	end
    catch e
        Pkg.add(p)
    end        
end
using Documenter
using SpectralClustering
try
   Pkg.installed("BibTeX")
catch
    Pkg.clone("https://github.com/bramtayl/BibTeX.jl.git")
end
try
    Pkg.installed("BibTeXFormat")
catch
    Pkg.clone("https://github.com/lucianolorenti/BibTeXFormat.jl.git")
end
