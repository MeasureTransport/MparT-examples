import UUIDs: UUID, uuid1

function cellType(cell_type, line)
    if !isnothing(cell_type)
        return cell_type
    end
    line == "# +" ? :code : :comment
end

function newCell(cells, output, cell_type)
    cell_id = uuid1()
    push!(cells, cell_id)
    write(output, "# ╔═╡ $cell_id\n")
    cell_type == :comment && write(output, "md\"\"\"\n")
end

function readHeader(input)
    readline(input)
    while readline(input) != "# ---"; end
end

function writeHeader(cells, output)
    write(output, """### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils\n\n""")
    newCell(cells, output, :code)
    write(output, """# ╠═╡ show_logs = false
    using Pkg; Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")

    """)
    newCell(cells, output, :code)
    pkgs = [:MParT, :Distributions, :LinearAlgebra,
            :Statistics, :Optimization, :OptimizationOptimJL,
            :GLMakie]
    pkgs = join(string.(pkgs), ", ")
    write(output,"using "*pkgs*"\n")
end

function parseComment(line)
    strip(line[2:end])
end

function commentCell(input, output)
    x = readline(input)
    while length(x) > 0
        write(output, parseComment(x))
        x = readline(input)
    end
    write(output, "\"\"\"")
end

function parseCode(line)
    packages = [
        "np", "mt", "plt"
    ]
    fcns = [
        ("linspace","range"), ("random.randn","randn"),
    ]
    member_fcns = [
        ("flatten","vec"), ("reshape","reshape"), ("fix","Fix"),
        ("Evaluate",), ("SetCoeffs",), ("CoeffGrad",), ("CoeffMap",)
    ]
    regexes = [
        r"\*\*" => s" .^",
        r"\'" => s"\"",
        r",\s*\)" => s")",
        r"(\S)\.shape\[(\d)\]" => s"size(\1,\2)",
        r"\.astype\((.*)\)" => s"",
        r"figure\(\)" => s"fig = Figure()\nax = Axis(fig)",
        r"title\((.*)\)" => s"ax.title = \1",
        r"plot\((.*)\)" => s"scatter!(ax, \1)",
        r"xlabel\((.*)\)" => s"ax.xlabel = \1",
        r"ylabel\((.*)\)" => s"ax.ylabel = \1",
        r"legend" => s"axislegend",
        r"color=\"(.*)\"" => s"color = :\1",
        r"color=\"(.*)\",(.*),\s*alpha=(.*)[,\)]" => s"color=(:\1,\3),\2",
        r"marker=\"\*\"" => s"marker = :star",
        r"linestyle=\"--\"" => s"linestyle = :dash",
        r", \"\*--\"" => s", linestyle=:dashdot",
        r"show\(\)" => s"fig",
        r"^(\S*)\s*=\s*(.*)\[None,:\]" => s"\1 = \2\n\1 = collect(reshape(\1, 1, length(\1)))"
    ]

    packages .*= "."
    for pkg in packages
        line = replace(line, Regex(pkg) => s"")
    end
    for fcn in fcns
        rep_str = Regex(fcn[1])*r"\((.*)\)([\.\s]*)"
        sub_str = SubstitutionString(fcn[2]*raw"(\1)\2")
        line = replace(line, rep_str => sub_str)
    end
    for member_fcn in member_fcns
        py_fcn = member_fcn[1]
        jl_fcn = length(member_fcn) > 1 ? member_fcn[2] : py_fcn
        rep_str = r"(?<=[\s\(,])([^,\s\()]*)\."*Regex(py_fcn)*r"\(([^\)]*)\)"
        sub_str = SubstitutionString(jl_fcn*raw"(\1, \2)")
        line = replace(line, rep_str => sub_str)
    end
    for reg in regexes
        line = replace(line, reg)
    end
    line
end

function codeCell(input, output)
    x = readline(input)
    while x != "# -"
        write(output, parseCode(x)*"\n")
        x = readline(input)
    end
end

function jupytext_to_julia(input, output)
    cells = UUID[]
    readHeader(input)
    writeHeader(cells, output)
    cell_type = nothing
    for x in eachline(input)
        cell_type_old = cell_type
        cell_type = cellType(cell_type_old, x)
        if cell_type != cell_type_old
            newCell(cells, output, cell_type)
            cell_type == :code && continue
        end
        if cell_type == :code
            if x == "# -"
                cell_type = nothing
                write(output, "\n")
            else
                write(output, parseCode(x))
            end
        elseif cell_type == :comment
            if length(x) == 0
                cell_type = nothing
                write(output, "\"\"\"\n")
            else
                write(output, parseComment(x))
            end
        end
        write(output, "\n")
    end
    write(output, "\n# ╔═╡ Cell order:\n")
    for cell_id in cells
        write(output, "# ╠═$cell_id\n")
    end
end

in_name, out_name = ARGS
@info "Performing jupytext to julia conversion on $in_name to $out_name"

open(in_name, "r") do input
    open(out_name, "w") do output
        jupytext_to_julia(input, output)
    end
end