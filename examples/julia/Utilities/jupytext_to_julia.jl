import UUIDs: UUID, uuid1

stack = Dict(:tex => false, :plotnum => 0)

function cellType(cell_type, line)
    if !isnothing(cell_type)
        return cell_type
    end
    line == "# +" || !startswith(line, '#') ? :code : :comment
end

function newCell(cells, output, cell_type)
    cell_id = uuid1()
    push!(cells, cell_id)
    write(output, "# ╔═╡ $cell_id\n")
    if cell_type == :comment
        write(output, "md\"\"\"\n")
    elseif cell_type == :code
        write(output, "begin\n")
    end
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
    newCell(cells, output, nothing)
    write(output, """# ╠═╡ show_logs = false
    using Pkg; Pkg.add(url="https://github.com/MeasureTransport/MParT.jl")\n\n"""
    )
    newCell(cells, output, nothing)
    pkgs = [:MParT, :Distributions, :LinearAlgebra,
            :Statistics, :Optimization, :OptimizationOptimJL,
            :GLMakie]
    pkgs = join(string.(pkgs), ", ")
    write(output,"using "*pkgs*"\n\n")
end

function parseComment(line)
    matches = collect(eachmatch(r"\$\$", line))
    line = strip(line[2:end])
    if length(matches) > 0
        if length(matches) == 1
            if stack[:tex]
                line = replace(line, r"\$\$" => s"```")
            else
                line = replace(line, r"\$\$" => s"```math")
            end
            stack[:tex] = !stack[:tex]
        else
            if stack[:tex]
                @error "Can't have multiple \$\$ in a row with tex already on, line: $line"
            end
            line = replace(line, r"\$\$(.*?)\$\$" => s"```math\n\1\n```")
        end
    end
    line
end

function parseCode(line)
    if line == "plt.figure()"
        stack[:plotnum] += 1
    end
    n = stack[:plotnum]
    packages = [
        "np", "mt", "plt"
    ]
    fcns = [
        # ("linspace","range")
    ]
    member_fcns = [
        ("flatten","vec"), ("reshape","reshape"), ("fix","Fix"),
        ("Evaluate",), ("SetCoeffs",), ("CoeffGrad",), ("CoeffMap",)
    ]
    regexes = [
        r"random.randn\(([^,\)]*?)\)([\.\s\)]*)" => s"randn(1,\1)\2", # randn(n) --> randn(1,n)
        r"linspace\((.*?)\)" => s"range(\1)", # linspace --> range
        r"\*\*" => s" .^", # exponentiation
        r">" => s" .>", # Broadcast >
        r"<" => s" .<", # Broadcast < 
        r"\'" => s"\"", # single quotes to double quotes
        r",\s*\)" => s")", # remove trailing commas
        r"(\S)\.shape\[(\d)\]" => s"size(\1,\2)", # Calculate shape appropriately
        r"def ([^\)]*)\(([^,]*),\s*([^\)]*)\):" => s"function \1(\2,p)\n\t\3 = p", # Start a function appropriately
        r"return (.*)" => s"\1\nend", # End a function appropriately
        r"^(\S*)\s*=\s*(.*?)\[None,:\]" => s"\1 = \2\n\1 = collect(reshape(\1, 1, length(\1)))", # Ensuring this reshape idiom behaves appropriately
        r"(\S*)\s*=\s*(.*)\.reshape\(-1,1\)(.*)" => s"\1 = \2\n\1 = reshape(\1,length(\1),1)\3", # Ensuring the reshape behaves appropriately
        r"=\s*(.*)\.astype\(int\)" => s"= Int.(\1)", # Remove astype
        r"sum\((.*?),(\d)\)/"=>s"sum(\1,dims=\2)/", # When sum is over a dimension and is divided by something
    ]
    plot_regexes = [
        (r"figure\(\)", "fig$n = Figure()\nax$n = Axis(fig$n[1,1])"), # Creating plot
        (r"title\((.*)\)", "ax$n.title = \\1"), # Setting plot title
        (r"plot\((.*)\)", "lines!(ax$n, \\1)\nscatter!(ax$n, \\1)"), # Creating lines on axis
        (r"xlabel\((.*)\)", "ax$n.xlabel = \\1"), # Setting plot xlabel
        (r"ylabel\((.*)\)", "ax$n.ylabel = \\1"), # Setting plot ylabel
        (r"legend", "axislegend"), # Creating legend
        (r"color=\"([^,\)]*)\"(,?)(.*),\s*alpha=([^,\)]*)", "color=(:\\1,\\4)\\2\\3"), # Setting color when alpha is not default
        (r"color\s*=\s*\"(.*)\"", "color = :\\1"), # Setting color when alpha is default
        (r"marker\s*=\s*\"\*\"", "marker=:star5"), # Setting for star marker
        (r"linestyle\s*=\s*\"--\"", "linestyle=:dash"), # Setting for dashed linestyle
        (r",(\s*)\"\*--\"", ",\\1linestyle=:dashdot"), # Setting for dashdot linestyle
        (r"show\(\)", "fig$n") # Showing plot
    ]
    optimize_regex = [
r"^(\S*)\s*=\s*minimize\(([^,]*),([^,]*),\s*args=([^\)]*\)),\s*jac=([^,]*),\s*method=\"(.*?)\".*\)"=>
        s"u0 = \3\np = \4\nfcn = OptimizationFunction(\2, grad=\5)\nprob = OptimizationProblem(fcn, u0, p)\n\1 = solve(prob, \6())"
    ]

    packages .*= "."
    for pkg in packages
        line = replace(line, Regex(pkg) => s"")
    end
    for fcn in fcns
        rep_str = Regex(fcn[1])*r"\((.*)\)([\.\s\)])"
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
    for reg in plot_regexes
        line = replace(line, reg[1]=>SubstitutionString(reg[2]))
    end
    for reg in optimize_regex
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
    readline(input)
    cell_type = nothing
    skipline = false
    for x in eachline(input)
        if skipline
            skipline = false
            continue
        end
        cell_type_old = cell_type
        cell_type = cellType(cell_type_old, x)
        if cell_type != cell_type_old
            newCell(cells, output, cell_type)
            cell_type == :code && continue
        end
        if cell_type == :code
            if x == "# -"
                cell_type = nothing
                write(output, "end\n")
                skipline = true
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