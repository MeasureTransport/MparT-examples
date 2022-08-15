import re
import nbformat as nbf
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="infile",
                  help="Read c++ code from  FILE", metavar="FILE")
parser.add_option("-o", "--output", dest="outfile",
                  help="Save jupyter notebook to FILE", metavar="FILE")

(options, args) = parser.parse_args()

assert(options.outfile is not None)
assert(options.infile is not None)


# Read the file
with open(options.infile) as fin:
    fileContent = fin.read()

# Split the file based on the /*** */ markdown delimiters
exprString = '(\/\*\*\*.*?\*\/)'
parts = re.split(exprString, fileContent, flags=re.S)

# Remove empty sections
parts = [part for part in parts if (len(part)>0)]

cells = []

for part in parts:

    # Markdown cell
    if('/***' in part):
        cells.append( nbf.v4.new_markdown_cell(part.strip()[4:-2]) )
               
    # Code cell
    else:
        cells.append( nbf.v4.new_code_cell(part.strip('\n')) )

# Create a notebook
kernelspec = dict(
   display_name="C++17",
   language="C++17",
   name="xcpp17"
)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = kernelspec
nb['cells'] = cells

with open(options.outfile, 'w') as f:
    nbf.write(nb, f)