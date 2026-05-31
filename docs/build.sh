#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <file.tex>"
	exit 1
fi

tex_file="$1"

if [[ ! -f "$tex_file" ]]; then
	echo "Error: file not found: $tex_file"
	exit 1
fi

latexmk -xelatex "$tex_file"
latexmk -c
rm -f *.xdv *.snm *.nav