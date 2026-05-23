latexmk -xelatex main.tex
latexmk -c
rm *.xdv
mv main.pdf report.pdf