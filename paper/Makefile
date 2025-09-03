ZIP_TARGETS:= README.md\
	Bacteria\ Modelling.bib\
	listings-rust.sty\
	pool_model.tex\
	pool_model.pdf\
	$(wildcard Figures/abm-homogenous/*)\
	$(wildcard Figures/abm-homogenous-low-diffusion/*)\
	$(wildcard Figures/abm-inhomogenous/*)\
	$(wildcard Figures/abm-random/*)\

all:
	latexmk -pdf -shell-escape pool_model.tex

clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.dvi
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.out
	rm -f *.pdf
	rm -f *.toc
	rm -f *.run.xml
	rm -f *.bbl-SAVE-ERROR
	rm -f *.bcf-SAVE-ERROR

fresh: clean all

zip: fresh
	zip release.zip $(ZIP_TARGETS)
