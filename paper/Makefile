ZIP_TARGETS:= README.md\
	$(wildcard *.pdf)\
	$(wildcard *.tex)\
	$(wildcard *.bib)\
	$(wildcard *.sty)\
	$(wildcard Figures/abm-homogenous/*)\
	$(wildcard Figures/abm-homogenous-low-diffusion/*)\
	$(wildcard Figures/abm-inhomogenous/*)\
	$(wildcard Figures/abm-random/*)\

all:
	latexmk -pdf -shell-escape pool_model.tex

clean_partial:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.dvi
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.out
	rm -f *.toc
	rm -f *.run.xml
	rm -f *.bbl-SAVE-ERROR
	rm -f *.bcf-SAVE-ERROR

clean: clean_partial
	rm -f *.pdf

fresh: clean all

zip: fresh clean_partial
	zip release.zip $(ZIP_TARGETS)
