TEXFILE   := main_fixed_1
PDFLATEX  := pdflatex -interaction=nonstopmode -halt-on-error
BIBER     := biber

.PHONY: all pdf clean cleanall view

all: pdf

## Full build: 3x pdflatex with a biber run in between, so that
## citations, cross-references, and glossary entries all resolve.
pdf:
	$(PDFLATEX) $(TEXFILE).tex
	$(BIBER) $(TEXFILE)
	$(PDFLATEX) $(TEXFILE).tex
	$(PDFLATEX) $(TEXFILE).tex

## Open the compiled PDF (Linux xdg-open; override OPEN=... on other OSes)
OPEN ?= xdg-open
view: pdf
	$(OPEN) $(TEXFILE).pdf

## Remove auxiliary/build files but keep the PDF
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.lof *.lot \
	      *.fls *.fdb_latexmk *.synctex.gz *.run.xml \
	      *.glo *.gls *.glg *.glsdefs *.acn *.acr *.alg *.ist

## Remove everything the build produces, including the PDF
cleanall: clean
	rm -f $(TEXFILE).pdf
