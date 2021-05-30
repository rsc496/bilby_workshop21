TARGET:=assignment_1
all: pdf
pdf: 
	pdflatex $(TARGET)
	bibtex $(TARGET)
	pdflatex $(TARGET)
	pdflatex $(TARGET)
.PHONY:clean
clean:
	/bin/rm -f *~ *.dvi *.log *.aux *.fls *.toc *.bbl *blg *.end *.fdb_latexmk *.out $(TARGET)Notes.bib */*~ */*.dvi */*.log */*.aux */*.toc */*.bbl */*blg */*.end */*.out $(TARGET)Notes.bib
