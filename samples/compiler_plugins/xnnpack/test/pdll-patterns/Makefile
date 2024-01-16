all: patterns_pdl_interp.mlir

patterns_pdl.mlir: patterns_pdll.mlir
	mlir-pdll $< -x=mlir -o $@

patterns_pdl_interp.mlir: patterns_pdl.mlir
	mlir-opt --convert-pdl-to-pdl-interp $< -o $@

clean:
	rm -f *_pdl.mlir *_pdl_interp.mlir

.PHONY: clean
