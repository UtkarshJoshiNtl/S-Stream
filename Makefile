.PHONY: help sim clean

help:
	@echo "make sim    Configure and build the desktop simulation"
	@echo "make clean  Remove generated build files"

sim:
	cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel

clean:
	cmake -E rm -rf build
