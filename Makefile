.PHONY: help sim test clean

help:
	@echo "make sim    Configure and build the desktop simulation"
	@echo "make test   Build and run native C tests"
	@echo "make clean  Remove generated build files"

sim:
	cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel

test: sim
	./build/test_voxelize
	cd simulation && MESA_GL_VERSION_OVERRIDE=4.3 LIBGL_ALWAYS_SOFTWARE=1 USE_EGL=0 xvfb-run -a ../build/test_lbm
	cd simulation && MESA_GL_VERSION_OVERRIDE=4.3 LIBGL_ALWAYS_SOFTWARE=1 USE_EGL=0 xvfb-run -a bash test/test_integration.sh

clean:
	cmake -E rm -rf build
