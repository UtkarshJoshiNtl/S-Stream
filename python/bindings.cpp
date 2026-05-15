#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "../cuda/lbm.h"

namespace py = pybind11;

class LBM2D {
private:
    int width;
    int height;
    float omega;
    float* d_f;
    float* d_rho;
    float* d_u;
    float* d_v;
    bool* d_obstacles;
    
public:
    LBM2D(int w, int h, float viscosity = 0.02) 
        : width(w), height(h) {
        omega = 1.0f / (3.0f * viscosity + 0.5f);
        
        size_t f_size = 9 * height * width * sizeof(float);
        size_t scalar_size = height * width * sizeof(float);
        size_t bool_size = height * width * sizeof(bool);
        
        cudaMalloc(&d_f, f_size);
        cudaMalloc(&d_rho, scalar_size);
        cudaMalloc(&d_u, scalar_size);
        cudaMalloc(&d_v, scalar_size);
        cudaMalloc(&d_obstacles, bool_size);
        
        // Initialize obstacles to false
        cudaMemset(d_obstacles, 0, bool_size);
        
        // Initialize with equilibrium
        initialize(1.0f, 0.0f, 0.0f);
    }
    
    ~LBM2D() {
        cudaFree(d_f);
        cudaFree(d_rho);
        cudaFree(d_u);
        cudaFree(d_v);
        cudaFree(d_obstacles);
    }
    
    void initialize(float rho, float u, float v) {
        // Initialize with equilibrium distribution
        const float w[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
                           1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
        const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
        
        float* h_f = new float[9 * height * width];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float u2 = u * u + v * v;
                for (int i = 0; i < 9; i++) {
                    float cu = cx[i] * u + cy[i] * v;
                    float feq = w[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
                    h_f[(i * height + y) * width + x] = feq;
                }
            }
        }
        
        cudaMemcpy(d_f, h_f, 9 * height * width * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_f;
    }
    
    void set_obstacles(py::array_t<bool> obstacles) {
        py::buffer_info buf = obstacles.request();
        bool* ptr = static_cast<bool*>(buf.ptr);
        
        cudaMemcpy(d_obstacles, ptr, height * width * sizeof(bool), cudaMemcpyHostToDevice);
    }
    
    void step() {
        lbm_step(d_f, d_rho, d_u, d_v, d_obstacles, omega, width, height);
    }
    
    void run(int steps) {
        for (int i = 0; i < steps; i++) {
            step();
        }
    }
    
    py::array_t<float> get_density() {
        py::array_t<float> result = py::array_t<float>(height * width);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        cudaMemcpy(ptr, d_rho, height * width * sizeof(float), cudaMemcpyDeviceToHost);
        
        result.resize({height, width});
        return result;
    }
    
    py::array_t<float> get_velocity() {
        py::array_t<float> result = py::array_t<float>(2 * height * width);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        cudaMemcpy(ptr, d_u, height * width * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ptr + height * width, d_v, height * width * sizeof(float), cudaMemcpyDeviceToHost);
        
        result.resize({height, width, 2});
        return result;
    }
    
    int get_width() const { return width; }
    int get_height() const { return height; }
    float get_omega() const { return omega; }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "CUDA-accelerated LBM fluid simulation";
    
    py::class_<LBM2D>(m, "LBM2D")
        .def(py::init<int, int, float>(), 
             py::arg("width"), py::arg("height"), py::arg("viscosity") = 0.02)
        .def("initialize", &LBM2D::initialize,
             py::arg("rho") = 1.0f, py::arg("u") = 0.0f, py::arg("v") = 0.0f)
        .def("set_obstacles", &LBM2D::set_obstacles)
        .def("step", &LBM2D::step)
        .def("run", &LBM2D::run)
        .def("get_density", &LBM2D::get_density)
        .def("get_velocity", &LBM2D::get_velocity)
        .def_property_readonly("width", &LBM2D::get_width)
        .def_property_readonly("height", &LBM2D::get_height)
        .def_property_readonly("omega", &LBM2D::get_omega);
}
