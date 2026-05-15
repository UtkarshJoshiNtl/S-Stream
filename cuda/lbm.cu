#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// D2Q9 Lattice constants
const int Q = 9;
const float w[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
                     1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

// Opposite directions for bounce-back
const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__device__ __host__ float equilibrium(float rho, float u, float v, int i) {
    float cu = cx[i] * u + cy[i] * v;
    float u2 = u * u + v * v;
    return w[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
}

__global__ void collision(float* f, float* rho, float* u, float* v, 
                          float omega, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Compute macroscopic quantities
    float local_rho = 0.0f;
    float local_u = 0.0f;
    float local_v = 0.0f;
    
    for (int i = 0; i < Q; i++) {
        int f_idx = (i * height + y) * width + x;
        float fi = f[f_idx];
        local_rho += fi;
        local_u += fi * cx[i];
        local_v += fi * cy[i];
    }
    
    local_u /= local_rho;
    local_v /= local_rho;
    
    // Store macroscopic quantities
    rho[idx] = local_rho;
    u[idx] = local_u;
    v[idx] = local_v;
    
    // Collision step (BGK)
    for (int i = 0; i < Q; i++) {
        int f_idx = (i * height + y) * width + x;
        float feq = equilibrium(local_rho, local_u, local_v, i);
        f[f_idx] = f[f_idx] * (1.0f - omega) + feq * omega;
    }
}

__global__ void streaming(float* f, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Stream particles to neighboring cells
    for (int i = 0; i < Q; i++) {
        int src_x = (x - cx[i] + width) % width;
        int src_y = (y - cy[i] + height) % height;
        
        int dst_idx = (i * height + y) * width + x;
        int src_idx = (i * height + src_y) * width + src_x;
        
        f[dst_idx] = f[src_idx];
    }
}

__global__ void apply_obstacles(float* f, const bool* obstacles, 
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    if (obstacles[idx]) {
        // Bounce-back boundary condition
        for (int i = 0; i < Q; i++) {
            int f_idx = (i * height + y) * width + x;
            int opp_idx = (opp[i] * height + y) * width + x;
            float temp = f[f_idx];
            f[f_idx] = f[opp_idx];
            f[opp_idx] = temp;
        }
    }
}

extern "C" void lbm_step(float* d_f, float* d_rho, float* d_u, float* d_v,
                         const bool* d_obstacles, float omega, 
                         int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    streaming<<<grid, block>>>(d_f, width, height);
    cudaDeviceSynchronize();
    
    apply_obstacles<<<grid, block>>>(d_f, d_obstacles, width, height);
    cudaDeviceSynchronize();
    
    collision<<<grid, block>>>(d_f, d_rho, d_u, d_v, omega, width, height);
    cudaDeviceSynchronize();
}

__global__ void initialize_kernel(float* f, float rho, float u, float v, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int i = 0; i < Q; i++) {
        int f_idx = (i * height + y) * width + x;
        f[f_idx] = equilibrium(rho, u, v, i);
    }
}

extern "C" void lbm_initialize(float* d_f, float rho, float u, float v,
                               int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    initialize_kernel<<<grid, block>>>(d_f, rho, u, v, width, height);
    cudaDeviceSynchronize();
}
