#ifndef LBM_H
#define LBM_H

#ifdef __cplusplus
extern "C" {
#endif

void lbm_step(float* d_f, float* d_rho, float* d_u, float* d_v,
              const bool* d_obstacles, float omega, 
              int width, int height);

void lbm_initialize(float* d_f, float rho, float u, float v,
                    int width, int height);

#ifdef __cplusplus
}
#endif

#endif // LBM_H
