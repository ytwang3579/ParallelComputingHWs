#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#define N_STEPS 200000
#define DT 60.
#define EPS 1e-3
#define G 6.674e-11
#define GRAVITY_DEVICE_MASS(M0, T) (M0 + 0.5 * M0 * fabs(sin(T/6000.)))
#define PLANET_RADIUS ((double) 1e7)
#define MISSILE_SPEED ((double) 1e6)
#define GET_MISSILE_COST(T) (1e5 + 1e3 * T)

__device__ double ax1[1024], ay1[1024], az1[1024];
__device__ int hit_time_step_cuda = -2;

// namespace param {
// const int n_steps = 200000;
// const double dt = 60;
// const double eps = 1e-3;
// const double G = 6.674e-11;
// double gravity_device_mass(double m0, double t) {
//     return m0 + 0.5 * m0 * fabs(sin(t / 6000));
// }
// const double planet_radius = 1e7;
// const double missile_speed = 1e6;
// double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
// }  // namespace param

void read_input(std::ifstream& fin, int& n,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    double* m, char** type) {

    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__global__ void run_step(int step, int n, double* qx, double* qy,
    double* qz, double* vx, double* vy,
    double* vz, const double* m,
    char** type) {
    
    if(hit_time_step_cuda != -2) return;

    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int gridStride = gridDim.x * blockDim.x;

    // compute accelerations
    __shared__ double ax[1024], ay[1024], az[1024];
    
    int i = blockIdx.x;
    int j = threadIdx.x;
    ax[j] = 0; ay[j] = 0; az[j] = 0;
    __syncthreads();

    if (j != i) {
        double mj = m[j];
        if (type[j][0] == 'd') {
            mj = GRAVITY_DEVICE_MASS(mj, step * DT);
        }
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + EPS * EPS, 1.5);
            
        ax[j] = G * mj * dx / dist3;
        ay[j] = G * mj * dy / dist3;
        az[j] = G * mj * dz / dist3;
        // printf("%d %d %lf %lf %lf\n", step, blockIdx.x, ax[j], ay[j], az[j]);
    }


    __syncthreads();
    if(j == i) {
        for(int jj=0; jj<n; jj++) {
            if(jj==i) continue;
            ax[i] += ax[jj];
            ay[i] += ay[jj];
            az[i] += az[jj];
        }
    }

    // update velocities
    if(j == i) {
        // if(j == 0) printf("%d %e %e %e\n", step, ax[0], ay[0], az[0]);
        vx[i] += ax[i] * DT;
        vy[i] += ay[i] * DT;
        vz[i] += az[i] * DT;
        qx[i] += vx[i] * DT;
        qy[i] += vy[i] * DT;
        qz[i] += vz[i] * DT;
    }

    // if(j == 0) {
        
    // }
    // __syncthreads();

    // // update positions
    // for (int i = idx; i < n; i+=gridStride) {
    //     qx[i] += vx[i] * DT;
    //     qy[i] += vy[i] * DT;
    //     qz[i] += vz[i] * DT;
    // }

}



__global__ void run_step3(int step, int n, double* qx, double* qy,
    double* qz, double* vx, double* vy,
    double* vz, const double* m,
    char** type) {


       // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int gridStride = gridDim.x * blockDim.x;

    // compute accelerations
    __shared__ double ax[1024], ay[1024], az[1024];
    
    int i = blockIdx.x;
    int j = threadIdx.x;
    ax[j] = 0; ay[j] = 0; az[j] = 0;
    __syncthreads();

    if (j != i) {
        double mj = m[j];
        if (type[j][0] == 'd') {
            mj = GRAVITY_DEVICE_MASS(mj, step * DT);
        }
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + EPS * EPS, 1.5);
            
        ax[j] = G * mj * dx / dist3;
        ay[j] = G * mj * dy / dist3;
        az[j] = G * mj * dz / dist3;
        // printf("%d %d %lf %lf %lf\n", step, blockIdx.x, ax[j], ay[j], az[j]);
    }


    __syncthreads();
    if(j == i) {
        for(int jj=0; jj<n; jj++) {
            if(jj==i) continue;
            ax[i] += ax[jj];
            ay[i] += ay[jj];
            az[i] += az[jj];
        }
    }

    // update velocities
    if(j == i) {
        // if(j == 0) printf("%d %e %e %e\n", step, ax[0], ay[0], az[0]);
        vx[i] += ax[i] * DT;
        vy[i] += ay[i] * DT;
        vz[i] += az[i] * DT;
        qx[i] += vx[i] * DT;
        qy[i] += vy[i] * DT;
        qz[i] += vz[i] * DT;
    }
}

__global__ void MissileDistance(int step, int planet, int i, double* qx, double* qy, double* qz, double* m, double* i_cost) {
    if(m[i] == 0) return;

    double dx = qx[i] - qx[planet];
    double dy = qy[i] - qy[planet];
    double dz = qz[i] - qz[planet];

    if(sqrt(dx * dx + dy * dy + dz * dz) < step * DT * MISSILE_SPEED) {
        if(*i_cost > GET_MISSILE_COST(12000000)) {
            m[i] = 0;
            *i_cost = GET_MISSILE_COST(step*DT);
        }
    }
}

__global__ void MinDist(int planet, int asteroid, double* qx, double* qy, double* qz, double* min_dist) {
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    *min_dist = min(*min_dist, dx * dx + dy * dy + dz * dz);
}



__global__ void HitTimeStep(int step, int planet, int asteroid, double* qx, double* qy, double* qz, int* hit_time_step_msg) {
    if(hit_time_step_cuda != -2) return;
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    if (dx * dx + dy * dy + dz * dz < PLANET_RADIUS * PLANET_RADIUS) {
        hit_time_step_cuda = step;
        *hit_time_step_msg = step;
        //printf("***%d\n", step);
    }
}

__global__ void IgnoreDevices(int n, double* m, char** type) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;

    for(int i=idx; i<n; i+=gridStride) {
        // printf("%s\n", type[i]);
        if (type[i][0] == 'd') {
            m[i] = 0;
        } else {
            break;
        }
    }
}

// __global__ void testinput(int n,
//     double* qx, double* qy, double* qz,
//     double* vx, double* vy, double* vz,
//     double* m, char** type) {
//     for (int i = 0; i < n; i++) {
//         printf("%lf %lf %lf %lf %lf %lf %lf %s\n", qx[i], qy[i], qz[i], vx[i], vy[i], vz[i], m[i], type[i]);
//     }
// }

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;

    std::ifstream fin(argv[1]);
    fin >> n >> planet >> asteroid;

    int threadnum = n;
    // if(n <= 32) {
    //     threadnum = 32;
    // } else if(n <= 64) {
    //     threadnum = 64;
    // } else if(n <= 128) {
    //     threadnum = 128;
    // } else {
    //     threadnum = 256;
    // }

    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    char **type;

    cudaMallocHost(&qx, n * sizeof(double));
    cudaMallocHost(&qy, n * sizeof(double));
    cudaMallocHost(&qz, n * sizeof(double));
    cudaMallocHost(&vx, n * sizeof(double));
    cudaMallocHost(&vy, n * sizeof(double));
    cudaMallocHost(&vz, n * sizeof(double));
    cudaMallocHost(&m, n * sizeof(double));
    cudaMallocHost(&type, n * sizeof(char*));
    for(int i=0; i<n; i++) {
        char* typei;
        cudaMallocHost(&typei, 11 * sizeof(char));
        type[i] = typei;
    }

    read_input(fin, n, qx, qy, qz, vx, vy, vz, m, type);

    double *qx_cuda, *qy_cuda, *qz_cuda, *vx_cuda, *vy_cuda, *vz_cuda, *m_cuda;
    char **type_cuda, **type_cuda_tmp;

    cudaMalloc(&qx_cuda, n * sizeof(double));
    cudaMalloc(&qy_cuda, n * sizeof(double));
    cudaMalloc(&qz_cuda, n * sizeof(double));
    cudaMalloc(&vx_cuda, n * sizeof(double));
    cudaMalloc(&vy_cuda, n * sizeof(double));
    cudaMalloc(&vz_cuda, n * sizeof(double));
    cudaMalloc(&m_cuda, n * sizeof(double));
    cudaMalloc(&type_cuda, n * sizeof(char*));
    cudaMallocHost(&type_cuda_tmp, n * sizeof(char*));
    for(int i=0; i<n; i++) {
        char* typei;
        cudaMalloc(&typei, 11 * sizeof(char));
        type_cuda_tmp[i] = typei;
        cudaMemcpyAsync(type_cuda_tmp[i], type[i], 11 * sizeof(char), cudaMemcpyHostToDevice);
    }
    cudaMemcpyAsync(type_cuda, type_cuda_tmp, n * sizeof(char*), cudaMemcpyHostToDevice);

    cudaMemcpyAsync(qx_cuda, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(qy_cuda, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(qz_cuda, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vx_cuda, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vy_cuda, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vz_cuda, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(m_cuda, m, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // testinput<<<1,1>>>(n, qx_cuda, qy_cuda, qz_cuda, vx_cuda, vy_cuda, vz_cuda, m_cuda, type_cuda);

    // cudaDeviceSynchronize();

    // for (int i = 0; i < n; i++) {
    //     printf("%lf %lf %lf %lf %lf %lf %lf %s\n", qx[i], qy[i], qz[i], vx[i], vy[i], vz[i], m[i], type[i]);
    // }

    // cudaStream_t stream1, stream2, stream3;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    // cudaStreamCreate(&stream3);


    // Problem 1
    
    double *min_dist, *min_dist_cuda;
    cudaMallocHost(&min_dist, sizeof(double));
    cudaMalloc(&min_dist_cuda, sizeof(double));
    *min_dist = std::numeric_limits<double>::infinity();
    cudaMemcpyAsync(min_dist_cuda, min_dist, sizeof(double), cudaMemcpyHostToDevice);
    
    IgnoreDevices<<<1,threadnum>>>(n, m_cuda, type_cuda);

    for (int step = 0; step <= N_STEPS; step++) {
        //if(step % 10000 == 0) printf("Step: %d\n", step);
        if (step > 0) {
            run_step<<<threadnum, threadnum>>>(step, n, qx_cuda, qy_cuda, qz_cuda, vx_cuda, vy_cuda, vz_cuda, m_cuda, type_cuda);
            
            // cudaError_t err = cudaGetLastError();
            // if(err != cudaSuccess) { printf("%d CUDA Error: %s\n", step, cudaGetErrorString(err));}
            
        }
        MinDist<<<1,1>>>(planet, asteroid, qx_cuda, qy_cuda, qz_cuda, min_dist_cuda);
        // cudaError_t err = cudaGetLastError();
        // if(err != cudaSuccess) { printf("%d MINDIST CUDA Error: %s\n", step, cudaGetErrorString(err));}
        
    }

    cudaMemcpy(min_dist, min_dist_cuda, sizeof(double), cudaMemcpyDeviceToHost);

    // Problem 2

    int *hit_time_step, *hit_time_step_msg;
    cudaMallocHost(&hit_time_step, sizeof(int));
    cudaMalloc(&hit_time_step_msg, sizeof(int));
    *hit_time_step = -2;
    // cudaMemcpyAsync(&hit_time_step_cuda, hit_time_step, sizeof(int), cudaMemcpyHostToDevice);
    

    cudaMemcpyAsync(qx_cuda, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(qy_cuda, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(qz_cuda, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vx_cuda, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vy_cuda, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vz_cuda, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(m_cuda, m, n * sizeof(double), cudaMemcpyHostToDevice);

    for (int step = 0; step <= N_STEPS; step++) {
        if (step > 0) {
            run_step<<<threadnum, threadnum>>>(step, n, qx_cuda, qy_cuda, qz_cuda, vx_cuda, vy_cuda, vz_cuda, m_cuda, type_cuda);
        }

        HitTimeStep<<<1,1>>>(step, planet, asteroid, qx_cuda, qy_cuda, qz_cuda, hit_time_step_msg);
    }

    cudaMemcpy(hit_time_step, hit_time_step_msg, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaError_t err = cudaGetLastError();
    // if(err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err));}

    // Problem 3
    // TODO
    // cudaSetDevice(1);
    // err = cudaGetLastError();
    // if(err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err));}
    int gravity_device_id = -999;
    double missile_cost = -999;

    if(*hit_time_step == -2) {
        gravity_device_id = -1;
        missile_cost = 0;
    } else {    
        double *i_cost, *i_cost_cuda;
        double *missile_min_dist, *missile_min_dist_cuda;
        cudaMallocHost(&i_cost, sizeof(double));
        cudaMallocHost(&missile_min_dist, sizeof(double));
        cudaMalloc(&i_cost_cuda, sizeof(double));
        cudaMalloc(&missile_min_dist_cuda, sizeof(double));

        for(int i=n-1; i>=0; i--) {
            if(type[i][0] == 'd') {
                
                
                *i_cost = std::numeric_limits<double>::infinity();
                *missile_min_dist = std::numeric_limits<double>::infinity();

                cudaMemcpyAsync(i_cost_cuda, i_cost, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(missile_min_dist_cuda, missile_min_dist, sizeof(double), cudaMemcpyHostToDevice);

                cudaMemcpyAsync(qx_cuda, qx, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(qy_cuda, qy, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(qz_cuda, qz, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(vx_cuda, vx, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(vy_cuda, vy, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(vz_cuda, vz, n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(m_cuda, m, n * sizeof(double), cudaMemcpyHostToDevice);

                for (int step = 0; step <= N_STEPS; step++) {
                    if (step > 0) {
                        MissileDistance<<<1,1>>>(step, planet, i, qx_cuda, qy_cuda, qz_cuda, m_cuda, i_cost_cuda);
                        run_step3<<<threadnum,threadnum>>>(step, n, qx_cuda, qy_cuda, qz_cuda, vx_cuda, vy_cuda, vz_cuda, m_cuda, type_cuda);
                    }
                    MinDist<<<1,1>>>(planet, asteroid, qx_cuda, qy_cuda, qz_cuda, missile_min_dist_cuda);
                }

                cudaMemcpyAsync(i_cost, i_cost_cuda, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(missile_min_dist, missile_min_dist_cuda, sizeof(double), cudaMemcpyDeviceToHost);

                if(*missile_min_dist > PLANET_RADIUS * PLANET_RADIUS) {
                    //std::cout << "Device " << i << ",\n missile_min_dist = " << *missile_min_dist << '\n';
                    if(missile_cost > 0) {
                        if(*i_cost < missile_cost) {
                            gravity_device_id = i;
                            missile_cost = *i_cost;
                        }
                    } else {
                        gravity_device_id = i;
                        missile_cost = *i_cost;
                    }
                }
            } else {
                break;
            }
        }
    
        if(gravity_device_id == -999) {
            gravity_device_id = -1;
            missile_cost = 0;
        }
    }

    write_output(argv[2], sqrt(*min_dist), *hit_time_step, gravity_device_id, missile_cost);
}
