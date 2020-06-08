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
#define GRAVITY_DEVICE_MASS(M0, T) (M0 + 0.5 * M0 * fabs(sin(T/6000)))
#define PLANET_RADIUS ((double) 1e7)
#define MISSILE_SPEED ((double) 1e6)
#define GET_MISSILE_COST(T) (1e5 + 1e3 * T)

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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = GRAVITY_DEVICE_MASS(mj, step * DT);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + EPS * EPS, 1.5);
            ax[i] += G * mj * dx / dist3;
            ay[i] += G * mj * dy / dist3;
            az[i] += G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * DT;
        vy[i] += ay[i] * DT;
        vz[i] += az[i] * DT;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * DT;
        qy[i] += vy[i] * DT;
        qz[i] += vz[i] * DT;
    }
}

__global__ void testinput(int n,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    double* m, char** type) {
    for (int i = 0; i < n; i++) {
        printf("%lf %lf %lf %lf %lf %lf %lf %s\n", qx[i], qy[i], qz[i], vx[i], vy[i], vz[i], m[i], type[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;

    std::ifstream fin(argv[1]);
    fin >> n >> planet >> asteroid;

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
    
    testinput<<<1,1>>>(n, qx_cuda, qy_cuda, qz_cuda, vx_cuda, vy_cuda, vz_cuda, m_cuda, type_cuda);

    cudaDeviceSynchronize();

    // for (int i = 0; i < n; i++) {
    //     printf("%lf %lf %lf %lf %lf %lf %lf %s\n", qx[i], qy[i], qz[i], vx[i], vy[i], vz[i], m[i], type[i]);
    // }


    // // Problem 1
    // double min_dist = std::numeric_limits<double>::infinity();

    // std::vector<double> qx1(qx), qy1(qy), qz1(qz), vx1(vx), vy1(vy), vz1(vz), m1(m);
    // std::vector<std::string> type1(type);
    
    // #pragma omp parallel for
    // for (int i = n-1; i >= 0; i--) {
    //     if (type1[i] == "device") {
    //         m1[i] = 0;
    //     } else {
    //         break;
    //     }
    // }
    // for (int step = 0; step <= N_STEPS; step++) {
    //     if (step > 0) {
    //         run_step(step, n, qx1, qy1, qz1, vx1, vy1, vz1, m1, type1);
    //     }
    //     double dx = qx1[planet] - qx1[asteroid];
    //     double dy = qy1[planet] - qy1[asteroid];
    //     double dz = qz1[planet] - qz1[asteroid];
    //     min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    // }

    // // Problem 2
    // int hit_time_step = -2;

    // std::vector<double> qx2(qx), qy2(qy), qz2(qz), vx2(vx), vy2(vy), vz2(vz), m2(m);
    // std::vector<std::string> type2(type);

    // for (int step = 0; step <= N_STEPS; step++) {
    //     if (step > 0) {
    //         run_step(step, n, qx2, qy2, qz2, vx2, vy2, vz2, m2, type2);
    //     }
    //     double dx = qx2[planet] - qx2[asteroid];
    //     double dy = qy2[planet] - qy2[asteroid];
    //     double dz = qz2[planet] - qz2[asteroid];
    //     if (dx * dx + dy * dy + dz * dz < PLANET_RADIUS * PLANET_RADIUS) {
    //         hit_time_step = step;
    //         break;
    //     }
    // }

    // // Problem 3
    // // TODO
    // int gravity_device_id = -999;
    // double missile_cost = -999;

    // if(hit_time_step == -2) {
    //     gravity_device_id = -1;
    //     missile_cost = 0;
    // } else {    
    //     std::vector<double> qx3(qx), qy3(qy), qz3(qz), vx3(vx), vy3(vy), vz3(vz), m3(m);
    //     std::vector<std::string> type3(type);

    //     auto distance = [&](int i, int j) -> double {
    //         double dx = qx3[i] - qx3[j];
    //         double dy = qy3[i] - qy3[j];
    //         double dz = qz3[i] - qz3[j];
    //         return sqrt(dx * dx + dy * dy + dz * dz);
    //     };
        
    //     for(int i=n-1; i>=0; i--) {
    //         if(type3[i] == "device") {
    //             double i_cost = std::numeric_limits<double>::infinity();
    //             double missile_min_dist = std::numeric_limits<double>::infinity();
                
    //             if( i != (n-1) ) { qx3 = qx; qy3 = qy; qz3 = qz; vx3 = vx; vy3 = vy; vz3 = vz; m3 = m;}

    //             for (int step = 0; step <= N_STEPS; step++) {
    //                 if (step > 0) {
    //                     if(distance(planet, i) < step * DT * MISSILE_SPEED) {
    //                         if(i_cost == std::numeric_limits<double>::infinity()) {
    //                             m3[i] = 0;
    //                             i_cost = std::min(i_cost, GET_MISSILE_COST(step*DT));
    //                             std::cout << "step: " << step << "Distance :" << distance(planet, i) << " Missile travel: " << step * MISSILE_SPEED << " Cost = " << i_cost << ";\n";
    //                         }
    //                     }
    //                     run_step(step, n, qx3, qy3, qz3, vx3, vy3, vz3, m3, type3);
    //                 }
    //                 double dx = qx3[planet] - qx3[asteroid];
    //                 double dy = qy3[planet] - qy3[asteroid];
    //                 double dz = qz3[planet] - qz3[asteroid];
    //                 missile_min_dist = std::min(missile_min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    
    //                 if(missile_min_dist < PLANET_RADIUS) {
    //                     std::cout << "Device " << i << " break at step " << step << ",\n missile_min_dist = " << missile_min_dist << '\n';
    //                     break;
    //                 }
    //                 if(step == N_STEPS) {
    //                     if(missile_cost > 0) {
    //                         if(i_cost < missile_cost) {
    //                             gravity_device_id = i;
    //                             missile_cost = i_cost;
    //                         }
    //                     } else {
    //                         gravity_device_id = i;
    //                         missile_cost = i_cost;
    //                     }
    //                 }
    //             }
    //         } else {
    //             break;
    //         }
    //     }
    
    //     if(gravity_device_id == -999) {
    //         gravity_device_id = -1;
    //         missile_cost = 0;
    //     }
    // }

    // write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
