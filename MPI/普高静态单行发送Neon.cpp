#include <mpi.h>//引入MPI头文件
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <pthread.h>
#include <chrono>//跨平台的高精度计时
#include <arm_neon.h> // 引入NEON头文件

int n = 4096;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr=0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum=0;

// 命名规则 MPIAB
// A代表代码版本，B代表不同规模
void Gaussian_Elimination(int myid, int numprocs) {
    int root;

    for (int k = 0; k < n; k++) {
        root = k % numprocs; // 确定根进程

        if (myid == root) { //根进程执行除法操作，并把结果传给其他的进程
            b[k] = b[k] / A[k][k];
            float64x2_t denom = vdupq_n_f64(A[k][k]);

            // 使用NEON进行除法操作
            int j;
            for (j = k + 1; j <= n - 2; j += 2) {
                float64x2_t a_line = vld1q_f64(&A[k][j]);
                float64x2_t result = vdivq_f64(a_line, denom);
                vst1q_f64(&A[k][j], result);
            }
            for (; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        // 广播第 k 行的数据
        MPI_Bcast(&A[k][0], n, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // 每个进程处理分配给它的行
        for (int i = myid; i < n; i += numprocs) {
            if (i > k) { // 只处理大于第k行部分的数据
                double factor = A[i][k];
                float64x2_t factor_simd = vdupq_n_f64(factor);

                // 使用NEON进行向量减法操作
                int j;
                for (j = k + 1; j <= n - 2; j += 2) {
                    float64x2_t a_line = vld1q_f64(&A[i][j]);
                    float64x2_t factor_a_line = vmulq_f64(factor_simd, vld1q_f64(&A[k][j]));
                    float64x2_t result = vsubq_f64(a_line, factor_a_line);
                    vst1q_f64(&A[i][j], result);
                }
                for (; j < n; j++) {
                    A[i][j] = A[i][j] - factor * A[k][j];
                }
                A[i][k] = 0;
                b[i] = b[i] - factor * b[k];
            }
        }
    }
}
void back_NEON(std::vector<double>& x) {
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        float64x2_t sum = vdupq_n_f64(b[i]);
        for (int j = i + 1; j < n; j += 2) {
            float64x2_t A_ij = vld1q_f64(&A[i][j]);  // 加载A[i][j] 和 A[i][j+1]
            float64x2_t x_j = vld1q_f64(&x[j]);      // 加载x[j] 和 x[j+1]
            float64x2_t prod = vmulq_f64(A_ij, x_j); // 计算A[i][j] * x[j] 和 A[i][j+1] * x[j+1]
            sum = vsubq_f64(sum, prod);              // 累积结果
        }
        double temp[2];
        vst1q_f64(temp, sum); // 将累积结果合并
        x[i] = (temp[0] + temp[1]) / A[i][i];
    }
}

void print_matrix(std::vector<std::vector<double>>& A) {
    for (int m = 0; m < A.size(); m++) {
        for (int n = 0; n < A[0].size(); n++) {
            std::cout << std::left << std::setw(11) << A[m][n] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(std::vector<double>& b) {
    for (int m = 0; m < b.size(); m++) {
        std::cout << std::left << std::setw(6) << b[m] << " ";
    }
    std::cout << std::endl;
}

void reset_Matrix(std::vector<std::vector<double>>& A, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0;
        }
        A[i][i] = 1;
        for (int j = i + 1; j < n; j++) {
            A[i][j] = std::rand() % 99991;
        }
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
            }
        }
    }
}

void reset_vector(std::vector<double>& b, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) { b[i] = std::rand() % 99991; }
}

int main(int argc, char** argv) {
    std::vector<double> x(n);
    MPI_Init(&argc, &argv);

    int myid, numprocs;//获取进程id和总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    double allDuration = 0;
    int count_ = 0;

    while (allDuration < 3) {
        count_++;
        reset_Matrix(A, n);
        reset_vector(b, n);

        auto start = std::chrono::high_resolution_clock::now();

        Gaussian_Elimination(myid, numprocs);
        if (myid == 0) { back_NEON(x); }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end - start;
        allDuration += duration2.count();
    }

    if (myid == 0) {//由进程0输出时间
        std::cout << "消去回代花费的平均时间(ms)是:" << (allDuration / count_) * 1000 << std::endl;
    }

    MPI_Finalize();
    return 0;
}
