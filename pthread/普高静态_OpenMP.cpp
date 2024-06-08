#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <emmintrin.h>
#include <omp.h>

#define NUM_THREADS 8

int n = 1024;
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);

void reset_Matrix(std::vector<std::vector<double>>& A, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0;
        }
        A[i][i] = 1;
        for (int j = i + 1; j < n; j++) {
            A[i][j] = rand() % 99991;
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
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 99991;
    }
}

void back_SSE2(std::vector<double>& x) {
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        __m128d sum = _mm_set1_pd(b[i]);
        for (int j = i + 1; j < n; j += 2) {
            __m128d A_ij = _mm_loadu_pd(&A[i][j]);
            __m128d x_j = _mm_loadu_pd(&x[j]);
            __m128d prod = _mm_mul_pd(A_ij, x_j);
            sum = _mm_sub_pd(sum, prod);
        }
        double temp[2];
        _mm_storeu_pd(temp, sum);
        x[i] = (temp[0] + temp[1]) / A[i][i];
    }
}

int main() {
    double allDuration = 0;
    int count_ = 0;

    while (allDuration < 0.5) {
        count_++;
        
        reset_Matrix(A, n);
        reset_vector(b, n);

        auto start = std::chrono::high_resolution_clock::now();

        
        #pragma omp parallel num_threads(NUM_THREADS) // 创建常驻线程，仅此一次
        {
            for (int k = 0; k < n; ++k) {
                #pragma omp single
                {
                    for (int j = k + 1; j < n; ++j) {
                        A[k][j] /= A[k][k];
                    }
                    b[k] /= A[k][k];
                    A[k][k] = 1;
                }

                #pragma omp for // 划分给当前存在的线程
                for (int i = k + 1; i < n; ++i) {
                    __m128d A_ik, A_kj, product, result;
                    A_ik = _mm_load1_pd(&A[i][k]);

                    b[i] -= A[i][k] * b[k];
                    for (int j = k + 1; j + 1 < n; j += 2) {
                        A_kj = _mm_loadu_pd(&A[k][j]);
                        product = _mm_mul_pd(A_ik, A_kj);
                        result = _mm_loadu_pd(&A[i][j]);
                        result = _mm_sub_pd(result, product);
                        _mm_storeu_pd(&A[i][j], result);
                    }
                    // 如果 n 是奇数，那么处理最后一个元素
                    for (int j = (n - 1) & ~1; j < n; ++j) // 抹掉最低位，确保是偶数
                    {
                        A[i][j] -= A[i][k] * A[k][j];
                    }
                    A[i][k] = 0;
                }
            }
        }

        std::vector<double> x(n);
        back_SSE2(x);  

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        allDuration += duration.count();
    }

    std::cout << "消去回代花费的平均时间(ms)是:"<<(allDuration/count_)*1000<< std::endl;
    return 0;
}