#include<iostream>
#include<vector>
#include <stdlib.h>
#include<iomanip>
#include<time.h>
#include<pthread.h>
#include <arm_neon.h> // NEON头文件替换SSE2
#include<chrono>//跨平台的高精度计时
#define NUM_THREADS 8//定义线程数量（需要能被n整除）

int n = 1024;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr = 0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum = 0;

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

void print_matrix(std::vector<std::vector<double>>& A)
{
    for (int m = 0; m < A.size(); m++)
    {
        for (int n = 0; n < A[0].size(); n++)
        {
            std::cout << std::left << std::setw(11) << A[m][n] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(std::vector<double>& b)
{
    for (int m = 0; m < b.size(); m++)
    {
        std::cout << std::left << std::setw(6) << b[m] << " ";
    }
    std::cout << std::endl;
}

void reset_Matrix(std::vector<std::vector<double>>& A, int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            A[i][j] = 0;
        }
        A[i][i] = 1;
        for (int j = i + 1; j < n; j++)
        {
            A[i][j] = std::rand() % 99991;
        }
    }
    for (int k = 0; k < n; k++)//相加的元素从第0行加到最后一行
    {
        for (int i = k + 1; i < n; i++)//加到第k行之后的行上
        {
            for (int j = 0; j < n; j++) { A[i][j] += A[k][j]; }//不再是上三角矩阵，但通过行变换确保能变成上三角矩阵
        }
    }
}

void reset_vector(std::vector<double>& b, int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) { b[i] = std::rand() % 99991; }
}

int main() {
    double allDuration = 0;
    int count_ = 0;
    while (allDuration < 10) {
        count_++;
        reset_Matrix(A, n);
        reset_vector(b, n);

        auto start = std::chrono::high_resolution_clock::now();

        // 消去过程
        for (int k = 0; k < n; k++) {
            // 处理第k行，不并行
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            b[k] /= A[k][k];
            A[k][k] = 1;

            #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
            for (int i = k + 1; i < n; i++) {
                float64x2_t A_ik, A_kj, product, result;
                A_ik = vdupq_n_f64(A[i][k]);

                b[i] -= A[i][k] * b[k];
                for (int j = k + 1; j + 1 < n; j += 2) {
                    A_kj = vld1q_f64(&A[k][j]);
                    product = vmulq_f64(A_ik, A_kj);
                    result = vld1q_f64(&A[i][j]);
                    result = vsubq_f64(result, product);
                    vst1q_f64(&A[i][j], result);
                }
                 // 如果 n 是奇数，那么处理最后一个元素
                for (int j = (n - 1) & ~1; j < n; ++j) // 抹掉最低位，确保是偶数
                {
                    A[i][j] -= A[i][k] * A[k][j];
                }
                A[i][k] = 0;
            }
        }

        // 回代过程
        std::vector<double> x2(n);
        back_NEON(x2);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end - start;
        allDuration += duration2.count();
    }

    std::cout << "消去回代花费的平均时间(ms)是:" << (allDuration / count_) * 1000 << std::endl;

    return 0;
}