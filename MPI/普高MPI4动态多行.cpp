#include <mpi.h> // 引入MPI头文件
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <pthread.h>
#include <chrono> // 跨平台的高精度计时

int n = 1024; // 调整n，调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task; // 新增
int next_arr = 0; // 新增，在主函数中赋值为k（从第k+1行开始）
double sum = 0;

// 命名规则 MPIAB
// A代表代码版本，B代表不同规模

void Gaussian_Elimination(int myid, int numprocs) {
    int root;
    for (int k = 0; k < n; k++) {
        // 确定根进程
        root = k % numprocs;
        if (myid == root) { // 根进程执行除法操作
            b[k] = b[k] / A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        // 广播第 k 行的数据
        MPI_Bcast(&A[k][0], n, MPI_DOUBLE, root, MPI_COMM_WORLD);
        // 广播b[k]数据
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // 动态任务分配
        int task;
        while (true) {
            // 动态获取任务
            MPI_Win win;
            MPI_Win_create(&next_arr, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
            MPI_Win_fence(0, win);
            if (myid == 0) {
                task = next_arr;
                next_arr += 4; // 获取四个连续的行任务
            }
            MPI_Win_fence(0, win);
            MPI_Bcast(&task, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Win_free(&win);
            if (task >= n) {
                break;
            }

            // 处理从 task 开始的四行
            for (int i = task; i < task + 4 && i < n; i++) {
                if (i > k) { // 处理大于第k行部分的数据
                    double factor = A[i][k];
                    for (int j = k + 1; j < n; j++) {
                        A[i][j] -= factor * A[k][j];
                    }
                    A[i][k] = 0;
                    b[i] -= factor * b[k];
                }
            }
        }
    }
}

void back(std::vector<double>& x)
{
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
       double sum = b[i];
       for (int j = i + 1; j < n; j++) {
           sum -= A[i][j] * x[j];
       }
       x[i] = sum / A[i][i];
   }
}
void print_matrix(const std::vector<std::vector<double>>& A) {
    for (int m = 0; m < A.size(); m++) {
        for (int n = 0; n < A[m].size(); n++) {
            std::cout << std::left << std::setw(11) << A[m][n] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(const std::vector<double>& b) {
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

    for (int k = 0; k < n; k++) { // 相加的元素从第0行加到最后一行
        for (int i = k + 1; i < n; i++) { // 加到第k行之后的行上
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j]; // 不再是上三角矩阵，但通过行变换确保能变成上三角矩阵
            }
        }
    }
}

void reset_vector(std::vector<double>& b, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        b[i] = std::rand() % 99991;
    }
}

int main(int argc, char** argv) {
    std::vector<double> x(n);
    MPI_Init(&argc, &argv);

    int myid, numprocs; // 获取进程id和总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    double allDuration = 0;
    int count_ = 0;

    while (allDuration < 2) {
        count_++;
        reset_Matrix(A, n);
        reset_vector(b, n);
        next_arr = 0; // 重置任务指针

        auto start = std::chrono::high_resolution_clock::now();

        Gaussian_Elimination(myid, numprocs);
        if(myid==0){back(x);} 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end - start;
        allDuration += duration2.count();
    }

    if (myid == 0) { // 由进程0输出时间
        std::cout << "消去回代花费的平均时间(ms)是:" << (allDuration / count_) * 1000 << std::endl;
    }

    MPI_Finalize();
    return 0;
}
