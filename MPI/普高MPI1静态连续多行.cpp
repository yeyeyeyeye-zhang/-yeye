#include <mpi.h>//引入MPI头文件
#include<iostream>
#include<vector>
#include <stdlib.h>
#include<iomanip>
#include<time.h>
#include<pthread.h>
#include<chrono>//跨平台的高精度计时

//记得加上回代部分（由进程0完成）的代码进行性能测试

int n = 4096;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr=0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum=0;

//命名规则 MPIAB
//A代表代码版本，B代表不同规模

void Gaussian_Elimination(int myid, int numprocs) {
    //std::cout<<"myid:"<<myid<<" ;numprocs"<<numprocs<<std::endl;
    int nrows = n / numprocs;  // 每个进程处理的行数
    int r1 = myid * nrows;//该进程处理的起始行号
    int r2 = (myid + 1) * nrows - 1;//该进程处理的终止行号
    int root;

    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {//处理的行数包括第k行的进程，需要执行除法操作，并把结果传给其他的进程
            root=myid;
            b[k]=b[k]/A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j]/A[k][k];
            }
            A[k][k] = 1.0;
            //std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I begin to send"<<std::endl;

            for (int j = 0; j < numprocs; j++) {
                if(j!=root)
                {
                    MPI_Send(&A[k][0], n, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
                    MPI_Send(&b[k], 1, MPI_DOUBLE, j, 1, MPI_COMM_WORLD);
                }
            }
            //std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I end sending"<<std::endl;
        } 
        else {//其他进程接收数据
            //std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I begin to receive"<<std::endl;
            MPI_Recv(&A[k][0], n, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&b[k], 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I end reciving"<<std::endl;
        }
        // std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I begin to receive"<<std::endl;
        // MPI_Recv(&A[k][0], n, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // std::cout<<"k="<<k<<";I'm myid"<<myid<<"; I end reciving"<<std::endl;

        for (int i = r1; i <= r2&&i<n; i++) {
            if (i > k) {//只处理大于第k行部分的数据
                double factor = A[i][k];
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j]-factor * A[k][j];
                }
                A[i][k] = 0;
                b[i] =b[i]-factor * b[k];
            }
        }
        //std::cout<<"end-k"<<std::endl;
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
	// print_matrix(A);
	// std::cout << "上面是A矩阵变换成普通矩阵前,原始上三角矩阵的值" << std::endl;
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


int main(int argc, char** argv) {
    //std::cout<<"begin"<<std::endl;
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
        if(myid==0){back(x);} 

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
