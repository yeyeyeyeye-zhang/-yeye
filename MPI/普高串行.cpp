#include<iostream>
#include<vector>
#include <stdlib.h>
#include<iomanip>
#include<time.h>
#include<pthread.h>
#include<chrono>//跨平台的高精度计时
#define NUM_THREADS 2//定义线程数量（需要能被n整除）
//待办：1.调试测试本文件，学习编程
//2.生成另外三个文件，书写报告
//3.性能测试结果处理
int n = 4096;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr=0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum=0;

//普通串行完整
void Gaussian_Elimination() {
  
   for (int k = 0; k < n-1; k++) {
       for (int i = k+1 ; i < n; i++) {
           double factor = A[i][k] / A[k][k];
           //std::cout << "factor=" << factor << std::endl;
           for (int j = k +1; j < n; j++) {
               A[i][j] -= factor * A[k][j];
           }
           A[i][k] = 0;
           b[i] -= factor * b[k];
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

int main() {
    double allDuration = 0;
    int count_ = 0;
	std::vector<double> x(n);
    while (allDuration < 0.5) {
        count_++;
        reset_Matrix(A, n);
        reset_vector(b, n);

        auto start = std::chrono::high_resolution_clock::now();

        Gaussian_Elimination();
		back(x);
		
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end - start;
        allDuration += duration2.count();
    }

    std::cout << "消去回代花费的平均时间(ms)是:"<<(allDuration/count_)*1000<< std::endl;

    return 0;
}
