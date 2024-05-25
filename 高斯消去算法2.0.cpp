#include<iostream>
#include<vector>
#include <stdlib.h>
#include<iomanip>
#include<time.h>
#include<pthread.h>

int n = 8;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr=0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum=0;

struct threadParam_t
{
	int k_;//消去的轮次
	int t_id_;//线程id(动态分配线程没有用到)
};
void* threadFunc(void* param)
{
	threadParam_t* p = (threadParam_t*)param;//把void*转换成需要的类型
    //动态分配任务
	int k = p->k_;
    int task=0;
    while(1)
    {
        pthread_mutex_lock(&mutex_task);
        task=next_arr++;//获取第next_arr行的任务
        pthread_mutex_unlock(&mutex_task);
        if(task>=n)break;
		b[task]-=A[task][k]*b[k];
        for(int j=k+1;j<n;j++)
        {
            A[task][j]=A[task][j]-A[task][k]*A[k][j];
        }
        A[task][k]=0;
    }
    pthread_exit(NULL);

}
void* threadFunc2(void* param)
{
	threadParam_t* p = (threadParam_t*)param;//把void*转换成需要的类型
    //动态分配任务
	int i = p->k_;
    int task=0;
    while(1)
    {
        pthread_mutex_lock(&mutex_task);
        task=next_arr++;//获取第next_arr行的任务
        pthread_mutex_unlock(&mutex_task);
        if(task>=n)break;

       
    }
    pthread_exit(NULL);

}
// void* threadFunc(void* param)
// {
// 	threadParam_t* p = (threadParam_t*)param;//把void*转换成需要的类型
//     //动态分配任务
// 	int k = p->k_;
// 	int t_id = p->t_id_;
// 	int i = k + t_id + 1;//获取自己的计算任务（从第k+1行到第n行）->如果i>n如何处理？该任务的划分是逐行分配
// 	for (int j = k; j < n; ++j)
// 	{
// 		A[i][j] = A[i][j] - A[i][k] * A[k][j];//从第k列到第j列（针对第i行）
// 	}
// 	A[i][k] = 0;
// 	pthread_exit(NULL);
// }

//待办：1.静态编程 2.动态编程 3.测试样例 4.OpenMP编程
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
	print_matrix(A);
	std::cout << "上面是A矩阵变换成普通矩阵前,原始上三角矩阵的值" << std::endl;
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

int main()
{

	reset_Matrix(A, n);
	reset_vector(b, n);
	print_vector(b);
	std::cout << "上面是b向量的初始值" << std::endl;
	print_matrix(A);
	std::cout << "上面是A矩阵的初始值" << std::endl;
	for (int k = 0; k < n; k++)//最外层循环步骤之间数据有关联，无法进行多线程
	{
        next_arr=k+1;//起点行的位置
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];//此时第k行的第k列之后的元素都缩小了 A[k][k]倍
		}
		A[k][k] = 1;//对角线上的元素
		b[k]/= A[k][k];//行变换，也都缩小A[k][k]倍

		//创建工作线程，进行消去操作
		int worker_count = 7; //7
		pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * worker_count); //创建对应的handle（动态分配内存）
		threadParam_t* param = (threadParam_t*)malloc(sizeof(threadParam_t) * worker_count); //创建对应的线程数据结构（这样可以一起传参）

		// 分配任务
		for (int t_id = 0; t_id < worker_count; t_id++) {
			param[t_id].k_ = k;
			param[t_id].t_id_ = t_id;
		}

		// 创建线程
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
		}

		// 主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_join(handles[t_id], NULL);
		}
		print_matrix(A);
	    std::cout << "上面是消去步骤中k=" <<k<<"时,A矩阵的值"<< std::endl;
		//完成消去之后要进行回代

        free(handles);//释放内存
        free(param);
	}
	std::vector<double> x(n);
	x[n-1]=b[n-1]/A[n-1][n-1];
	//默认n的值都大于8,当内层n-i-1>7时再考虑创建线程
	for(int i=n-2;i>=n-8;i--)//外层循环彼此有勾连
	{
		sum=b[i];
		for(int j=i+1;j<n;j++)//内层循环可以考虑多线程编程
		{
			sum-=A[i][j]*x[j];
		}
		x[i]=sum/A[i][i];
	}
	for(int i=n-9;i>=0;i--)
	{
		sum=b[i];
		next_arr=i+1;
		pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * worker_count);
		threadParam_t* param = (threadParam_t*)malloc(sizeof(threadParam_t) * worker_count); 
		for (int t_id = 0; t_id < worker_count; t_id++) {
			param[t_id].k_ = i;//把i赋给k
			param[t_id].t_id_ = t_id;
		}
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_create(&handles[t_id], NULL, threadFunc2, (void*)&param[t_id]);
		}
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_join(handles[t_id], NULL);
		}

		x[i]=sum/A[i][i];
		free(handles);//释放内存
        free(param);
	}


	return 0;
}