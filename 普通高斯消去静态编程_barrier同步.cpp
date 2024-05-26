#include<iostream>
#include<vector>
#include <stdlib.h>
#include<iomanip>
#include<time.h>
#include<pthread.h>
#include <emmintrin.h>
#include<semaphore.h>//引入信号量
#include<chrono>//跨平台的高精度计时
#define NUM_THREADS 4//定义线程数量（需要能被n整除）
//待办：1.debug:出现了不断运行的情况（可能存在死锁）
//2.实验报告对应部分
//3.测试表格

int n = 8;//调整n,调整问题规模（矩阵大小）
std::vector<std::vector<double>> A(n, std::vector<double>(n));
std::vector<double> b(n);
pthread_mutex_t mutex_task;//新增
int next_arr=0;//新增，在主函数中赋值为k（从第k+1行开始）
double sum=0;

struct threadParam_t
{
	int t_id_;//线程id(动态分配线程没有用到)
};

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

void* threadFunc(void* param)
{
    threadParam_t *p=(threadParam_t*)param;
    int t_id=p->t_id_;
    for(int k=0;k<n;++k)
    {
        if(t_id==0)
        {
            b[k]=b[k]/A[k][k];
            for(int j=k+1;j<n;j++)
            {
                A[k][j]= A[k][j]/A[k][k];
            }
            A[k][k]=1;
        }
       //第一个同步点
        pthread_barrier_wait(&barrier_Divsion);
        for(int i=k+1+t_id;i<n;i+=NUM_THREADS)
        {
            b[i]=b[i]-A[i][k]*b[k];
            for(int j=k+1;j<n;++j)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
           
        }
         // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
}

void* threadFunc_SSE2(void* param)
{
    threadParam_t *p=(threadParam_t*)param;
    int t_id=p->t_id_;
    for(int k=0;k<n;++k)
    {
        if(t_id==0)
        {
            b[k]=b[k]/A[k][k];
            for(int j=k+1;j<n;j++)
            {
                A[k][j]= A[k][j]/A[k][k];
            }
            A[k][k]=1;
        }
       //第一个同步点
        pthread_barrier_wait(&barrier_Divsion);
                 __m128d A_ik, A_kj, product, result;
        //循环划分任务
        for(int i=k+1+t_id;i<n;i+=NUM_THREADS)//行数（确定了行数之后，可以自己跑自己的）
        {
            A_ik = _mm_load1_pd(&A[i][k]); // 加载 A[i][k] 至两个双精度位置
            b[i]=b[i]-A[i][k]*b[k];
            for (int j = k + 1; j + 1 < n; j += 2) // 用 SSE2 对两个相邻元素进行操作
            {
                A_kj = _mm_loadu_pd(&A[k][j]); // 加载 A[k][j] 和 A[k][j+1]
                product = _mm_mul_pd(A_ik, A_kj); // 计算 A[i][k] * A[k][j] 和 A[i][k] * A[k][j+1]
                
                result = _mm_loadu_pd(&A[i][j]); // 加载 A[i][j] 和 A[i][j+1]
                result = _mm_sub_pd(result, product); // 计算 A[i][j] - A[i][k] * A[k][j]
                _mm_storeu_pd(&A[i][j], result); // 存储结果
            }

            // 如果 n 是奇数，那么处理最后一个元素
            for (int j = (n - 1) & ~1; j < n; ++j) // 抹掉最低位，确保是偶数
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0; // 已经被消去的行列元素设置为0
        }
         // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
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

void back_SSE2(std::vector<double>& x) {
    x[n-1] = b[n-1] / A[n-1][n-1];
    for(int i = n - 2; i >= 0; --i) {//要求n是偶数
        __m128d sum = _mm_set1_pd(b[i]);
        for(int j = i + 1; j < n; j += 2) {
            __m128d A_ij = _mm_loadu_pd(&A[i][j]);  // 加载A[i][j] 和 A[i][j+1]
            __m128d x_j = _mm_loadu_pd(&x[j]);      // 加载x[j] 和 x[j+1]
            __m128d prod = _mm_mul_pd(A_ij, x_j);   // 计算A[i][j] * x[j] 和 A[i][j+1] * x[j+1]
            sum = _mm_sub_pd(sum, prod);            // 累积结果
        }
        // 将累积结果合并
        double temp[2];
        _mm_storeu_pd(temp, sum);
        x[i] = (temp[0] + temp[1]) / A[i][i];
    }
}

int main()
{
    double allDuration=0;
	int count_=0;
	while(allDuration<0.5)
	{
		count_++;
        reset_Matrix(A, n);
        reset_vector(b, n);

        // print_vector(b);
        // std::cout << "上面是b向量的初始值" << std::endl;
        // print_matrix(A);
        // std::cout << "上面是A矩阵的初始值" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        //消去过程
        pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
        pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

        //创建线程
        pthread_t handles[NUM_THREADS];
        threadParam_t param[NUM_THREADS];
        for(int t_id=0;t_id<NUM_THREADS;t_id++)
        {
            param[t_id].t_id_ = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_SSE2, (void*)&param[t_id]);
            //pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }

    
        for(int t_id=0;t_id<NUM_THREADS;++t_id)
        {
            pthread_join(handles[t_id], NULL);//等待所有线程退出
        }

        pthread_barrier_destroy(&barrier_Divsion);
        pthread_barrier_destroy(&barrier_Elimination);


        //回代过程
        // print_vector(b);
        // std::cout << "上面是b向量消去过程结束后的值" << std::endl;
        std::vector<double> x(n);
        std::vector<double> x2(n);
        back_SSE2(x2);
        // print_vector(x2);
        // std::cout << "上面是SSE2编程消去得到的结果向量x的值"<< std::endl;
        x[n - 1] = b[n - 1] / A[n - 1][n - 1];
        for (int i = n - 2; i >= 0; i--) {
            double sum = b[i];
            for (int j = i + 1; j < n; j++) {
                sum -= A[i][j] * x[j];
            }
            x[i] = sum / A[i][i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration2 = end - start;
        allDuration += duration2.count();
        // print_vector(x);
        // std::cout << "上面是串行消去得到的结果向量x的值"<< std::endl;

	}
	std::cout << "消去回代花费的平均时间(s)是:"<<allDuration/count_<< std::endl;




	return 0;
}

