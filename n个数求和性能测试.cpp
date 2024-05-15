#include<iostream>
#include<chrono>//跨平台的高精度计时（不只是windows）
using namespace std;
//任务分解
//1.向量内积算法√
//2.计时工具学习√
//3.n个数求和算法√
//4.设计向量内积规模与测试手段√
//5.n个数求和测试√
//6.学习书写目录和代码√
//7.撰写基础要求实验报告初稿√
//8.可视化输出结果√
//9.撰写基础要求实验报告终稿（源码链接+结果分析√+摘要√）
//10.采用循环展开技术√
//11.反汇编尝试
//12.进阶要求撰写1循环展开√
//13.进阶要求撰写2反汇编

//记得初始化sum=0,a[i]=0
double a[10000];

void CalTime(int n, double* a,int(*f)(int n, double* a))
{
	double allDuration = 0.0;//多次执行代码的总时长
	double sum = 0.0;
	double _count = 0.0;//执行次数，为了后续的浮点计算设置为浮点数
	while (allDuration < 1)//确保总时长大于1000ms,得到更精确的结果
	{//一些初始化的基础操作
		_count++;
		for (int j = 0; j < n; j++)a[j] = double(j);
		//开始计时
		auto start = chrono::high_resolution_clock::now();
		sum = f(n, a);
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> duration = end - start;
		allDuration += duration.count();
	}
	cout << allDuration / _count << "秒" << endl;
	cout << sum << endl;
}

int CommonAlg(int n,double* a)//逐一累加
{
	double sum = 0.0;
	for (int i = 0; i < n; i++)sum += a[i];
	return sum;
}

int OptimizedAlg1(int n,double* a)//普通的分成两路链式相加
{
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum = 0.0;
	for (int i = 0; i < n-1; i += 2)//调整了一下终止条件
	{
		sum1 += a[i];
		sum2 += a[i + 1];
	}
	sum = sum1 + sum2;
	return sum;
}

int OptimizedAlg2(int n, double* a)//递归,不断的折半相加，直至最后的a[0]为最终结果
{
	if (n == 1)return a[0];
	else
	{
		for (int i = 0; i < n / 2; i++)a[i] += a[n - i - 1];//首尾对应相加
		//把后半部分加到前半部分
		OptimizedAlg2(n / 2, a);
	}
	return 0;
}

int OptimizedAlg2New(int n, double* a)//递归,不过不是首尾加，而是连续的两个元素加
{
	if (n == 1)return a[0];
	else
	{
		for (int i = 0; i < n / 2; i++)a[i] = a[2 * i] + a[2 * i + 1];
		//把后半部分加到前半部分
		OptimizedAlg2(n / 2, a);
	}
	return 0;
}

int OptimizedAlg3(int n, double* a)//循环，折半相加（不过是连续的两个元素）
{
	for (int m = n; m > 1; m /= 2)//实现每轮的折半
	{
		for (int i = 0; i < m / 2; i++)a[i] = a[2 * i] + a[2 * i + 1];//连续两个元素相加
		//a0=a0+a1,a1=a2+a3,a2=a4+a5
		//两两元素相加的和存储在数组的前半部分
		//m=2时，a0=a0+a1,a0中存储有最终结果，跳出循环
	}
	return a[0];
}

int OptimizedAlg3New(int n, double* a)//循环，不过是首尾的两个元素相加
{
	for (int m = n; m > 1; m /= 2)//实现每轮的折半
	{
		for (int i = 0; i < m / 2; i++)a[i] += a[n - i - 1];
	}
	return a[0];
}

int UnrollAlg12(int n, double* a)
{
	double sum = 0.0;
	for (int i = 0; i < n; i += 2)
	{
		sum += a[i];
		sum += a[i + 1];//默认n的规模>3
	}
	return sum;
}

int UnrollAlg14(int n, double* a)
{
	double sum = 0.0;
	for (int i = 0; i < n; i += 4)
	{
		sum += a[i];
		sum += a[i + 1];//默认n的规模>5
		sum += a[i + 2];
		sum += a[i + 3];
	}
	return sum;
}

int UnrollAlg18(int n, double* a)
{
	double sum = 0.0;
	for (int i = 0; i < n; i += 8)
	{
		sum += a[i];
		sum += a[i + 1];//默认n的规模>=8
		sum += a[i + 2];
		sum += a[i + 3];
		sum += a[i + 4];
		sum += a[i + 5];
		sum += a[i + 6];
		sum += a[i + 7];
	}
	return sum;
}


int UnrollAlg44(int n, double* a)
{
	double sum = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
	for (int i = 0; i < n; i += 4)
	{
		sum1 += a[i];
		sum2 += a[i + 1];//默认n的规模>6
		sum3 += a[i + 2];
		sum4 += a[i + 3];
	}
	sum = sum1 + sum2 + sum3 + sum4;
	return sum;
}

int UnrollAlg88(int n, double* a)
{
	double sum = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0;
	double sum6 = 0.0, sum7 = 0.0, sum8 = 0.0;
	for (int i = 0; i < n; i += 8)
	{
		sum1 += a[i];
		sum2 += a[i + 1];//默认n的规模>=8
		sum3 += a[i + 2];
		sum4 += a[i + 3];
		sum5 += a[i + 4];
		sum6 += a[i + 5];
		sum7 += a[i + 6];
		sum8 += a[i + 7];
	}
	sum = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8;
	return sum;
}



int main()
{
	int scaleNum[] = { 8,128,1024,8192 };//设置4种问题规模
	for (int i = 0; i < 4; i++)//不同问题规模的开展
	{
		int n = scaleNum[i];
		//cout << "规模为" << n << "的平凡算法的平均执行时间是：";
		//CalTime(n, a, CommonAlg);
		//cout << "规模为" << n << "的两路链式算法的平均执行时间是：";
		//CalTime(n, a, OptimizedAlg1);
		//cout << "规模为" << n << "的递归首尾相加算法的平均执行时间是：";
		//CalTime(n, a, OptimizedAlg2);
		//cout << "规模为" << n << "的递归连续相加算法的平均执行时间是：";
		//CalTime(n, a, OptimizedAlg2New);
		//cout << "规模为" << n << "的循环首尾相加算法的平均执行时间是：";
		//CalTime(n, a, OptimizedAlg3New);
		//cout << "规模为" << n << "的循环连续相加算法的平均执行时间是：";
		//CalTime(n, a, OptimizedAlg3);
		cout << "规模为" << n << "的平凡算法(1x1)的平均执行时间是：";
		CalTime(n, a, CommonAlg);
		cout << "规模为" << n << "的两路链式算法(2x2循环展开)的平均执行时间是：";
		CalTime(n, a, OptimizedAlg1);
		cout << "规模为" << n << "的1x4循环展开算法的平均执行时间是：";
		CalTime(n, a, UnrollAlg14);
		cout << "规模为" << n << "的1x8循环展开算法的平均执行时间是：";
		CalTime(n, a, UnrollAlg18);
		cout << "规模为" << n << "的4x4循环展开算法的平均执行时间是：";
		CalTime(n, a, UnrollAlg44);
		cout << "规模为" << n << "的8x8循环展开算法的平均执行时间是：";
		CalTime(n, a, UnrollAlg88);
	}
	return 0;
}
