#include <iostream>
#include <chrono>
#include <cstdlib>

double CalTime(void(*f)())
{
    double allDuration = 0.0;//多次执行代码的总时长
    double _count = 0.0;//执行次数，为了后续的浮点计算设置为浮点数
    while (allDuration < 1)//确保总时长大于1000ms,得到更精确的结果
    {//一些初始化的基础操作
        _count++;
        //开始计时
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        allDuration += duration.count();
    }
    return allDuration / _count;
}


// 模拟分支冒险的函数
void branch_hazard() {
    // 假设 branch_var 的值不能在编译期确定
    volatile int branch_var = std::rand() % 2;
    volatile int value = 1;
    for (int i = 0; i < 100000; i++)
    {
        // 分支预测可能会失败的地方
        if (branch_var) {
            // 做一些操作，这些操作只有在 branch_var 为真时才会执行
            //__asm__ __volatile__("nop");//该指令只适用于华为鲲鹏服务平台（所以VS上跑不了）
            value++;
            //std::cout << "Branch taken" << std::endl;
        }
        else {
            // branch_var 为假时执行的代码
            value += 2;
            //std::cout << "Branch not taken" << std::endl;
        }
    }

}

// 模拟分支冒险的函数
void no_branch_hazard() {
    // 假设 branch_var 的值不能在编译期确定
    volatile int branch_var = std::rand() % 2;
    volatile int value = 1, count = 1;
    for (int i = 0; i < 100000; i++)
    {
        // 分支预测可能会失败的地方
        if (value) {
            // 做一些操作，这些操作只有在 branch_var 为真时才会执行
            count++;//该指令只适用于华为鲲鹏服务平台（所以VS上跑不了）
            //std::cout << "Branch taken" << std::endl;
        }
        else {
            // branch_var 为假时执行的代码
            count += 2;
            //std::cout << "Branch not taken" << std::endl;
        }
    }

}

// 模拟异常处理的函数
void exception_handling() {
    // 这里使用除零错误来产生异常
    volatile int a = 1;
    volatile int b = 0;
    volatile int count = 1;
    for (int i = 0; i < 100000; i++)
    {
        try {
            // 故意制造除零异常
            volatile int result = a / b;
            count++;
            //std::cout << result << std::endl; // 这一行不会被执行
        }
        catch (const std::exception& e) {
            // 处理异常
            count += 2;
            //std::cout << "Caught exception: division by zero" << std::endl;
        }
    }

}

// 模拟异常处理的函数
void no_exception_handling() {
    // 这里使用除零错误来产生异常
    volatile int a = 1;
    volatile int b = 0;
    volatile int count = 1;
    for (int i = 0; i < 100000; i++)
    {
        try {
            // 故意制造除零异常
            volatile int result = b / a;
            count++;
            //std::cout << result << std::endl; // 这一行不会被执行
        }
        catch (const std::exception& e) {
            // 处理异常
            count += 2;
            //std::cout << "Caught exception: division by zero" << std::endl;
        }
    }

}


int main() {


    double time1 = 1.0, time2 = 1.0;
    // 测试分支冒险
    std::cout << "存在分支冒险情况的运行时间：";
    time1 = CalTime(branch_hazard);
    std::cout << time1 << "秒" << std::endl;

    std::cout << "不存在分支冒险情况的运行时间：";
    time2 = CalTime(no_branch_hazard);
    std::cout << time2 << "秒" << std::endl;

    std::cout << "存在分支冒险/不存在分支冒险的时间比" << time1 / time2 << std::endl << std::endl;
    // 测试异常处理导致的控制冒险
    std::cout << "存在异常情况的运行时间：";
    time1 = CalTime(exception_handling);
    std::cout << time1 << "秒" << std::endl;

    std::cout << "不存在异常情况的运行时间：";
    time2 = CalTime(no_exception_handling);
    std::cout << time2 << "秒" << std::endl;
    std::cout << "存在异常/不存在异常的时间比" << time1 / time2 << std::endl << std::endl;
    std::cout << "------------------------------------" << std::endl;


    return 0;
}
