#include <iostream>
#include <vector>
#include <bitset>
#include <fstream>
#include <emmintrin.h>
#include<chrono>//跨平台的高精度计时（不只是windows）
#include <arm_neon.h>

using namespace std;

//待办：
//1.不同问题规模，不同编程策略下的时间差异（修改取余的值，以及修改主函数）
//2.针对1中时间作表、作图
//3.用godbolt分析性能差异来源
//4.书写摘要、问题概述之子问题
//5.上传git代码并附上仓库链接
//5.尝试用Neon编程，并尝试登录华为鲲鹏平台
// 位向量大小
const int BITMAP_SIZE = 8192;//根据问题规模调整
int num_scale[4] = { 16,128,1024,8192 };//设置4种问题规模
int remainders[4] = { 13,127,1019,8191 };//给出规模内的最小质数，方便对数据集进行取余操作，控制数据的范围

void CalTime(const vector<bitset<BITMAP_SIZE>>& bitmaps, bitset<BITMAP_SIZE>(*f)(const vector<bitset<BITMAP_SIZE>>& bitmaps))
{
    double allDuration = 0.0;//多次执行代码的总时长
    double sum = 0.0;
    double _count = 0.0;//执行次数，为了后续的浮点计算设置为浮点数
    bitset<BITMAP_SIZE> result;
    while (allDuration < 1)//确保总时长大于1000ms,得到更精确的结果
    {
        _count++;
        //开始计时
        auto start = chrono::high_resolution_clock::now();
        result = f(bitmaps);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        allDuration += duration.count();
    }
    cout << allDuration / _count << "秒" << endl;
    std::cout << "求交结果：" << result << std::endl;
}

// 从文件中读取数据，并将数据存储在倒排索引中
void readData(vector<vector<int>>& invertedIndex, const string& filename,int n,int remainder) {//传入问题规模与取余的数
    std::fstream file;
    file.open(filename, ios::binary | ios::in);
    if (!file.is_open())
    {
        std::cout << "Error opening file" << filename << std::endl;
        return;
    }
    for (int i = 0; i < n; i++)//总共读取n个倒排链表（可以修改问题规模）
    {
        int numElements;  // 存储文件中的值，即元素个数
        file.read(reinterpret_cast<char*>(&numElements), sizeof(int));  // 从文件中读取值
        std::vector<int>* t = new std::vector<int>();  // 动态分配 vector 并设置大小为文件中的值
        for (int j = 0; j < numElements; j++)
        {
            unsigned int docId;
            file.read((char*)&docId, sizeof(docId));
            t->push_back(docId%remainder);//取余操作，控制数据集docId的范围
        }
        invertedIndex.push_back(*t);
        delete t;
    }
}

// 将链表转换为位向量
bitset<BITMAP_SIZE> listToBitmap(const vector<int>& list) {
    bitset<BITMAP_SIZE> bitmap;
    for (int docId : list) {
        bitmap.set(docId);
    }
    return bitmap;
}




bitset<BITMAP_SIZE> bitmapAnd_SSE_2(const bitset<BITMAP_SIZE>& bitmap1, const bitset<BITMAP_SIZE>& bitmap2) {
    __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(&bitmap1)); //直接从内存中加载一个 128 位的 SSE 寄存器
    __m128i b = _mm_load_si128(reinterpret_cast<const __m128i*>(&bitmap2));
    __m128i result = _mm_and_si128(a, b);
    unsigned long long lower = _mm_cvtsi128_si64(result);
    return bitset<BITMAP_SIZE>(lower);
}

// 对多个位向量求交并返回结果（优化3：并行化）
std::bitset<BITMAP_SIZE> bitmapAndOptimized(const std::vector<std::bitset<BITMAP_SIZE>>& bitmaps) {
    std::bitset<BITMAP_SIZE> result = bitmaps[0];

#pragma omp parallel for//告诉编译器将循环内的迭代分配到多个线程中执行，以实现并行计算
    for (int i = 1; i < bitmaps.size(); i++) {
#pragma omp critical//创建一个临界区，保证多个线程对共享资源的访问互斥，避免竞态条件和数据不一致
        result = bitmapAnd_SSE_2(result, bitmaps[i]);//先用改良版的SSE2
    }

    return result;
}



bitset<BITMAP_SIZE> bitmapAnd_SSE_2(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (int i = 1; i < bitmaps.size(); i++) {
        result = bitmapAnd_SSE_2(result, bitmaps[i]);
        //std::cout << "第" << i << "次求交后结果为：" << result << std::endl;
    }
    return result;
}

// 对两个位向量求交并返回结果
bitset<BITMAP_SIZE> bitmapAnd(const bitset<BITMAP_SIZE>& bitmap1, const bitset<BITMAP_SIZE>& bitmap2) {
    return bitmap1 & bitmap2;
}

// 对多个位向量求交并返回结果
bitset<BITMAP_SIZE> bitmapAnd(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (int i = 1; i < bitmaps.size(); i++) {
        //result &= bitmaps[i];
        result = bitmapAnd(result, bitmaps[i]);
        //std::cout << "第" << i << "次求交后结果为：" << result << std::endl;
    }
    return result;
}

bitset<BITMAP_SIZE> bitmapAnd_straight(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (int i = 1; i < bitmaps.size(); i++) {
        result &= bitmaps[i];//直接进行按位与操作，减少调用函数的开销
        //std::cout << "第" << i << "次求交后结果为：" << result << std::endl;
    }
    return result;
}


bitset<BITMAP_SIZE> bitmapAnd_Neon(const bitset<BITMAP_SIZE>& bitmap1, const bitset<BITMAP_SIZE>& bitmap2) {
    bitset<BITMAP_SIZE> result;

    // 拆分位向量为多个字向量
    uint64_t* resultPtr = reinterpret_cast<uint64_t*>(&result);
    uint64_t* bitmap1Ptr = reinterpret_cast<uint64_t*>(const_cast<bitset<BITMAP_SIZE>*>(&bitmap1));
    uint64_t* bitmap2Ptr = reinterpret_cast<uint64_t*>(const_cast<bitset<BITMAP_SIZE>*>(&bitmap2));

    uint64x2_t resultVec = vld1q_u64(resultPtr);
    uint64x2_t bitmap1Vec = vld1q_u64(bitmap1Ptr);
    uint64x2_t bitmap2Vec = vld1q_u64(bitmap2Ptr);

    // 使用NEON的位与运算指令求交并
    resultVec = vandq_u64(bitmap1Vec, bitmap2Vec);

    // 将结果存储回位向量
    vst1q_u64(resultPtr, resultVec);

    return result;
}

int main() {
  
    std::cout << "下面是串行算法的倒排索引求交过程：" << std::endl;
    // 读取数据
    vector<vector<int>> invertedIndex0;
    int n = num_scale[3];//调整不同规模传入
    int remainder = remainders[3];
    readData(invertedIndex0, "ExpIndex", n, remainder);
    // 测试未优化的链表转换为位向量
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps(BITMAP_SIZE);
    //std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps[i] = listToBitmap(invertedIndex0[i]);
        //std::cout << "Word " << i << ": " << bitmaps[i] << std::endl;
    }
    CalTime(bitmaps, bitmapAnd);//输出时间与结果
    std::cout << "===================================================================" << std::endl;


    std::cout << "下面是减小调用函数开销后的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps2(BITMAP_SIZE);
    //std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps2[i] = listToBitmap(invertedIndex0[i]);
        //std::cout << "Word " << i << ": " << bitmaps2[i] << std::endl;
    }

    CalTime(bitmaps2, bitmapAnd_straight);
    std::cout << "===================================================================" << std::endl;
    std::cout << "下面是SSE2编程的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps3(BITMAP_SIZE);
    //std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps3[i] = listToBitmap(invertedIndex0[i]);
        //std::cout << "Word " << i << ": " << bitmaps3[i] << std::endl;
    }

    CalTime(bitmaps3, bitmapAnd_SSE_2);
    std::cout << "===================================================================" << std::endl;
    std::cout << "下面是SSE2编程结合多线程的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps4(BITMAP_SIZE);
   // std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps4[i] = listToBitmap(invertedIndex0[i]);
        //std::cout << "Word " << i << ": " << bitmaps4[i] << std::endl;
    }

    CalTime(bitmaps4, bitmapAndOptimized);

    return 0;
}
