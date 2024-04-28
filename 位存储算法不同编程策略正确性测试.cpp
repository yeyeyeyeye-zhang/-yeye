#include <iostream>
#include <vector>
#include <bitset>
#include <fstream>
#include <emmintrin.h>
#include<chrono>//跨平台的高精度计时（不只是windows）

using namespace std;

// 位向量大小
const int BITMAP_SIZE = 10;

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
void readData(vector<vector<int>>& invertedIndex, const string& filePath) {
    ifstream fin(filePath);
    int docId, wordId;
    while (fin >> docId >> wordId) {
        if (wordId >= invertedIndex.size()) {
            invertedIndex.resize(wordId + 1);
        }
        invertedIndex[wordId].push_back(docId);
    }
    fin.close();
}

// 将链表转换为位向量
bitset<BITMAP_SIZE> listToBitmap(const vector<int>& list) {
    bitset<BITMAP_SIZE> bitmap;
    for (int docId : list) {
        bitmap.set(docId);
    }
    return bitmap;
}

// 对两个位向量求交并返回结果
bitset<BITMAP_SIZE> bitmapAnd_SSE(const bitset<BITMAP_SIZE>& bitmap1, const bitset<BITMAP_SIZE>& bitmap2) {
    __m128i a = _mm_set1_epi32(bitmap1.to_ulong());//将参数 `bitmap1.to_ulong()` 的值复制四次，然后将这四个复制的值存储
    __m128i b = _mm_set1_epi32(bitmap2.to_ulong());
    /* __m128i a = _mm_set_epi32(0, 0, 0, bitmap1.to_ulong());
     __m128i b = _mm_set_epi32(0, 0, 0, bitmap1.to_ulong());*/
    __m128i result = _mm_and_si128(a, b);//同时对多位进行与运算
    unsigned long long lower = _mm_cvtsi128_si64(result);//将result的低64位转换成long long类型，赋值给lower
    return bitset<BITMAP_SIZE>(lower);
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


// 对多个位向量求交并返回结果
bitset<BITMAP_SIZE> bitmapAnd_SSE(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (int i = 1; i < bitmaps.size(); i++) {
        result = bitmapAnd_SSE(result, bitmaps[i]);
        //std::cout << "第" << i << "次求交后结果为：" << result << std::endl;
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
// 对多个位向量求交并返回结果
bitset<BITMAP_SIZE> bitmapAnd_straight(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (int i = 1; i < bitmaps.size(); i++) {
        result &= bitmaps[i];
       // result = bitmapAnd(result, bitmaps[i]);
        //std::cout << "第" << i << "次求交后结果为：" << result << std::endl;
    }
    return result;
}


int main() {
    //// 读取数据
    //vector<vector<int>> invertedIndex;
    //readData(invertedIndex, "data.txt");

    //// 将链表转换为位向量
    //vector<bitset<BITMAP_SIZE>> bitmaps;
    //for (const vector<int>& list : invertedIndex) {
    //    bitmaps.push_back(listToBitmap(list));
    //}

    //// 对两个位向量求交
    //bitset<BITMAP_SIZE> result = bitmapAnd(bitmaps[0], bitmaps[1]);

    //// 对多个位向量求交
    //result = bitmapAnd(bitmaps);

       // 生成测试数据
    std::vector<std::vector<int>> invertedIndex(BITMAP_SIZE);
    for (int i = 0; i < 5; i++) {
        invertedIndex[i] = { i,i + 1,i + 2,i + 3, i + 4,i + 5 };//wordID为i的docID有：i,i+4,i+8,i+12
    }
    for (int i = 5; i < 10; i++) {
        invertedIndex[i] = { i,i - 1,i - 2,i - 3, i - 4 };//wordID为i的docID有：i,i+4,i+8,i+12
    }
    //=======================================================================
    std::cout << "下面是串行算法的倒排索引求交过程：" << std::endl;
    // 测试未优化的链表转换为位向量
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps(BITMAP_SIZE);
    std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps[i] = listToBitmap(invertedIndex[i]);
        std::cout << "Word " << i << ": " << bitmaps[i] << std::endl;
    }

    //bitset<BITMAP_SIZE> result = bitmapAnd(bitmaps);
    //std::cout << "求交结果：" << result << std::endl;
    CalTime(bitmaps, bitmapAnd);
    std::cout << "===================================================================" << std::endl;
    std::cout << "下面是减小调用函数开销后的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps2(BITMAP_SIZE);
    std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps2[i] = listToBitmap(invertedIndex[i]);
        std::cout << "Word " << i << ": " << bitmaps2[i] << std::endl;
    }

    /*bitset<BITMAP_SIZE> result2 = bitmapAnd_SSE(bitmaps2);
    std::cout << "求交结果：" << result2 << std::endl;*/
    CalTime(bitmaps2, bitmapAnd_straight);
    std::cout << "===================================================================" << std::endl;
    std::cout << "下面是SSE2编程的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps3(BITMAP_SIZE);
    std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps3[i] = listToBitmap(invertedIndex[i]);
        std::cout << "Word " << i << ": " << bitmaps3[i] << std::endl;
    }

   /* bitset<BITMAP_SIZE> result3 = bitmapAnd_SSE_2(bitmaps3);
    std::cout << "求交结果：" << result3 << std::endl;*/
    CalTime(bitmaps3, bitmapAnd_SSE_2);
    std::cout << "===================================================================" << std::endl;
    std::cout << "下面是SSE2编程结合多线程的倒排索引求交过程：" << std::endl;
    std::vector<std::bitset<BITMAP_SIZE>> bitmaps4(BITMAP_SIZE);
    std::cout << "把单词映射文档的列表转换成位向量：" << std::endl;
    for (int i = 0; i < BITMAP_SIZE; i++) {
        bitmaps4[i] = listToBitmap(invertedIndex[i]);
        std::cout << "Word " << i << ": " << bitmaps4[i] << std::endl;
    }

   /* bitset<BITMAP_SIZE> result4 = bitmapAndOptimized(bitmaps4);
    std::cout << "求交结果：" << result4 << std::endl;*/
    CalTime(bitmaps4, bitmapAndOptimized);

    return 0;
}
