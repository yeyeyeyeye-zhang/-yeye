// 将链表转换为位向量
bitset<BITMAP_SIZE> listToBitmap(const vector<int>& list) {
    bitset<BITMAP_SIZE> bitmap;
    for (int docId : list) {
        bitmap.set(docId);
    }
    return bitmap;
}

bitset<BLOCK_SIZE> level2Index(const bitset<BITMAP_SIZE>& bitmap) {
    bitset<BLOCK_SIZE> index;
    for (size_t i = 0; i < BITMAP_SIZE; i += BLOCK_SIZE) {
        bool is_nonzero = bitmap.to_ullong() != 0;
        index.set(i / BLOCK_SIZE, is_nonzero);
    }
    return index;
}

bitset<BITMAP_SIZE> bitmapAnd_withLevel2Index(const bitset<BITMAP_SIZE>& index1, const bitset<BITMAP_SIZE>& index2, const bitset<BITMAP_SIZE>& bitmap1, const bitset<BITMAP_SIZE>& bitmap2) {
    bitset<BITMAP_SIZE> result;
    for (size_t i = 0; i < LEVEL2_INDEX_SIZE; i++) {
        if (index1[i] && index2[i]) {
            size_t offset = i * BLOCK_SIZE;
            for (size_t j = 0; j < BLOCK_SIZE; j += 128) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&bitmap1) + (offset / 128));
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&bitmap2) + (offset / 128));
                __m128i res = _mm_and_si128(a, b);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&result) + (offset / 128), res);
            }
        }
    }
    return result;
}

bitset<BITMAP_SIZE> bitmapAnd_withLevel2Index(const vector<bitset<BITMAP_SIZE>>& bitmaps) {
    if (bitmaps.empty()) {
        return bitset<BITMAP_SIZE>();
    }

    // 制作二级索引
    vector<bitset<BLOCK_SIZE>> indexes;
    for (const auto& bitmap : bitmaps) {
        indexes.push_back(level2Index(bitmap));
    }

    // 应用二级索引进行位图求交
    bitset<BITMAP_SIZE> result = bitmaps[0];
    for (size_t i = 1; i < bitmaps.size(); i++) {
        result = bitmapAnd_withLevel2Index(indexes[0], indexes[i], result, bitmaps[i]);
        //cout << "第" << i << "次求交后结果为：" << result << endl;
    }
    return result;
}
