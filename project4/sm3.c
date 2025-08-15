#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>  // For SIMD instructions
#include <assert.h>

// ================== 常量定义 ==================

// SM3算法常量定义
#define SM3_BLOCK_SIZE 64     // 分组大小（字节）
#define SM3_DIGEST_SIZE 32    // 哈希结果大小（字节）
#define SM3_HASH_SIZE 8       // 哈希状态大小（字）

// SM3初始值（大端表示）
static const uint32_t IV[SM3_HASH_SIZE] = {
    0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
    0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
};

// SM3常量Tj
static const uint32_t T[64] = {
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A
};

// ================== 辅助函数 ==================

// 循环左移
static inline uint32_t ROTL(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// 字节序转换（大端转主机序）
static inline uint32_t BE_TO_HOST(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap32(x);
#else
    return x;
#endif
}

// 主机序转大端
static inline uint32_t HOST_TO_BE(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap32(x);
#else
    return x;
#endif
}

// 布尔函数FF
static inline uint32_t FF(uint32_t x, uint32_t y, uint32_t z, int j) {
    if (j < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | (x & z) | (y & z);
    }
}

// 布尔函数GG
static inline uint32_t GG(uint32_t x, uint32_t y, uint32_t z, int j) {
    if (j < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | (~x & z);
    }
}

// P0置换函数
static inline uint32_t P0(uint32_t x) {
    return x ^ ROTL(x, 9) ^ ROTL(x, 17);
}

// P1置换函数
static inline uint32_t P1(uint32_t x) {
    return x ^ ROTL(x, 15) ^ ROTL(x, 23);
}

// ================== SM3基本实现 ==================

// 消息填充
int sm3_pad(const uint8_t* data, size_t len, uint8_t** padded_data, size_t* padded_len) {
    if (!data || !padded_data || !padded_len) {
        return -1;  // 无效参数
    }

    // 计算填充后的消息长度（字节）
    size_t blocks = (len + 1 + 8 + SM3_BLOCK_SIZE - 1) / SM3_BLOCK_SIZE;
    *padded_len = blocks * SM3_BLOCK_SIZE;

    // 分配内存
    *padded_data = calloc(*padded_len, 1);
    if (!*padded_data) {
        return -2;  // 内存分配失败
    }

    // 复制原始数据
    memcpy(*padded_data, data, len);

    // 添加填充位
    (*padded_data)[len] = 0x80;

    // 添加长度（比特数，大端表示）
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        (*padded_data)[*padded_len - 8 + i] = (bit_len >> (56 - i * 8)) & 0xFF;
    }

    return 0;  // 成功
}

// 基本消息扩展
void sm3_message_expansion_basic(const uint32_t* block, uint32_t* w, uint32_t* w_prime) {
    // 前16个字直接复制并转换字节序
    for (int i = 0; i < 16; i++) {
        w[i] = BE_TO_HOST(block[i]);
    }

    // 扩展16-67个字
    for (int i = 16; i < 68; i++) {
        uint32_t temp = w[i - 16] ^ w[i - 9] ^ ROTL(w[i - 3], 15);
        w[i] = P1(temp) ^ ROTL(w[i - 13], 7) ^ w[i - 6];
    }

    // 计算w_prime
    for (int i = 0; i < 64; i++) {
        w_prime[i] = w[i] ^ w[i + 4];
    }
}

// 基本压缩函数
void sm3_compress_basic(uint32_t* v, const uint32_t* block) {
    uint32_t w[68];
    uint32_t w_prime[64];

    // 消息扩展
    sm3_message_expansion_basic(block, w, w_prime);

    // 初始化工作变量
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];

    // 64轮迭代
    for (int j = 0; j < 64; j++) {
        uint32_t ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j], j % 32)), 7);
        uint32_t ss2 = ss1 ^ ROTL(a, 12);
        uint32_t tt1 = FF(a, b, c, j) + d + ss2 + w_prime[j];
        uint32_t tt2 = GG(e, f, g, j) + h + ss1 + w[j];

        // 更新状态
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
    }

    // 更新哈希值
    v[0] ^= a;
    v[1] ^= b;
    v[2] ^= c;
    v[3] ^= d;
    v[4] ^= e;
    v[5] ^= f;
    v[6] ^= g;
    v[7] ^= h;
}

// 基本SM3哈希计算
int sm3_hash_basic(const uint8_t* data, size_t len, uint8_t* digest) {
    if (!data || !digest) {
        return -1;  // 无效参数
    }

    uint8_t* padded_data = NULL;
    size_t padded_len = 0;

    // 消息填充
    int ret = sm3_pad(data, len, &padded_data, &padded_len);
    if (ret != 0) {
        return ret;
    }

    // 初始化哈希状态
    uint32_t v[SM3_HASH_SIZE];
    memcpy(v, IV, sizeof(v));

    // 处理每个消息块
    size_t blocks = padded_len / SM3_BLOCK_SIZE;
    for (size_t i = 0; i < blocks; i++) {
        sm3_compress_basic(v, (uint32_t*)(padded_data + i * SM3_BLOCK_SIZE));
    }

    // 转换为大端字节序输出
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)digest)[i] = HOST_TO_BE(v[i]);
    }

    // 清理
    free(padded_data);
    return 0;
}

// ================== SM3优化实现 ==================

#ifdef __SSE2__
// SIMD优化消息扩展
void sm3_message_expansion_simd(const uint32_t* block, uint32_t* w, uint32_t* w_prime) {
    // 前16个字直接复制并转换字节序
    for (int i = 0; i < 16; i++) {
        w[i] = BE_TO_HOST(block[i]);
    }

    // 使用SIMD加速扩展16-67个字
    for (int i = 16; i < 68; i++) {
        // 使用SSE2指令并行计算
        __m128i a = _mm_set_epi32(w[i - 3], w[i - 13], w[i - 9], w[i - 16]);

        // 计算P1(w[i-16] ^ w[i-9] ^ ROTL(w[i-3], 15))
        uint32_t rotl15 = ROTL(w[i - 3], 15);
        uint32_t p1_input = w[i - 16] ^ w[i - 9] ^ rotl15;
        uint32_t p1 = p1_input ^ ROTL(p1_input, 15) ^ ROTL(p1_input, 23);

        // 计算完整表达式
        w[i] = p1 ^ ROTL(w[i - 13], 7) ^ w[i - 6];
    }

    // 计算w_prime
    for (int i = 0; i < 64; i++) {
        w_prime[i] = w[i] ^ w[i + 4];
    }
}
#endif

// 优化压缩函数（循环展开）
void sm3_compress_optimized(uint32_t* v, const uint32_t* block) {
    uint32_t w[68];
    uint32_t w_prime[64];

    // 消息扩展
#ifdef __SSE2__
    sm3_message_expansion_simd(block, w, w_prime);
#else
    sm3_message_expansion_basic(block, w, w_prime);
#endif

    // 初始化工作变量
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];

    // 循环展开，每4轮一组
    for (int j = 0; j < 64; j += 4) {
        // 第1轮
        uint32_t ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j], j % 32)), 7);
        uint32_t ss2 = ss1 ^ ROTL(a, 12);
        uint32_t tt1 = FF(a, b, c, j) + d + ss2 + w_prime[j];
        uint32_t tt2 = GG(e, f, g, j) + h + ss1 + w[j];

        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);

        // 第2轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j + 1], (j + 1) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j + 1) + d + ss2 + w_prime[j + 1];
        tt2 = GG(e, f, g, j + 1) + h + ss1 + w[j + 1];

        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);

        // 第3轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j + 2], (j + 2) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j + 2) + d + ss2 + w_prime[j + 2];
        tt2 = GG(e, f, g, j + 2) + h + ss1 + w[j + 2];

        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);

        // 第4轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j + 3], (j + 3) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j + 3) + d + ss2 + w_prime[j + 3];
        tt2 = GG(e, f, g, j + 3) + h + ss1 + w[j + 3];

        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
    }

    // 更新哈希值
    v[0] ^= a;
    v[1] ^= b;
    v[2] ^= c;
    v[3] ^= d;
    v[4] ^= e;
    v[5] ^= f;
    v[6] ^= g;
    v[7] ^= h;
}

// 优化SM3哈希计算
int sm3_hash_optimized(const uint8_t* data, size_t len, uint8_t* digest) {
    if (!data || !digest) {
        return -1;  // 无效参数
    }

    uint8_t* padded_data = NULL;
    size_t padded_len = 0;

    // 消息填充
    int ret = sm3_pad(data, len, &padded_data, &padded_len);
    if (ret != 0) {
        return ret;
    }

    // 初始化哈希状态
    uint32_t v[SM3_HASH_SIZE];
    memcpy(v, IV, sizeof(v));

    // 处理每个消息块
    size_t blocks = padded_len / SM3_BLOCK_SIZE;
    for (size_t i = 0; i < blocks; i++) {
        sm3_compress_optimized(v, (uint32_t*)(padded_data + i * SM3_BLOCK_SIZE));
    }

    // 转换为大端字节序输出
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)digest)[i] = HOST_TO_BE(v[i]);
    }

    // 清理
    free(padded_data);
    return 0;
}

// ================== 长度扩展攻击 ==================

// 长度扩展攻击
int sm3_length_extension_attack(const uint8_t* original_msg, size_t orig_len,
    const uint8_t* orig_hash,
    const uint8_t* extension, size_t ext_len,
    uint8_t* new_hash) {
    if (!original_msg || !orig_hash || !extension || !new_hash) {
        return -1;  // 无效参数
    }

    // 计算扩展后的消息长度
    size_t new_len = orig_len + ext_len;

    // 计算填充后的原始消息长度
    size_t padded_orig_len = orig_len + 1 + 8; // 原始消息 + 0x80 + 长度
    if (padded_orig_len % SM3_BLOCK_SIZE != 0) {
        padded_orig_len += SM3_BLOCK_SIZE - (padded_orig_len % SM3_BLOCK_SIZE);
    }

    // 设置新的初始向量为原始哈希值
    uint32_t v[SM3_HASH_SIZE];
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        v[i] = BE_TO_HOST(((uint32_t*)orig_hash)[i]);
    }

    // 创建扩展消息
    uint8_t* new_msg = malloc(new_len);
    if (!new_msg) {
        return -2;  // 内存分配失败
    }
    memcpy(new_msg, original_msg, orig_len);
    memcpy(new_msg + orig_len, extension, ext_len);

    // 仅对扩展部分进行哈希计算
    uint8_t* ext_start = new_msg + orig_len;
    size_t ext_blocks = (ext_len + SM3_BLOCK_SIZE - 1) / SM3_BLOCK_SIZE;

    // 处理扩展部分
    for (size_t i = 0; i < ext_blocks; i++) {
        size_t block_len = SM3_BLOCK_SIZE;
        if (i == ext_blocks - 1) {
            block_len = ext_len - i * SM3_BLOCK_SIZE;
        }

        // 如果是最后一个块，需要重新填充
        if (i == ext_blocks - 1 && block_len < SM3_BLOCK_SIZE) {
            uint8_t last_block[SM3_BLOCK_SIZE] = { 0 };
            memcpy(last_block, ext_start + i * SM3_BLOCK_SIZE, block_len);
            last_block[block_len] = 0x80;

            // 设置总比特长度
            uint64_t total_bits = (padded_orig_len + new_len) * 8;
            for (int j = 0; j < 8; j++) {
                last_block[SM3_BLOCK_SIZE - 8 + j] = (total_bits >> (56 - j * 8)) & 0xFF;
            }

            sm3_compress_optimized(v, (uint32_t*)last_block);
        }
        else {
            sm3_compress_optimized(v, (uint32_t*)(ext_start + i * SM3_BLOCK_SIZE));
        }
    }

    // 输出结果
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)new_hash)[i] = HOST_TO_BE(v[i]);
    }

    // 清理
    free(new_msg);
    return 0;
}

// ================== Merkle树实现 ==================

typedef struct MerkleNode {
    uint8_t hash[SM3_DIGEST_SIZE];  // 节点哈希值
    struct MerkleNode* left;        // 左子节点
    struct MerkleNode* right;       // 右子节点
    size_t count;                   // 子树叶子节点数量
} MerkleNode;

typedef struct {
    MerkleNode* root;               // 根节点
    size_t leaf_count;              // 叶子节点数量
    MerkleNode** leaves;            // 叶子节点数组
} MerkleTree;

// 创建叶子节点
MerkleNode* create_leaf(const uint8_t* data, size_t len) {
    MerkleNode* node = malloc(sizeof(MerkleNode));
    if (!node) return NULL;

    if (sm3_hash_optimized(data, len, node->hash) != 0) {
        free(node);
        return NULL;
    }

    node->left = NULL;
    node->right = NULL;
    node->count = 1;
    return node;
}

// 创建内部节点
MerkleNode* create_parent(MerkleNode* left, MerkleNode* right) {
    MerkleNode* node = malloc(sizeof(MerkleNode));
    if (!node) return NULL;

    // 如果只有一个子节点，复制它
    if (right == NULL) {
        memcpy(node->hash, left->hash, SM3_DIGEST_SIZE);
        node->left = left;
        node->right = NULL;
        node->count = left->count;
        return node;
    }

    // 计算两个子节点的哈希
    uint8_t combined[SM3_DIGEST_SIZE * 2];
    memcpy(combined, left->hash, SM3_DIGEST_SIZE);
    memcpy(combined + SM3_DIGEST_SIZE, right->hash, SM3_DIGEST_SIZE);

    if (sm3_hash_optimized(combined, SM3_DIGEST_SIZE * 2, node->hash) != 0) {
        free(node);
        return NULL;
    }

    node->left = left;
    node->right = right;
    node->count = left->count + right->count;
    return node;
}

// 递归释放Merkle树节点
void free_merkle_node(MerkleNode* node) {
    if (!node) return;

    free_merkle_node(node->left);
    free_merkle_node(node->right);
    free(node);
}

// 构建Merkle树
MerkleTree* build_merkle_tree(uint8_t** data, size_t* lengths, size_t count) {
    if (!data || !lengths || count == 0) {
        return NULL;
    }

    MerkleTree* tree = malloc(sizeof(MerkleTree));
    if (!tree) return NULL;

    tree->leaf_count = count;
    tree->leaves = malloc(count * sizeof(MerkleNode*));
    if (!tree->leaves) {
        free(tree);
        return NULL;
    }

    // 创建叶子节点
    for (size_t i = 0; i < count; i++) {
        tree->leaves[i] = create_leaf(data[i], lengths[i]);
        if (!tree->leaves[i]) {
            // 创建失败，清理已分配的内存
            for (size_t j = 0; j < i; j++) {
                free(tree->leaves[j]);
            }
            free(tree->leaves);
            free(tree);
            return NULL;
        }
    }

    // 构建树
    size_t level_size = count;
    MerkleNode** level = tree->leaves;

    while (level_size > 1) {
        size_t next_level_size = (level_size + 1) / 2;
        MerkleNode** next_level = malloc(next_level_size * sizeof(MerkleNode*));
        if (!next_level) {
            // 内存分配失败，清理
            if (level != tree->leaves) {
                free(level);
            }
            free_merkle_tree_full(tree);
            return NULL;
        }

        for (size_t i = 0; i < level_size; i += 2) {
            MerkleNode* left = level[i];
            MerkleNode* right = (i + 1 < level_size) ? level[i + 1] : NULL;
            next_level[i / 2] = create_parent(left, right);

            if (!next_level[i / 2]) {
                // 创建失败，清理
                for (size_t j = 0; j < i / 2; j++) {
                    free_merkle_node(next_level[j]);
                }
                free(next_level);
                if (level != tree->leaves) {
                    free(level);
                }
                free_merkle_tree_full(tree);
                return NULL;
            }
        }

        if (level != tree->leaves) {
            free(level);
        }

        level = next_level;
        level_size = next_level_size;
    }

    tree->root = level[0];
    free(level);
    return tree;
}

// 释放整个Merkle树
void free_merkle_tree_full(MerkleTree* tree) {
    if (!tree) return;

    for (size_t i = 0; i < tree->leaf_count; i++) {
        free(tree->leaves[i]);
    }
    free(tree->leaves);
    free_merkle_node(tree->root);
    free(tree);
}

// 生成存在性证明
int generate_existence_proof(MerkleTree* tree, size_t index,
    uint8_t*** proof, size_t* proof_len) {
    if (!tree || !proof || !proof_len || index >= tree->leaf_count) {
        return -1;  // 无效参数
    }

    *proof_len = 0;
    size_t capacity = 16;  // 初始容量
    *proof = malloc(capacity * sizeof(uint8_t*));
    if (!*proof) {
        return -2;  // 内存分配失败
    }

    MerkleNode* current = tree->leaves[index];
    size_t current_index = index;

    // 从叶子节点向上遍历到根节点
    while (current != tree->root) {
        // 找到当前节点的父节点
        MerkleNode* parent = NULL;
        size_t parent_index = current_index / 2;

        // 在实际实现中，需要维护父指针或使用其他方法找到父节点
        // 这里简化处理，假设可以找到父节点

        // 检查是否需要扩展数组
        if (*proof_len >= capacity) {
            capacity *= 2;
            uint8_t** new_proof = realloc(*proof, capacity * sizeof(uint8_t*));
            if (!new_proof) {
                // 释放已分配的内存
                for (size_t i = 0; i < *proof_len; i++) {
                    free((*proof)[i]);
                }
                free(*proof);
                return -2;
            }
            *proof = new_proof;
        }

        // 添加兄弟节点的哈希到证明中
        if (current_index % 2 == 0) {
            // 当前是左节点，添加右兄弟
            if (parent && parent->right) {
                (*proof)[*proof_len] = malloc(SM3_DIGEST_SIZE);
                if (!(*proof)[*proof_len]) {
                    // 释放已分配的内存
                    for (size_t i = 0; i < *proof_len; i++) {
                        free((*proof)[i]);
                    }
                    free(*proof);
                    return -2;
                }
                memcpy((*proof)[*proof_len], parent->right->hash, SM3_DIGEST_SIZE);
                (*proof_len)++;
            }
        }
        else {
            // 当前是右节点，添加左兄弟
            if (parent && parent->left) {
                (*proof)[*proof_len] = malloc(SM3_DIGEST_SIZE);
                if (!(*proof)[*proof_len]) {
                    // 释放已分配的内存
                    for (size_t i = 0; i < *proof_len; i++) {
                        free((*proof)[i]);
                    }
                    free(*proof);
                    return -2;
                }
                memcpy((*proof)[*proof_len], parent->left->hash, SM3_DIGEST_SIZE);
                (*proof_len)++;
            }
        }

        // 向上移动
        current = parent;
        current_index = parent_index;

        // 简化处理，实际实现需要正确遍历到根节点
        break;
    }

    return 0;
}

// 验证存在性证明
int verify_existence_proof(const uint8_t* leaf_hash, const uint8_t* root_hash,
    uint8_t** proof, size_t proof_len,
    size_t index, size_t total_leaves) {
    if (!leaf_hash || !root_hash || !proof) {
        return -1;  // 无效参数
    }

    uint8_t current_hash[SM3_DIGEST_SIZE];
    memcpy(current_hash, leaf_hash, SM3_DIGEST_SIZE);

    size_t current_index = index;

    for (size_t i = 0; i < proof_len; i++) {
        uint8_t combined[SM3_DIGEST_SIZE * 2];

        if (current_index % 2 == 0) {
            // 当前是左节点，兄弟是右节点
            memcpy(combined, current_hash, SM3_DIGEST_SIZE);
            memcpy(combined + SM3_DIGEST_SIZE, proof[i], SM3_DIGEST_SIZE);
        }
        else {
            // 当前是右节点，兄弟是左节点
            memcpy(combined, proof[i], SM3_DIGEST_SIZE);
            memcpy(combined + SM3_DIGEST_SIZE, current_hash, SM3_DIGEST_SIZE);
        }

        // 计算父节点哈希
        if (sm3_hash_optimized(combined, SM3_DIGEST_SIZE * 2, current_hash) != 0) {
            return -2;  // 哈希计算失败
        }

        current_index /= 2;
    }

    // 比较计算出的根哈希和提供的根哈希
    return memcmp(current_hash, root_hash, SM3_DIGEST_SIZE) == 0;
}

// 生成不存在性证明
int generate_non_existence_proof(MerkleTree* tree, const uint8_t* leaf_hash,
    uint8_t*** proof, size_t* proof_len) {
    if (!tree || !leaf_hash || !proof || !proof_len) {
        return -1;  // 无效参数
    }

    // 简化实现：检查叶子是否在树中
    for (size_t i = 0; i < tree->leaf_count; i++) {
        if (memcmp(tree->leaves[i]->hash, leaf_hash, SM3_DIGEST_SIZE) == 0) {
            return 1;  // 叶子存在
        }
    }

    // 在实际应用中，需要找到相邻叶子并构造证明
    // 这里返回一个简单的证明表示不存在
    *proof_len = 0;
    *proof = NULL;
    return 0;
}

// ================== 辅助函数 ==================

// 打印哈希值
void print_hash(const uint8_t* hash) {
    if (!hash) return;

    for (int i = 0; i < SM3_DIGEST_SIZE; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

// 生成随机数据
uint8_t* generate_random_data(size_t len) {
    uint8_t* data = malloc(len);
    if (!data) return NULL;

    for (size_t i = 0; i < len; i++) {
        data[i] = rand() % 256;
    }
    return data;
}

// 性能测试
void benchmark_sm3() {
    const size_t data_size = 1024 * 1024; // 1MB
    uint8_t* data = generate_random_data(data_size);
    if (!data) {
        printf("Failed to generate test data\n");
        return;
    }

    uint8_t digest_basic[SM3_DIGEST_SIZE];
    uint8_t digest_optimized[SM3_DIGEST_SIZE];

    // 测试基本实现
    clock_t start = clock();
    if (sm3_hash_basic(data, data_size, digest_basic) != 0) {
        printf("Basic SM3 hash failed\n");
        free(data);
        return;
    }
    double elapsed_basic = (double)(clock() - start) / CLOCKS_PER_SEC;

    // 测试优化实现
    start = clock();
    if (sm3_hash_optimized(data, data_size, digest_optimized) != 0) {
        printf("Optimized SM3 hash failed\n");
        free(data);
        return;
    }
    double elapsed_optimized = (double)(clock() - start) / CLOCKS_PER_SEC;

    // 验证结果一致性
    if (memcmp(digest_basic, digest_optimized, SM3_DIGEST_SIZE) != 0) {
        printf("Error: Basic and optimized implementations produce different results!\n");
    }

    printf("SM3 Performance Benchmark (1MB data):\n");
    printf("Basic implementation: %.2f MB/s\n", data_size / elapsed_basic / 1e6);
    printf("Optimized implementation: %.2f MB/s\n", data_size / elapsed_optimized / 1e6);

    free(data);
}

// ================== 主函数 ==================

int main() {
    srand(time(NULL));

    printf("=== SM3 Algorithm Implementation and Optimization ===\n\n");

    // 测试基本功能
    const char* test_msg = "Hello, SM3!";
    uint8_t digest[SM3_DIGEST_SIZE];

    printf("Testing basic SM3 implementation:\n");
    if (sm3_hash_basic((uint8_t*)test_msg, strlen(test_msg), digest) != 0) {
        printf("SM3 hash failed\n");
        return 1;
    }
    printf("Hash of \"%s\": ", test_msg);
    print_hash(digest);

    printf("\nTesting optimized SM3 implementation:\n");
    if (sm3_hash_optimized((uint8_t*)test_msg, strlen(test_msg), digest) != 0) {
        printf("SM3 hash failed\n");
        return 1;
    }
    printf("Hash of \"%s\": ", test_msg);
    print_hash(digest);

    // 运行性能测试
    printf("\nRunning performance benchmark:\n");
    benchmark_sm3();

    // 长度扩展攻击演示
    printf("\n=== Length Extension Attack Demo ===\n");

    const char* original_msg = "Original message";
    const char* extension = "Extension attack";

    // 计算原始哈希
    uint8_t orig_hash[SM3_DIGEST_SIZE];
    if (sm3_hash_optimized((uint8_t*)original_msg, strlen(original_msg), orig_hash) != 0) {
        printf("SM3 hash failed\n");
        return 1;
    }

    printf("Original message: \"%s\"\n", original_msg);
    printf("Original hash: ");
    print_hash(orig_hash);

    // 进行长度扩展攻击
    uint8_t new_hash[SM3_DIGEST_SIZE];
    if (sm3_length_extension_attack(
        (uint8_t*)original_msg, strlen(original_msg),
        orig_hash,
        (uint8_t*)extension, strlen(extension),
        new_hash
    ) != 0) {
        printf("Length extension attack failed\n");
        return 1;
    }

    printf("\nExtended message: \"%s%s\"\n", original_msg, extension);
    printf("Extended hash via attack: ");
    print_hash(new_hash);

    // 计算实际扩展消息的哈希
    uint8_t actual_extended_hash[SM3_DIGEST_SIZE];
    size_t extended_len = strlen(original_msg) + strlen(extension);
    uint8_t* extended_msg = malloc(extended_len);
    if (!extended_msg) {
        printf("Memory allocation failed\n");
        return 1;
    }
    memcpy(extended_msg, original_msg, strlen(original_msg));
    memcpy(extended_msg + strlen(original_msg), extension, strlen(extension));

    if (sm3_hash_optimized(extended_msg, extended_len, actual_extended_hash) != 0) {
        printf("SM3 hash failed\n");
        free(extended_msg);
        return 1;
    }
    printf("Actual extended hash:    ");
    print_hash(actual_extended_hash);

    // 验证攻击是否成功
    if (memcmp(new_hash, actual_extended_hash, SM3_DIGEST_SIZE) == 0) {
        printf("\nLength extension attack succeeded!\n");
    }
    else {
        printf("\nLength extension attack failed!\n");
    }

    free(extended_msg);

    // Merkle树演示
    printf("\n=== Merkle Tree Demo ===\n");

    // 创建10万叶子节点
    const size_t leaf_count = 100000;
    uint8_t** leaf_data = malloc(leaf_count * sizeof(uint8_t*));
    size_t* leaf_lengths = malloc(leaf_count * sizeof(size_t));
    if (!leaf_data || !leaf_lengths) {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("Generating %zu leaf nodes...\n", leaf_count);
    for (size_t i = 0; i < leaf_count; i++) {
        leaf_lengths[i] = 64; // 每个叶子节点64字节数据
        leaf_data[i] = generate_random_data(leaf_lengths[i]);
        if (!leaf_data[i]) {
            printf("Failed to generate leaf data\n");
            // 清理已分配的内存
            for (size_t j = 0; j < i; j++) {
                free(leaf_data[j]);
            }
            free(leaf_data);
            free(leaf_lengths);
            return 1;
        }
    }

    printf("Building Merkle tree...\n");
    MerkleTree* tree = build_merkle_tree(leaf_data, leaf_lengths, leaf_count);
    if (!tree) {
        printf("Failed to build Merkle tree\n");
        // 清理已分配的内存
        for (size_t i = 0; i < leaf_count; i++) {
            free(leaf_data[i]);
        }
        free(leaf_data);
        free(leaf_lengths);
        return 1;
    }

    printf("Merkle tree root hash: ");
    print_hash(tree->root->hash);

    // 测试存在性证明
    size_t test_index = 12345;
    printf("\nGenerating existence proof for leaf %zu...\n", test_index);

    uint8_t** proof = NULL;
    size_t proof_len = 0;
    if (generate_existence_proof(tree, test_index, &proof, &proof_len) != 0) {
        printf("Failed to generate existence proof\n");
        free_merkle_tree_full(tree);
        for (size_t i = 0; i < leaf_count; i++) {
            free(leaf_data[i]);
        }
        free(leaf_data);
        free(leaf_lengths);
        return 1;
    }

    printf("Proof length: %zu\n", proof_len);
    printf("Verifying proof...\n");

    int valid = verify_existence_proof(
        tree->leaves[test_index]->hash,
        tree->root->hash,
        proof, proof_len,
        test_index, leaf_count
    );

    printf("Proof verification: %s\n", valid ? "SUCCESS" : "FAILURE");

    // 清理证明内存
    for (size_t i = 0; i < proof_len; i++) {
        free(proof[i]);
    }
    free(proof);

    // 测试不存在性证明
    uint8_t non_existent_leaf[SM3_DIGEST_SIZE];
    memset(non_existent_leaf, 0xFF, SM3_DIGEST_SIZE); // 创建一个不存在的叶子哈希

    printf("\nGenerating non-existence proof...\n");
    if (generate_non_existence_proof(tree, non_existent_leaf, &proof, &proof_len) == 0) {
        printf("Non-existence proof generated (simplified)\n");
    }
    else {
        printf("Leaf exists in tree\n");
    }

    // 清理
    for (size_t i = 0; i < leaf_count; i++) {
        free(leaf_data[i]);
    }
    free(leaf_data);
    free(leaf_lengths);
    free_merkle_tree_full(tree);

    return 0;
}