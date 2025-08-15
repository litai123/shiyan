#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <wmmintrin.h> // AES-NI

#define SM4_BLOCK_SIZE 16
#define SM4_KEY_SIZE 16
#define SM4_ROUNDS 32
#define SM4_GCM_IV_MAX_SIZE 64
#define SM4_GCM_TAG_SIZE 16

// S盒
static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
};

// 系统参数FK
static const uint32_t FK[4] = {
    0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc
};

// 固定参数CK
static const uint32_t CK[32] = {
    0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
    0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
    0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
    0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
    0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
    0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
    0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
    0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279
};

// ==================== 基础SM4实现 ====================

typedef struct {
    uint32_t rk[SM4_ROUNDS]; // 轮密钥
} sm4_ctx;

// 循环左移
static inline uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// 合成变换T
static uint32_t tau(uint32_t x) {
    uint32_t b0 = SBOX[(x >> 24) & 0xFF];
    uint32_t b1 = SBOX[(x >> 16) & 0xFF];
    uint32_t b2 = SBOX[(x >> 8) & 0xFF];
    uint32_t b3 = SBOX[x & 0xFF];
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

// 线性变换L
static uint32_t L(uint32_t x) {
    return x ^ rotl(x, 2) ^ rotl(x, 10) ^ rotl(x, 18) ^ rotl(x, 24);
}

// 密钥扩展线性变换L'
static uint32_t L_prime(uint32_t x) {
    return x ^ rotl(x, 13) ^ rotl(x, 23);
}

// 轮函数F
static uint32_t F(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk) {
    return x0 ^ L(tau(x1 ^ x2 ^ x3 ^ rk));
}

void sm4_setkey_enc(sm4_ctx* ctx, const unsigned char key[SM4_KEY_SIZE]) {
    uint32_t MK[4], K[36];

    // 初始化中间密钥
    MK[0] = (key[0] << 24) | (key[1] << 16) | (key[2] << 8) | key[3];
    MK[1] = (key[4] << 24) | (key[5] << 16) | (key[6] << 8) | key[7];
    MK[2] = (key[8] << 24) | (key[9] << 16) | (key[10] << 8) | key[11];
    MK[3] = (key[12] << 24) | (key[13] << 16) | (key[14] << 8) | key[15];

    K[0] = MK[0] ^ FK[0];
    K[1] = MK[1] ^ FK[1];
    K[2] = MK[2] ^ FK[2];
    K[3] = MK[3] ^ FK[3];

    // 生成轮密钥
    for (int i = 0; i < 32; i++) {
        K[i + 4] = K[i] ^ L_prime(tau(K[i + 1] ^ K[i + 2] ^ K[i + 3] ^ CK[i]));
        ctx->rk[i] = K[i + 4];
    }
}

void sm4_crypt_ecb(const sm4_ctx* ctx, int mode,
    const unsigned char input[SM4_BLOCK_SIZE],
    unsigned char output[SM4_BLOCK_SIZE]) {
    uint32_t X[36];

    // 初始化
    X[0] = (input[0] << 24) | (input[1] << 16) | (input[2] << 8) | input[3];
    X[1] = (input[4] << 24) | (input[5] << 16) | (input[6] << 8) | input[7];
    X[2] = (input[8] << 24) | (input[9] << 16) | (input[10] << 8) | input[11];
    X[3] = (input[12] << 24) | (input[13] << 16) | (input[14] << 8) | input[15];

    // 32轮迭代
    for (int i = 0; i < 32; i++) {
        int rk_idx = (mode == 1) ? i : (31 - i); // 加密/解密选择轮密钥顺序
        X[i + 4] = F(X[i], X[i + 1], X[i + 2], X[i + 3], ctx->rk[rk_idx]);
    }

    // 反序变换
    output[0] = (X[35] >> 24) & 0xFF;
    output[1] = (X[35] >> 16) & 0xFF;
    output[2] = (X[35] >> 8) & 0xFF;
    output[3] = X[35] & 0xFF;
    output[4] = (X[34] >> 24) & 0xFF;
    output[5] = (X[34] >> 16) & 0xFF;
    output[6] = (X[34] >> 8) & 0xFF;
    output[7] = X[34] & 0xFF;
    output[8] = (X[33] >> 24) & 0xFF;
    output[9] = (X[33] >> 16) & 0xFF;
    output[10] = (X[33] >> 8) & 0xFF;
    output[11] = X[33] & 0xFF;
    output[12] = (X[32] >> 24) & 0xFF;
    output[13] = (X[32] >> 16) & 0xFF;
    output[14] = (X[32] >> 8) & 0xFF;
    output[15] = X[32] & 0xFF;
}

// ==================== 优化SM4实现 ====================

typedef struct {
    __m128i rk[32]; // 使用SIMD存储轮密钥
    uint32_t T_table[4][256]; // T-table
} sm4_opt_ctx;

// 初始化T-table
static void init_T_table(sm4_opt_ctx* ctx) {
    for (int i = 0; i < 256; i++) {
        uint32_t a = SBOX[i];
        ctx->T_table[0][i] = a ^ rotl(a, 2) ^ rotl(a, 10) ^ rotl(a, 18) ^ rotl(a, 24);
        ctx->T_table[1][i] = rotl(ctx->T_table[0][i], 24);
        ctx->T_table[2][i] = rotl(ctx->T_table[0][i], 16);
        ctx->T_table[3][i] = rotl(ctx->T_table[0][i], 8);
    }
}

void sm4_opt_setkey(sm4_opt_ctx* ctx, const unsigned char key[16]) {
    // 初始化T-table
    init_T_table(ctx);

    // 使用AES-NI加速密钥扩展
    __m128i k = _mm_loadu_si128((const __m128i*)key);
    __m128i fk = _mm_set_epi32(0xb27022dc, 0x677d9197, 0x56aa3350, 0xa3b1bac6);
    k = _mm_xor_si128(k, fk);

    // 密钥扩展过程
    uint32_t K[36];
    K[0] = _mm_extract_epi32(k, 0);
    K[1] = _mm_extract_epi32(k, 1);
    K[2] = _mm_extract_epi32(k, 2);
    K[3] = _mm_extract_epi32(k, 3);

    for (int i = 0; i < 32; i++) {
        // 使用AES-NI指令加速S盒变换
        __m128i x = _mm_aesenc_si128(_mm_set_epi32(K[i + 3], K[i + 2], K[i + 1], K[i]),
            _mm_set1_epi32(CK[i]));

        K[i + 4] = K[i] ^ L_prime(_mm_extract_epi32(x, 0));
        ctx->rk[i] = _mm_set1_epi32(K[i + 4]);
    }
}

void sm4_opt_encrypt(const sm4_opt_ctx* ctx,
    const unsigned char input[16],
    unsigned char output[16]) {
    uint32_t x[4];
    memcpy(x, input, 16);

    for (int i = 0; i < 32; i++) {
        // T-table优化轮函数
        uint32_t t = _mm_extract_epi32(ctx->rk[i], 0) ^ x[1] ^ x[2] ^ x[3];
        uint32_t f = ctx->T_table[0][(t >> 24) & 0xFF] ^
            ctx->T_table[1][(t >> 16) & 0xFF] ^
            ctx->T_table[2][(t >> 8) & 0xFF] ^
            ctx->T_table[3][t & 0xFF];

        uint32_t tmp = x[0] ^ f;
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = tmp;
    }

    // 反序输出
    uint32_t tmp = x[0];
    x[0] = x[3];
    x[3] = tmp;
    tmp = x[1];
    x[1] = x[2];
    x[2] = tmp;

    memcpy(output, x, 16);
}

// ==================== SM4-GCM实现 ====================

typedef struct {
    sm4_ctx sm4_ctx;
    uint8_t H[16];  // 哈希子密钥
    uint8_t J0[16]; // 预计算计数器
} sm4_gcm_ctx;

// GF(2^128)乘法
static void gf128_mul(uint8_t* x, const uint8_t* y) {
    uint8_t z[16] = { 0 };
    uint8_t v[16];
    memcpy(v, y, 16);

    for (int i = 0; i < 128; i++) {
        int byte_pos = i / 8;
        int bit_pos = 7 - (i % 8);

        if (x[byte_pos] & (1 << bit_pos)) {
            for (int j = 0; j < 16; j++) {
                z[j] ^= v[j];
            }
        }

        uint8_t carry = v[15] & 0x01;
        for (int j = 15; j > 0; j--) {
            v[j] = (v[j] >> 1) | ((v[j - 1] & 0x01) << 7);
        }
        v[0] >>= 1;
        if (carry) {
            v[0] ^= 0xE1; // 不可约多项式x^128 + x^7 + x^2 + x + 1
        }
    }

    memcpy(x, z, 16);
}

// GHASH函数
static void ghash(const uint8_t* H, const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ciphertext_len,
    uint8_t* output) {
    uint8_t X[16] = { 0 };
    size_t i;

    // 处理附加认证数据(AAD)
    for (i = 0; i + 16 <= aad_len; i += 16) {
        for (int j = 0; j < 16; j++) {
            X[j] ^= aad[i + j];
        }
        gf128_mul(X, H);
    }

    // 处理剩余AAD
    if (i < aad_len) {
        size_t remaining = aad_len - i;
        uint8_t block[16] = { 0 };
        memcpy(block, aad + i, remaining);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gf128_mul(X, H);
    }

    // 处理密文
    for (i = 0; i + 16 <= ciphertext_len; i += 16) {
        for (int j = 0; j < 16; j++) {
            X[j] ^= ciphertext[i + j];
        }
        gf128_mul(X, H);
    }

    // 处理剩余密文
    if (i < ciphertext_len) {
        size_t remaining = ciphertext_len - i;
        uint8_t block[16] = { 0 };
        memcpy(block, ciphertext + i, remaining);

        for (int j = 0; j < 16; j++) {
            X[j] ^= block[j];
        }
        gf128_mul(X, H);
    }

    // 添加长度信息
    uint64_t len_bits_aad = aad_len * 8;
    uint64_t len_bits_ct = ciphertext_len * 8;
    uint8_t len_block[16];
    memcpy(len_block, &len_bits_aad, 8);
    memcpy(len_block + 8, &len_bits_ct, 8);

    for (int j = 0; j < 16; j++) {
        X[j] ^= len_block[j];
    }
    gf128_mul(X, H);

    memcpy(output, X, 16);
}

void sm4_gcm_init(sm4_gcm_ctx* ctx, const uint8_t* key, size_t key_len) {
    uint8_t zero_block[16] = { 0 };

    // 初始化SM4密钥
    sm4_setkey_enc(&ctx->sm4_ctx, key);

    // 计算哈希子密钥H = SM4(0^128)
    sm4_crypt_ecb(&ctx->sm4_ctx, 1, zero_block, ctx->H);
}

void sm4_gcm_encrypt(sm4_gcm_ctx* ctx,
    const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* ciphertext,
    uint8_t* tag, size_t tag_len) {
    uint8_t J0[16], ctr[16], hash[16] = { 0 };
    size_t blocks, remainder;

    // 步骤1: 生成J0
    if (iv_len == 12) {
        memcpy(J0, iv, 12);
        memset(J0 + 12, 0, 3);
        J0[15] = 1;
    }
    else {
        // GHASH计算J0
        memset(J0, 0, 16);
        size_t iv_blocks = iv_len / 16;
        size_t iv_remain = iv_len % 16;

        for (size_t i = 0; i < iv_blocks; i++) {
            for (int j = 0; j < 16; j++) {
                J0[j] ^= iv[i * 16 + j];
            }
            gf128_mul(J0, ctx->H);
        }

        if (iv_remain) {
            uint8_t block[16] = { 0 };
            memcpy(block, iv + iv_blocks * 16, iv_remain);

            for (int j = 0; j < 16; j++) {
                J0[j] ^= block[j];
            }
            gf128_mul(J0, ctx->H);
        }

        uint64_t len_bits = iv_len * 8;
        uint8_t len_block[16];
        memset(len_block, 0, 16);
        memcpy(len_block + 8, &len_bits, 8);

        for (int j = 0; j < 16; j++) {
            J0[j] ^= len_block[j];
        }
        gf128_mul(J0, ctx->H);
    }
    memcpy(ctx->J0, J0, 16);

    // 步骤2: 计算密文
    memcpy(ctr, J0, 16);
    ctr[15]++;

    blocks = plaintext_len / 16;
    remainder = plaintext_len % 16;

    for (size_t i = 0; i < blocks; i++) {
        uint8_t keystream[16];
        sm4_crypt_ecb(&ctx->sm4_ctx, 1, ctr, keystream);

        for (int j = 0; j < 16; j++) {
            ciphertext[i * 16 + j] = plaintext[i * 16 + j] ^ keystream[j];
        }

        // 更新计数器
        for (int j = 15; j >= 0; j--) {
            if (++ctr[j] != 0) break;
        }
    }

    // 处理最后一个不完整块
    if (remainder) {
        uint8_t keystream[16];
        sm4_crypt_ecb(&ctx->sm4_ctx, 1, ctr, keystream);

        for (size_t j = 0; j < remainder; j++) {
            ciphertext[blocks * 16 + j] = plaintext[blocks * 16 + j] ^ keystream[j];
        }
    }

    // 步骤3: 计算认证标签
    ghash(ctx->H, aad, aad_len, ciphertext, plaintext_len, hash);

    // 最终标签计算
    uint8_t S[16];
    sm4_crypt_ecb(&ctx->sm4_ctx, 1, J0, S);

    for (size_t i = 0; i < tag_len && i < 16; i++) {
        tag[i] = hash[i] ^ S[i];
    }
}

// ==================== 测试代码 ====================

void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void test_basic_sm4() {
    printf("=== SM4 Basic Test ===\n");

    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t plaintext[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t ciphertext[16], decrypted[16];

    sm4_ctx ctx;
    sm4_setkey_enc(&ctx, key);
    sm4_crypt_ecb(&ctx, 1, plaintext, ciphertext);

    sm4_setkey_enc(&ctx, key); // 解密使用相同的密钥
    sm4_crypt_ecb(&ctx, 0, ciphertext, decrypted);

    print_hex("Key      ", key, 16);
    print_hex("Plaintext", plaintext, 16);
    print_hex("Cipher   ", ciphertext, 16);
    print_hex("Decrypted", decrypted, 16);

    if (memcmp(plaintext, decrypted, 16) == 0) {
        printf("Test PASSED\n");
    }
    else {
        printf("Test FAILED\n");
    }
    printf("\n");
}

void test_optimized_sm4() {
    printf("=== SM4 Optimized Test ===\n");

    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t plaintext[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t ciphertext[16], decrypted[16];

    sm4_opt_ctx ctx;
    sm4_opt_setkey(&ctx, key);
    sm4_opt_encrypt(&ctx, plaintext, ciphertext);

    // 解密使用相同的密钥和过程（SM4解密与加密相同，只是轮密钥顺序相反）
    sm4_opt_setkey(&ctx, key);
    sm4_opt_encrypt(&ctx, ciphertext, decrypted);

    print_hex("Key      ", key, 16);
    print_hex("Plaintext", plaintext, 16);
    print_hex("Cipher   ", ciphertext, 16);
    print_hex("Decrypted", decrypted, 16);

    if (memcmp(plaintext, decrypted, 16) == 0) {
        printf("Test PASSED\n");
    }
    else {
        printf("Test FAILED\n");
    }
    printf("\n");
}

void test_sm4_gcm() {
    printf("=== SM4-GCM Test ===\n");

    uint8_t key[16] = {
        0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c,
        0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30, 0x83, 0x08
    };

    uint8_t iv[12] = {
        0xca, 0xfe, 0xba, 0xbe, 0xfa, 0xce, 0xdb, 0xad,
        0xde, 0xca, 0xf8, 0x88
    };

    uint8_t aad[20] = {
        0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad, 0xbe, 0xef,
        0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad, 0xbe, 0xef,
        0xab, 0xad, 0xda, 0xd2
    };

    const char* plaintext_str = "This is a test message for SM4-GCM mode implementation.";
    size_t plaintext_len = strlen(plaintext_str);
    uint8_t plaintext[64], ciphertext[64], tag[16], decrypted[64];
    memcpy(plaintext, plaintext_str, plaintext_len);

    int auth_failed = 0;

    sm4_gcm_ctx ctx;
    sm4_gcm_init(&ctx, key, 16);

    // 加密
    sm4_gcm_encrypt(&ctx, iv, 12, aad, 20, plaintext, plaintext_len,
        ciphertext, tag, 16);

    // 解密
    sm4_gcm_encrypt(&ctx, iv, 12, aad, 20, ciphertext, plaintext_len,
        decrypted, tag, 16);

    // 验证认证标签
    uint8_t verify_tag[16];
    sm4_gcm_encrypt(&ctx, iv, 12, aad, 20, ciphertext, plaintext_len,
        NULL, verify_tag, 16);

    if (memcmp(tag, verify_tag, 16) != 0) {
        auth_failed = 1;
    }

    print_hex("Key      ", key, 16);
    print_hex("IV       ", iv, 12);
    print_hex("AAD      ", aad, 20);
    printf("Plaintext: %s\n", plaintext);
    print_hex("Cipher   ", ciphertext, plaintext_len);
    print_hex("Tag      ", tag, 16);
    printf("Decrypted: %s\n", decrypted);
    printf("Auth %s\n", auth_failed ? "FAILED" : "SUCCESS");

    if (memcmp(plaintext, decrypted, plaintext_len) == 0 && !auth_failed) {
        printf("Test PASSED\n");
    }
    else {
        printf("Test FAILED\n");
    }
    printf("\n");
}

int main() {
    test_basic_sm4();
    test_optimized_sm4();
    test_sm4_gcm();
    return 0;
}