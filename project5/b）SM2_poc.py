import secrets
from gmssl import sm3, func

# 定义椭圆曲线参数
CURVE_A = 0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498
CURVE_B = 0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A
PRIME_FIELD = 0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3
ORDER = 0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7
BASE_X = 0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D
BASE_Y = 0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2
BASE_POINT = (BASE_X, BASE_Y)

# 计算模逆的函数
def mod_inv(a, m):
    a = a % m  # 保证a在模m范围内
    g, x, y = extended_euclid(a, m)
    if g != 1:
        return None  # 如果g不是1，表示a和m不互质
    return x % m

# 扩展欧几里得算法，计算a和b的最大公约数以及系数
def extended_euclid(a, b):
    if a == 0:
        return (b, 0, 1)
    g, y, x = extended_euclid(b % a, a)
    return (g, x - (b // a) * y, y)

# 椭圆曲线点加法函数
def elliptic_add(p, q):
    # 处理无穷远点的情况
    if p == (0, 0):
        return q
    if q == (0, 0):
        return p

    x1, y1 = p
    x2, y2 = q

    # 确保所有值都在正确的范围内
    x1, y1 = x1 % PRIME_FIELD, y1 % PRIME_FIELD
    x2, y2 = x2 % PRIME_FIELD, y2 % PRIME_FIELD

    # 计算斜率，点相加或点加倍
    if x1 == x2:
        if (y1 + y2) % PRIME_FIELD == 0:
            return (0, 0)  # 互为逆元，返回无穷远点
        num = (3 * pow(x1, 2, PRIME_FIELD) + CURVE_A) % PRIME_FIELD
        den = (2 * y1) % PRIME_FIELD
        inv_den = mod_inv(den, PRIME_FIELD)  # 求模逆
        if inv_den is None:
            raise ValueError("点加倍时模逆不存在，无效的点运算")
        slope = (num * inv_den) % PRIME_FIELD
    else:
        dx = (x2 - x1) % PRIME_FIELD
        dy = (y2 - y1) % PRIME_FIELD
        inv_dx = mod_inv(dx, PRIME_FIELD)  # 求dx的模逆
        if inv_dx is None:
            raise ValueError("点加法时模逆不存在，无效的点运算")
        slope = (dy * inv_dx) % PRIME_FIELD

    # 计算结果点的坐标
    x3 = (pow(slope, 2, PRIME_FIELD) - x1 - x2) % PRIME_FIELD
    y3 = (slope * (x1 - x3) - y1) % PRIME_FIELD
    return (x3, y3)

# 椭圆曲线点的标量乘法
def scalar_multiply(k, p):
    if k <= 0 or k >= ORDER:
        raise ValueError(f"标量k必须在(0, {ORDER})范围内")

    result = (0, 0)  # 初始点为无穷远点
    point = p
    while k > 0:
        if k % 2 == 1:
            result = elliptic_add(result, point)  # 点加法
        point = elliptic_add(point, point)  # 点加倍
        k //= 2  # k右移一位
    return result

# 生成密钥对
def generate_key():
    max_attempts = 5
    attempts = 0
    while attempts < max_attempts:
        try:
            private_key = secrets.randbelow(ORDER - 1) + 1  # 私钥范围 [1, ORDER-1]
            public_key = scalar_multiply(private_key, BASE_POINT)  # 计算公钥
            return private_key, public_key
        except ValueError as e:
            attempts += 1
            if attempts == max_attempts:
                raise RuntimeError(f"经过{max_attempts}次尝试仍无法生成有效密钥对: {e}")

# 计算用户标识哈希ZA
def compute_hash(user_id, public_key):
    """计算用户标识哈希ZA = SM3(ENTL || ID || a || b || Gx || Gy || px || py)"""
    uid_bytes = user_id.encode('utf-8')
    entl = len(uid_bytes) * 8  # 用户ID的长度（位）

    # 拼接哈希输入组件
    components = [
        entl.to_bytes(2, 'big'),  # 用户ID的位数（2字节）
        uid_bytes,  # 用户ID
        CURVE_A.to_bytes(32, 'big'),  # 曲线参数a
        CURVE_B.to_bytes(32, 'big'),  # 曲线参数b
        BASE_X.to_bytes(32, 'big'),  # 基点x坐标
        BASE_Y.to_bytes(32, 'big'),  # 基点y坐标
        public_key[0].to_bytes(32, 'big'),  # 公钥x坐标
        public_key[1].to_bytes(32, 'big')  # 公钥y坐标
    ]
    hash_input = b''.join(components)
    return sm3.sm3_hash(func.bytes_to_list(hash_input))

# SM2签名算法误用场景测试类
class SM2Tester:
    """SM2签名误用测试"""

    def test_k_leakage(self):
        """测试1：随机数k泄露导致私钥泄露"""
        print("=== 测试1：随机数k泄露导致私钥泄露 ===")
        try:
            # 生成密钥对
            sk, pk = generate_key()
            user_id = "user_01"
            message = "secret data"

            # 模拟k泄露，直接设置k值
            k = secrets.randbelow(ORDER - 1) + 1
            za = compute_hash(user_id, pk)
            e = int(sm3.sm3_hash(func.bytes_to_list(bytes.fromhex(za) + message.encode())), 16)
            kG = scalar_multiply(k, BASE_POINT)
            r = (e + kG[0]) % ORDER
            s = (mod_inv(1 + sk, ORDER) * (k - r * sk)) % ORDER

            # 攻击者利用泄露的k恢复私钥
            denominator = (s + r) % ORDER
            if denominator == 0:
                print("攻击失败：分母为零")
                return
            inv_den = mod_inv(denominator, ORDER)
            if inv_den is None:
                print("攻击失败：分母的模逆不存在")
                return
            recovered_sk = ((k - s) * inv_den) % ORDER

            print(f"原始私钥: 0x{sk:064x}")
            print(f"恢复的私钥: 0x{recovered_sk:064x}")
            print(f"验证结果: {'成功' if recovered_sk == sk else '失败'}\n")
        except Exception as e:
            print(f"测试失败: {str(e)}\n")

    def test_reused_k(self):
        """测试2：重复使用k导致私钥泄露"""
        print("=== 测试2：重复使用k导致私钥泄露 ===")
        try:
            # 生成密钥对
            sk, pk = generate_key()
            user_id = "user_02"
            msg1 = "message 1"
            msg2 = "message 2"

            # 重复使用相同的k进行签名
            k = secrets.randbelow(ORDER - 1) + 1
            za = compute_hash(user_id, pk)

            # 第一个签名
            e1 = int(sm3.sm3_hash(func.bytes_to_list(bytes.fromhex(za) + msg1.encode())), 16)
            kG = scalar_multiply(k, BASE_POINT)
            r1 = (e1 + kG[0]) % ORDER
            s1 = (mod_inv(1 + sk, ORDER) * (k - r1 * sk)) % ORDER

            # 第二个签名
            e2 = int(sm3.sm3_hash(func.bytes_to_list(bytes.fromhex(za) + msg2.encode())), 16)
            r2 = (e2 + kG[0]) % ORDER
            s2 = (mod_inv(1 + sk, ORDER) * (k - r2 * sk)) % ORDER

            # 攻击者利用两个签名恢复私钥
            numerator = (s2 - s1) % ORDER
            denominator = (s1 - s2 + r1 - r2) % ORDER
            if denominator == 0:
                print("攻击失败：分母为零")
                return
            inv_den = mod_inv(denominator, ORDER)
            if inv_den is None:
                print("攻击失败：分母的模逆不存在")
                return
            recovered_sk = (numerator * inv_den) % ORDER

            print(f"原始私钥: 0x{sk:064x}")
            print(f"恢复的私钥: 0x{recovered_sk:064x}")
            print(f"验证结果: {'成功' if recovered_sk == sk else '失败'}\n")
        except Exception as e:
            print(f"测试失败: {str(e)}\n")

    def test_forgery(self):
        """测试3：哈希计算错误导致伪造签名"""
        print("=== 测试3：哈希计算错误导致伪造签名 ===")
        try:
            # 生成密钥对
            sk, pk = generate_key()
            user_id = "user_03"
            original_msg = "original contract"

            # 错误的哈希计算导致伪造签名
            e_wrong = int(sm3.sm3_hash(func.bytes_to_list(original_msg.encode())), 16)
            k = secrets.randbelow(ORDER - 1) + 1
            kG = scalar_multiply(k, BASE_POINT)
            r = (e_wrong + kG[0]) % ORDER
            s = (mod_inv(1 + sk, ORDER) * (k - r * sk)) % ORDER

            # 攻击者伪造签名
            fake_msg = "forged contract"
            fake_e = e_wrong

            # 验证伪造签名
            t = (r + s) % ORDER
            sG = scalar_multiply(s, BASE_POINT)
            tP = scalar_multiply(t, pk)
            x3, _ = elliptic_add(sG, tP)
            verify_result = (fake_e + x3) % ORDER == r

            print(f"原始消息: {original_msg}")
            print(f"伪造消息: {fake_msg}")
            print(f"伪造签名验证: {'成功' if verify_result else '失败'}\n")
        except Exception as e:
            print(f"测试失败: {str(e)}\n")

if __name__ == "__main__":
    tester = SM2Tester()
    tester.test_k_leakage()
    tester.test_reused_k()
    tester.test_forgery()
