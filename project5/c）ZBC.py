import random
import hashlib
from functools import lru_cache


class ECCurve:
    def __init__(self):
        # 定义素数域参数
        self.p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        # 椭圆曲线参数 y^2 = x^3 + ax + b
        self.a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        # 基点G的坐标
        self.Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
        # 基点的阶
        self.n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
        # 余因子
        self.h = 1

    def is_on_curve(self, x, y):
        """检查点(x, y)是否在椭圆曲线上"""
        if x is None and y is None:
            return True  # 无穷远点
        return (pow(y, 2, self.p) - (pow(x, 3, self.p) + self.a * x + self.b) % self.p) % self.p == 0


class SM2Signer:
    def __init__(self):
        self.curve = ECCurve()
        self.private_key = None
        self.public_key = None
        # 预计算基点，避免重复计算
        self.G = (self.curve.Gx, self.curve.Gy)

    @lru_cache(maxsize=1024)
    def inverse_mod(self, a, m):
        """计算模逆并缓存结果以提高性能"""
        g, x, y = self.extended_euclid(a, m)
        if g != 1:
            return None  # 无法计算模逆
        return x % m

    def extended_euclid(self, a, b):
        """扩展欧几里得算法"""
        if a == 0:
            return b, 0, 1
        else:
            g, y, x = self.extended_euclid(b % a, a)
            return g, x - (b // a) * y, y

    def add_points(self, p1, p2):
        """计算两个椭圆曲线点的和"""
        x1, y1 = p1
        x2, y2 = p2

        # 处理无穷远点
        if x1 is None and y1 is None:
            return p2
        if x2 is None and y2 is None:
            return p1
        if x1 == x2 and y1 != y2:
            return (None, None)  # 互为逆元，返回无穷远点

        # 计算斜率
        if x1 != x2:
            dx = (x2 - x1) % self.curve.p
            dy = (y2 - y1) % self.curve.p
            inv_dx = self.inverse_mod(dx, self.curve.p)
            k = (dy * inv_dx) % self.curve.p
        else:
            # 点加倍
            k_num = (3 * pow(x1, 2, self.curve.p) + self.curve.a) % self.curve.p
            k_den = (2 * y1) % self.curve.p
            k = (k_num * self.inverse_mod(k_den, self.curve.p)) % self.curve.p

        # 计算结果点
        x3 = (pow(k, 2, self.curve.p) - x1 - x2) % self.curve.p
        y3 = (k * (x1 - x3) - y1) % self.curve.p

        return (x3, y3)

    def multiply_point(self, k, point):
        """对椭圆曲线上的点进行标量乘法"""
        result = (None, None)  # 初始点为无穷远点
        current = point
        k_binary = bin(k)[2:]  # 获取二进制表示

        # 使用二进制展开法优化点乘
        for bit in k_binary:
            result = self.add_points(result, result)  # 点加倍
            if bit == '1':
                result = self.add_points(result, current)  # 累加

        return result

    def generate_key_pair(self, rand_func=None):
        """生成密钥对"""
        rand = rand_func or (lambda: random.randint(2, self.curve.n - 2))
        self.private_key = rand()
        self.public_key = self.multiply_point(self.private_key, self.G)
        return self.private_key, self.public_key

    def set_private_key(self, private_key):
        """根据私钥生成公钥"""
        self.private_key = private_key
        self.public_key = self.multiply_point(private_key, self.G)
        return self.public_key

    @staticmethod
    def hash_message(data):
        """对消息进行SM3哈希计算"""
        return hashlib.sha256(data).digest()

    def sign_message(self, message):
        """生成消息签名"""
        if not self.private_key:
            raise ValueError("请先生成私钥")
        if not isinstance(message, bytes):
            raise TypeError("消息必须是bytes类型")

        # 计算消息哈希值
        e = int.from_bytes(self.hash_message(message), byteorder='big')

        # 生成随机数k
        k = random.SystemRandom().randint(1, self.curve.n - 1)
        x1, y1 = self.multiply_point(k, self.G)

        # 计算签名
        r = (e + x1) % self.curve.n
        if r == 0 or r + k == self.curve.n:
            return self.sign_message(message)  # 重新生成

        inv_d_plus_1 = self.inverse_mod((self.private_key + 1) % self.curve.n, self.curve.n)
        s = (inv_d_plus_1 * (k - r * self.private_key)) % self.curve.n

        if s == 0:
            return self.sign_message(message)  # 重新生成

        return (r, s)

    def verify_signature(self, message, signature, public_key):
        """验证签名"""
        if not isinstance(message, bytes):
            raise TypeError("消息必须是bytes类型")
        if not isinstance(signature, tuple) or len(signature) != 2:
            raise TypeError("签名必须是包含两个元素的元组")

        r, s = signature
        px, py = public_key

        # 验证签名参数范围
        if not (1 <= r < self.curve.n and 1 <= s < self.curve.n):
            return False
        if not self.curve.is_on_curve(px, py):
            return False

        # 计算消息哈希
        e = int.from_bytes(self.hash_message(message), byteorder='big')

        # 验证签名
        t = (r + s) % self.curve.n
        if t == 0:
            return False

        x1, y1 = self.multiply_point(s, self.G)
        x2, y2 = self.multiply_point(t, (px, py))
        x3, y3 = self.add_points((x1, y1), (x2, y2))

        if x3 is None:
            return False

        return (e + x3) % self.curve.n == r


def simulate_signature_forgery(sm2, message, fake_public_key):
    """模拟伪造签名"""
    # 计算消息哈希
    e = int.from_bytes(sm2.hash_message(message), byteorder='big')

    # 尝试多次直到伪造成功
    attempts = 0
    max_attempts = 1000

    while attempts < max_attempts:
        attempts += 1

        # 随机选择r和s值
        r = random.randint(1, sm2.curve.n - 1)
        s = random.randint(1, sm2.curve.n - 1)

        # 计算验证所需的点
        t = (r + s) % sm2.curve.n
        if t == 0:
            continue

        x1, y1 = sm2.multiply_point(s, sm2.G)
        x2, y2 = sm2.multiply_point(t, fake_public_key)
        x3, y3 = sm2.add_points((x1, y1), (x2, y2))

        if x3 is None:
            continue

        # 检查验证条件
        if (e + x3) % sm2.curve.n == r:
            print(f"伪造成功，尝试了 {attempts} 次")
            return (r, s)

    raise Exception(f"经过 {max_attempts} 次尝试仍未伪造成功")


if __name__ == "__main__":
    import time

    sm2 = SM2Signer()

    # 生成伪造的中本聪公钥
    _, fake_satoshi_pubkey = sm2.generate_key_pair()
    print(f"伪造的中本聪公钥: (0x{fake_satoshi_pubkey[0]:064x}, 0x{fake_satoshi_pubkey[1]:064x})")

    # 准备消息
    message = "我是中本聪，我批准此交易".encode('utf-8')
    print(f"消息内容: {message.decode('utf-8')}")

    start_time = time.time()
    try:
        fake_signature = simulate_signature_forgery(sm2, message, fake_satoshi_pubkey)
        print(f"伪造的签名: (0x{fake_signature[0]:064x}, 0x{fake_signature[1]:064x})")

        is_valid = sm2.verify_signature(message, fake_signature, fake_satoshi_pubkey)
        print(f"签名验证结果: {'有效' if is_valid else '无效'}")
    except Exception as e:
        print(f"伪造失败: {str(e)}")

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f}秒")

