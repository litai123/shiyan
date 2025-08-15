import secrets
import hashlib
from typing import List, Tuple, Dict
from math import gcd
from collections import defaultdict


class HomomorphicEncryptionScheme:
    """支持加法同态加密的系统"""

    def __init__(self, key_size=1024):
        self.prime_p = self._generate_large_prime(key_size // 2)
        self.prime_q = self._generate_large_prime(key_size // 2)

        self.modulus_n = self.prime_p * self.prime_q
        self.generator = self.modulus_n + 1
        self.modulus_sq = self.modulus_n ** 2

        carmichael = (self.prime_p - 1) * (self.prime_q - 1) // gcd(
            self.prime_p - 1, self.prime_q - 1)
        self.private_key = (carmichael, pow(carmichael, -1, self.modulus_n))

    def _generate_large_prime(self, bits):
        """生成密码学安全的大素数"""
        while True:
            candidate = secrets.randbits(bits)
            if candidate % 2 == 0:
                candidate += 1
            if self._is_probable_prime(candidate):
                return candidate

    def _is_probable_prime(self, n, trials=128):
        """使用Miller-Rabin进行素性检测"""
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(trials):
            a = secrets.randbelow(n - 3) + 2
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def encrypt_data(self, plaintext):
        """加密数据"""
        rand = secrets.randbelow(self.modulus_n)
        while gcd(rand, self.modulus_n) != 1:
            rand = secrets.randbelow(self.modulus_n)
        return (pow(self.generator, plaintext, self.modulus_sq) *
                pow(rand, self.modulus_n, self.modulus_sq)) % self.modulus_sq

    def decrypt_data(self, ciphertext):
        """解密数据"""
        l_func = (pow(ciphertext, self.private_key[0], self.modulus_sq) - 1) // self.modulus_n
        return (l_func * self.private_key[1]) % self.modulus_n

    def homomorphic_add(self, cipher1, cipher2):
        """同态加法"""
        return (cipher1 * cipher2) % self.modulus_sq


class SecureComparisonProtocol:
    """安全比较协议实现"""

    def __init__(self, prime_bits=512):
        self.large_prime, self.subgroup_order = self._setup_parameters(prime_bits)
        self.generator = self._find_generator()

    def _setup_parameters(self, bits):
        """建立安全参数"""
        while True:
            q = self._generate_large_prime(bits)
            p = 2 * q + 1
            if self._is_probable_prime(p):
                return p, q

    def _find_generator(self):
        """寻找生成元"""
        while True:
            g = secrets.randbelow(self.large_prime - 2) + 2
            if pow(g, self.subgroup_order, self.large_prime) == 1:
                return g

    def hash_to_subgroup(self, data: str):
        """哈希到子群"""
        digest = int(hashlib.sha3_256(data.encode()).hexdigest(), 16)
        return pow(self.generator, digest % self.subgroup_order, self.large_prime)

    def exponentiate_hash(self, data: str, exponent: int):
        """哈希后指数运算"""
        hashed = self.hash_to_subgroup(data)
        return pow(hashed, exponent, self.large_prime)


def check_compromised_credentials(
        client_credentials: List[str],
        server_breach_db: List[str]
) -> List[Tuple[str, bool]]:
    """
    隐私保护的凭证泄露检查

    参数:
        client_credentials: 客户端凭证列表
        server_breach_db: 服务端泄露凭证数据库

    返回:
        每个凭证及其是否泄露的状态
    """
    # 初始化加密系统
    crypto_system = HomomorphicEncryptionScheme()
    comparison_protocol = SecureComparisonProtocol()

    # 服务端处理
    server_secret = secrets.randbelow(comparison_protocol.subgroup_order - 1) + 1
    server_processed = []
    breach_hashes = {}

    for item in server_breach_db:
        hashed = comparison_protocol.hash_to_subgroup(item)
        breach_hashes[hashed] = item
        exp_hash = comparison_protocol.exponentiate_hash(item, server_secret)
        encrypted_flag = crypto_system.encrypt_data(1)
        server_processed.append((exp_hash, encrypted_flag))

    random.shuffle(server_processed)

    # 客户端处理
    client_processed = []
    credential_map = {}

    for cred in client_credentials:
        hashed = comparison_protocol.hash_to_subgroup(cred)
        credential_map[hashed] = cred
        exp_hash = comparison_protocol.exponentiate_hash(cred, server_secret)
        client_processed.append(exp_hash)

    random.shuffle(client_processed)

    # 安全比较
    server_lookup = {h: flag for h, flag in server_processed}
    encrypted_results = []

    for client_hash in client_processed:
        encrypted_results.append(
            server_lookup.get(client_hash, crypto_system.encrypt_data(0)))

    # 解密结果
    final_results = []
    result_mapping = defaultdict(list)

    for cipher in encrypted_results:
        decrypted = crypto_system.decrypt_data(cipher)
        final_results.append(decrypted == 1)

    # 建立正确映射
    output = []
    for cred in client_credentials:
        hashed = comparison_protocol.hash_to_subgroup(cred)
        exp_hash = comparison_protocol.exponentiate_hash(cred, server_secret)
        is_breached = exp_hash in server_lookup
        output.append((cred, is_breached))

    return output


def demonstrate_usage():
    """演示使用示例"""
    test_credentials = [
        "securePassword1",
        "compromised123",
        "anotherSecure",
        "leakedPassword"
    ]

    breach_database = [
        "compromised123",
        "leakedPassword",
        "commonPassword",
        "weak123"
    ]

    print("\n=== 隐私保护的凭证检查演示 ===")
    print("客户端凭证:", test_credentials)
    print("泄露数据库:", breach_database)

    results = check_compromised_credentials(test_credentials, breach_database)

    print("\n检查结果:")
    for credential, is_breached in results:
        status = "存在泄露风险" if is_breached else "安全"
        print(f"凭证: {credential.ljust(20)} 状态: {status}")


if __name__ == "__main__":
    demonstrate_usage()