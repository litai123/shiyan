import secrets
from hashlib import sha256
from functools import lru_cache


class EllipticCurveConfig:
    def __init__(self):
        self.prime = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        self.coef_a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.coef_b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        self.base_x = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.base_y = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
        self.order = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
        self.cofactor = 1

    def contains_point(self, x, y):
        if x is None and y is None:
            return True
        left = pow(y, 2, self.prime)
        right = (pow(x, 3, self.prime) + self.coef_a * x + self.coef_b) % self.prime
        return left == right


class SM2CryptoSystem:
    def __init__(self):
        self.config = EllipticCurveConfig()
        self.secret = None
        self.public = None
        self.base_point = (self.config.base_x, self.config.base_y)

    @lru_cache(maxsize=512)
    def compute_inverse(self, value, modulus):
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            g, y, x = extended_gcd(b % a, a)
            return g, x - (b // a) * y, y

        g, x, _ = extended_gcd(value, modulus)
        return x % modulus if g == 1 else None

    def add_points(self, p, q):
        px, py = p
        qx, qy = q

        if px is None and py is None:
            return q
        if qx is None and qy is None:
            return p
        if px == qx and py != qy:
            return (None, None)

        if px != qx:
            delta_x = (qx - px) % self.config.prime
            delta_y = (qy - py) % self.config.prime
            inv_delta_x = self.compute_inverse(delta_x, self.config.prime)
            slope = (delta_y * inv_delta_x) % self.config.prime
        else:
            numerator = (3 * pow(px, 2, self.config.prime) + self.config.coef_a
            denominator = (2 * py) % self.config.prime
            slope = (numerator * self.compute_inverse(denominator, self.config.prime)) % self.config.prime

        rx = (pow(slope, 2, self.config.prime) - px - qx) % self.config.prime
        ry = (slope * (px - rx) - py) % self.config.prime

        return (rx, ry)

    def multiply_point(self, scalar, point):
        result = (None, None)
        current = point
        for bit in bin(scalar)[3:]:
            result = self.add_points(result, result)
            if bit == '1':
                result = self.add_points(result, current)
        return result

    def create_keys(self, custom_rand=None):
        rand_func = custom_rand or (lambda: secrets.randbelow(self.config.order - 2) + 1)
        self.secret = rand_func()
        self.public = self.multiply_point(self.secret, self.base_point)
        return self.secret, self.public

    def assign_key(self, private_key):
        self.secret = private_key
        self.public = self.multiply_point(private_key, self.base_point)
        return self.public

    @staticmethod
    def hash_data(data):
        return sha256(data).digest()

    def create_signature(self, data):
        if not self.secret:
            raise RuntimeError("Private key not initialized")
        if not isinstance(data, bytes):
            raise TypeError("Input must be bytes")

        hash_val = int.from_bytes(self.hash_data(data), 'big')
        nonce = secrets.randbelow(self.config.order - 1) + 1
        x1, _ = self.multiply_point(nonce, self.base_point)

        r = (hash_val + x1) % self.config.order
        if r == 0 or r + nonce == self.config.order:
            return self.create_signature(data)

        inv_factor = self.compute_inverse((self.secret + 1) % self.config.order, self.config.order)
        s = (inv_factor * (nonce - r * self.secret)) % self.config.order

        if s == 0:
            return self.create_signature(data)

        return (r, s)

    def check_signature(self, data, sig, pub_key):
        if not isinstance(data, bytes):
            raise TypeError("Input must be bytes")
        if not isinstance(sig, tuple) or len(sig) != 2:
            raise TypeError("Signature must be a tuple of two values")

        r, s = sig
        pub_x, pub_y = pub_key

        if not (0 < r < self.config.order and 0 < s < self.config.order):
            return False
        if not self.config.contains_point(pub_x, pub_y):
            return False

        hash_val = int.from_bytes(self.hash_data(data), 'big')
        composite = (r + s) % self.config.order
        if composite == 0:
            return False

        x1, y1 = self.multiply_point(s, self.base_point)
        x2, y2 = self.multiply_point(composite, (pub_x, pub_y))
        x3, y3 = self.add_points((x1, y1), (x2, y2))

        return x3 is not None and (hash_val + x3) % self.config.order == r

    def encrypt_data(self, plaintext, pub_key):
        if not isinstance(plaintext, bytes):
            raise TypeError("Input must be bytes")
        if not self.config.contains_point(*pub_key):
            raise ValueError("Invalid public key")

        nonce = secrets.randbelow(self.config.order - 1) + 1
        c1 = self.multiply_point(nonce, self.base_point)
        shared = self.multiply_point(nonce, pub_key)
        shared_x, shared_y = shared

        key_material = shared_x.to_bytes(32, 'big') + shared_y.to_bytes(32, 'big')
        derived_key = self.hash_data(key_material)

        pad_len = len(plaintext)
        if len(derived_key) < pad_len:
            derived_key = (derived_key * ((pad_len // len(derived_key)) + 1))[:pad_len]

        ciphertext = bytes(p ^ k for p, k in zip(plaintext, derived_key))
        auth_tag = self.hash_data(shared_x.to_bytes(32, 'big') + plaintext + shared_y.to_bytes(32, 'big'))

        return (*c1, ciphertext, auth_tag)

    def decrypt_data(self, encrypted):
        if not self.secret:
            raise RuntimeError("Private key not initialized")

        c1x, c1y, ciphertext, tag = encrypted
        c1 = (c1x, c1y)

        if not self.config.contains_point(*c1):
            raise ValueError("Invalid ciphertext")

        shared = self.multiply_point(self.secret, c1)
        shared_x, shared_y = shared

        key_material = shared_x.to_bytes(32, 'big') + shared_y.to_bytes(32, 'big')
        derived_key = self.hash_data(key_material)

        pad_len = len(ciphertext)
        if len(derived_key) < pad_len:
            derived_key = (derived_key * ((pad_len // len(derived_key)) + 1))[:pad_len]

        plaintext = bytes(c ^ k for c, k in zip(ciphertext, derived_key))

        computed_tag = self.hash_data(shared_x.to_bytes(32, 'big') + plaintext + shared_y.to_bytes(32, 'big'))
        if computed_tag != tag:
            raise ValueError("Authentication failed")

        return plaintext


def demo_usage():
    crypto = SM2CryptoSystem()
    priv, pub = crypto.create_keys()
    print(f"Private: {priv:x}\nPublic: ({pub[0]:x}, {pub[1]:x})")

    msg = b"Important secret message"
    sig = crypto.create_signature(msg)
    print(f"Signature valid: {crypto.check_signature(msg, sig, pub)}")

    cipher = crypto.encrypt_data(msg, pub)
    decrypted = crypto.decrypt_data(cipher)
    print(f"Decrypted: {decrypted.decode()}")


if __name__ == "__main__":
    demo_usage()