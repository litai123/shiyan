import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import platform
import traceback
import random
import math
import time
from tqdm import tqdm


class AdvancedWatermarkProcessor:
    def __init__(self):
        self.wm_content = ""
        self.src_img = None
        self.wm_img = None
        self.extracted_wm = ""
        self.cn_font = self._find_chinese_font()
        self.orig_dims = None
        self.block_dim = 8  # 块尺寸
        self.wm_power = 5.0  # 水印强度
        self.rand_seed = 42  # 随机种子
        random.seed(self.rand_seed)

        # 配置Matplotlib字体
        if self.cn_font:
            plt.rcParams['font.sans-serif'] = [self.cn_font]
        plt.rcParams['axes.unicode_minus'] = False

    def _find_chinese_font(self):
        """定位系统中可用的中文字体"""
        try:
            # 常见中文字体列表
            font_options = [
                'SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun',
                'FangSong', 'STXihei', 'STKaiti', 'STSong', 'STFangsong',
                'LiHei Pro', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei'
            ]

            # 获取可用字体
            available = [f.name for f in fm.fontManager.ttflist]

            # 查找可用字体
            for font in font_options:
                if font in available:
                    return font

            # 系统路径查找
            os_type = platform.system()
            if os_type == "Windows":
                font_path = "C:/Windows/Fonts/simhei.ttf"
                if os.path.exists(font_path):
                    return "SimHei"
            elif os_type == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/PingFang.ttc"
                if os.path.exists(font_path):
                    return "PingFang SC"
            elif os_type == "Linux":
                font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
                if os.path.exists(font_path):
                    return "WenQuanYi Zen Hei"

            print("警告: 未找到中文字体，中文显示可能异常")
            return None

        except Exception as e:
            print(f"字体错误: {str(e)}")
            return None

    def load_img(self, img_path):
        """加载图像文件"""
        self.src_img = cv2.imread(img_path)
        if self.src_img is None:
            # 创建示例图像
            self.src_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(self.src_img, "示例图像", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(img_path, self.src_img)
            print(f"创建示例图像: {img_path}")

        # 保存原始尺寸
        self.orig_dims = self.src_img.shape
        return self.src_img.copy()

    def set_watermark(self, text):
        """设置水印文本"""
        self.wm_content = text

    def embed_robust_watermark(self):
        """嵌入鲁棒水印"""
        if self.src_img is None:
            raise ValueError("请先加载图像")
        if not self.wm_content:
            raise ValueError("请设置水印文本")

        # 创建副本
        watermarked = self.src_img.copy()

        # 文本编码
        text_bytes = self.wm_content.encode('utf-8')

        # 添加元数据
        len_prefix = len(text_bytes).to_bytes(2, 'big')
        checksum = sum(text_bytes) % 65536
        checksum_bytes = checksum.to_bytes(2, 'big')
        full_data = len_prefix + text_bytes + checksum_bytes

        # 转换为二进制
        watermark_bits = ''.join(format(byte, '08b') for byte in full_data)

        # 检查容量
        h, w, _ = watermarked.shape
        max_bits = (h // self.block_dim) * (w // self.block_dim)
        if len(watermark_bits) > max_bits:
            raise ValueError(f"水印太大! 最大支持 {max_bits // 8 - 4} 字符, 当前 {len(self.wm_content)} 字符")

        # 在DCT域嵌入水印
        watermarked = self._dct_embed(watermarked, watermark_bits)

        self.wm_img = watermarked
        return watermarked

    def _dct_embed(self, img, wm_bits):
        """在DCT系数中嵌入水印"""
        # 转换到YUV空间
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_ch = yuv_img[:, :, 0].astype(np.float32)

        # 图像尺寸
        h, w = y_ch.shape
        block_size = self.block_dim
        blocks_y = h // block_size
        blocks_x = w // block_size

        # 计算所需块数
        bits_needed = len(wm_bits)
        total_blocks = blocks_y * blocks_x
        if bits_needed > total_blocks:
            raise ValueError("水印过大，无法嵌入")

        # 随机块索引
        block_indices = [(i * block_size, j * block_size)
                         for i in range(blocks_y)
                         for j in range(blocks_x)]
        random.shuffle(block_indices)

        # 嵌入水印位
        bit_idx = 0
        for block_pos in tqdm(block_indices[:bits_needed], desc="嵌入水印", unit="块"):
            y, x = block_pos

            # 边界检查
            if y + block_size > h or x + block_size > w:
                continue

            block = y_ch[y:y + block_size, x:x + block_size]

            # DCT变换
            dct_block = cv2.dct(block)

            # 选择中频系数
            if dct_block.shape[0] > 4 and dct_block.shape[1] > 4:
                coeff1 = dct_block[3, 4]
                coeff2 = dct_block[4, 3]
            else:
                mid = block_size // 2
                coeff1 = dct_block[mid, mid + 1] if mid + 1 < block_size else dct_block[mid, mid]
                coeff2 = dct_block[mid + 1, mid] if mid + 1 < block_size else dct_block[mid, mid]

            # 嵌入位
            if bit_idx < len(wm_bits):
                bit_val = int(wm_bits[bit_idx])

                # 调整系数
                avg_val = (coeff1 + coeff2) / 2
                diff_val = abs(coeff1 - coeff2)

                min_diff = self.wm_power
                if diff_val < min_diff:
                    if bit_val == 1:
                        coeff1 = avg_val + min_diff / 2
                        coeff2 = avg_val - min_diff / 2
                    else:
                        coeff1 = avg_val - min_diff / 2
                        coeff2 = avg_val + min_diff / 2
                else:
                    if bit_val == 1:
                        if coeff1 < coeff2:
                            coeff1, coeff2 = coeff2, coeff1
                    else:
                        if coeff1 > coeff2:
                            coeff1, coeff2 = coeff2, coeff1

                # 更新系数
                if dct_block.shape[0] > 4 and dct_block.shape[1] > 4:
                    dct_block[3, 4] = coeff1
                    dct_block[4, 3] = coeff2
                else:
                    mid = block_size // 2
                    if mid + 1 < block_size:
                        dct_block[mid, mid + 1] = coeff1
                        dct_block[mid + 1, mid] = coeff2
                    else:
                        dct_block[mid, mid] = (coeff1 + coeff2) / 2

                bit_idx += 1

            # 逆DCT
            idct_block = cv2.idct(dct_block)
            y_ch[y:y + block_size, x:x + block_size] = idct_block

        # 合并通道
        yuv_img[:, :, 0] = np.clip(y_ch, 0, 255)
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

    def extract_robust_watermark(self, img):
        """提取鲁棒水印"""
        # 调整尺寸
        if img.shape != self.orig_dims:
            img = cv2.resize(img, (self.orig_dims[1], self.orig_dims[0]))

        # 转换到YUV
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_ch = yuv_img[:, :, 0].astype(np.float32)

        # 图像尺寸
        h, w = y_ch.shape
        block_size = self.block_dim
        blocks_y = h // block_size
        blocks_x = w // block_size

        # 随机块索引
        block_indices = [(i * block_size, j * block_size)
                         for i in range(blocks_y)
                         for j in range(blocks_x)]
        random.seed(self.rand_seed)
        random.shuffle(block_indices)

        # 提取水印位
        bit_data = []
        extracted_count = 0
        max_bits = (blocks_y * blocks_x) // 8 * 8

        for block_pos in tqdm(block_indices, desc="提取水印", unit="块"):
            if extracted_count >= max_bits:
                break

            y, x = block_pos
            if y + block_size > h or x + block_size > w:
                continue

            block = y_ch[y:y + block_size, x:x + block_size]
            dct_block = cv2.dct(block)

            # 提取系数
            if dct_block.shape[0] > 4 and dct_block.shape[1] > 4:
                coeff1 = dct_block[3, 4]
                coeff2 = dct_block[4, 3]
            else:
                mid = block_size // 2
                coeff1 = dct_block[mid, mid + 1] if mid + 1 < block_size else dct_block[mid, mid]
                coeff2 = dct_block[mid + 1, mid] if mid + 1 < block_size else dct_block[mid, mid]

            # 提取位
            if coeff1 > coeff2:
                bit_val = '1'
            else:
                bit_val = '0'

            bit_data.append(bit_val)
            extracted_count += 1

            # 提前检查
            if extracted_count >= 16:
                wm_text = self._decode_bits(bit_data)
                if not wm_text.startswith("提取失败"):
                    return wm_text

        return self._decode_bits(bit_data)

    def _decode_bits(self, bit_list):
        """解析二进制数据"""
        byte_data = []
        for i in range(0, len(bit_list), 8):
            if i + 8 > len(bit_list):
                break
            byte_str = ''.join(bit_list[i:i + 8])
            byte_data.append(int(byte_str, 2))

        # 检查长度
        if len(byte_data) < 2:
            return "提取失败: 数据过短"

        # 长度前缀
        try:
            text_len = int.from_bytes(bytes(byte_data[:2]), 'big')
        except:
            return "提取失败: 无效长度"

        # 检查完整性
        if len(byte_data) < 4 + text_len:
            return f"提取失败: 数据不完整 (需要 {4 + text_len} 字节, 实际 {len(byte_data)} 字节)"

        # 提取数据
        text_bytes = bytes(byte_data[2:2 + text_len])
        checksum_bytes = bytes(byte_data[2 + text_len:4 + text_len])
        stored_cs = int.from_bytes(checksum_bytes, 'big')

        # 验证校验和
        actual_cs = sum(text_bytes) % 65536
        if stored_cs != actual_cs:
            return f"提取失败: 校验和错误 (存储: {stored_cs}, 计算: {actual_cs})"

        # 解码文本
        try:
            return text_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            return f"提取失败: 解码错误 - {str(e)}"

    def add_visible_watermark(self, text=None, position=(30, 50), font_size=30, alpha=0.5):
        """添加可见水印"""
        if text is None:
            text = self.wm_content
        if not text:
            raise ValueError("请提供水印文本")

        # OpenCV转PIL
        img_pil = Image.fromarray(cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        # 字体处理
        try:
            font_path = self._get_chinese_font_path()
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
                print("警告: 使用默认字体，中文可能异常")
        except Exception as e:
            print(f"字体错误: {str(e)}")
            font = ImageFont.load_default()

        # 文本位置
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        img_w, img_h = img_pil.size
        x = max(10, min(position[0], img_w - text_w - 10))
        y = max(text_h + 10, min(position[1], img_h - 10))

        # 添加水印
        draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * alpha)))

        # PIL转OpenCV
        watermarked = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.wm_img = watermarked
        return watermarked

    def _get_chinese_font_path(self):
        """获取中文字体路径"""
        os_type = platform.system()
        try:
            if os_type == "Windows":
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/simsun.ttc",
                    "C:/Windows/Fonts/msyh.ttc",
                ]
            elif os_type == "Darwin":  # macOS
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                ]
            else:  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                ]

            # 检查字体
            for path in font_paths:
                if os.path.exists(path):
                    return path

            # 查找其他字体
            for font in fm.findSystemFonts():
                if "hei" in font.lower() or "black" in font.lower():
                    return font
                if "song" in font.lower() or "sun" in font.lower():
                    return font

            return None

        except Exception as e:
            print(f"字体路径错误: {str(e)}")
            return None

    def save_image(self, img, path):
        """保存图像"""
        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            path += '.png'
        cv2.imwrite(path, img)
        return path

    def apply_attacks(self, img):
        """应用攻击测试"""
        results = {}
        h, w = img.shape[:2]

        # 各种攻击
        results["水平翻转"] = cv2.flip(img, 1)
        results["垂直翻转"] = cv2.flip(img, 0)

        # 平移
        M = np.float32([[1, 0, 30], [0, 1, 20]])
        results["平移"] = cv2.warpAffine(img, M, (w, h))

        # 裁剪
        crop_h, crop_w = int(h * 0.8), int(w * 0.8)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        results["裁剪"] = img[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 对比度调整
        results["高对比度"] = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        results["低对比度"] = cv2.convertScaleAbs(img, alpha=0.7, beta=0)

        # 亮度调整
        results["高亮度"] = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
        results["低亮度"] = cv2.convertScaleAbs(img, alpha=1.0, beta=-30)

        # 噪声
        noise = np.zeros_like(img, dtype=np.uint8)
        cv2.randn(noise, 0, 20)
        results["高斯噪声"] = cv2.add(img, noise)

        # 旋转
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1)
        results["旋转10度"] = cv2.warpAffine(img, rot_mat, (w, h))

        # 缩放
        results["放大"] = cv2.resize(img, (int(w * 1.1), int(h * 1.1)), interpolation=cv2.INTER_LINEAR)
        results["缩小"] = cv2.resize(img, (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_LINEAR)

        # 模糊
        results["轻度模糊"] = cv2.GaussianBlur(img, (5, 5), 0)
        results["重度模糊"] = cv2.GaussianBlur(img, (9, 9), 0)

        # JPEG压缩
        cv2.imwrite("temp.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        results["JPEG80"] = cv2.imread("temp.jpg")
        cv2.imwrite("temp.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        results["JPEG50"] = cv2.imread("temp.jpg")
        os.remove("temp.jpg")

        # 椒盐噪声
        noisy = img.copy()
        noise_count = int(0.01 * h * w)
        for _ in range(noise_count):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            noisy[y, x] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        results["椒盐噪声"] = noisy

        # 锐化
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        results["锐化"] = cv2.filter2D(img, -1, kernel)

        # 添加文字
        text_img = img.copy()
        cv2.putText(text_img, "额外文字", (w // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        results["添加文字"] = text_img

        # 颜色转换
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results["灰度"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return results

    def test_robustness(self):
        """测试水印鲁棒性"""
        if self.wm_img is None:
            raise ValueError("请先嵌入水印")

        # 应用攻击
        attack_results = self.apply_attacks(self.wm_img)

        # 保存结果
        os.makedirs("robustness_results", exist_ok=True)
        for name, img in attack_results.items():
            cv2.imwrite(f"robustness_results/{name}.png", img)

        # 测试提取
        report = []
        for name, img in tqdm(attack_results.items(), desc="鲁棒性测试", unit="测试"):
            try:
                extracted = self.extract_robust_watermark(img)
                status = "成功" if extracted == self.wm_content else "失败"
                report.append({
                    "测试": name,
                    "结果": extracted,
                    "状态": status,
                    "图像": img
                })
            except Exception as e:
                report.append({
                    "测试": name,
                    "结果": f"错误: {str(e)}",
                    "状态": "失败",
                    "图像": img
                })

        return report

    def visualize_results(self, report):
        """可视化测试结果"""
        num = len(report)
        if num == 0:
            print("无测试结果可显示")
            return

        plt.figure(figsize=(15, num * 4))

        for i, res in enumerate(report):
            # 显示图像
            plt.subplot(num, 2, i * 2 + 1)
            plt.imshow(cv2.cvtColor(res["图像"], cv2.COLOR_BGR2RGB))
            plt.title(f'测试: {res["测试"]}')
            plt.axis('off')

            # 显示结果
            plt.subplot(num, 2, i * 2 + 2)
            color = 'green' if res["状态"] == "成功" else 'red'
            plt.text(0.1, 0.5, f"状态: {res['状态']}\n结果: {res['结果']}",
                     fontsize=10, bbox=dict(facecolor=color, alpha=0.3))
            plt.title('水印提取')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("robustness_results.png", dpi=150)
        plt.show()

    def visualize_dct(self, img=None):
        """可视化DCT域水印"""
        if img is None:
            if self.wm_img is None:
                raise ValueError("无水印图像")
            img = self.wm_img

        # 转换到YUV
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_ch = yuv_img[:, :, 0].astype(np.float32)

        # 图像尺寸
        h, w = y_ch.shape
        block_size = self.block_dim
        blocks_y = h // block_size
        blocks_x = w // block_size

        # 创建可视化
        dct_viz = np.zeros_like(y_ch)

        for i in range(blocks_y):
            for j in range(blocks_x):
                y = i * block_size
                x = j * block_size
                if y + block_size > h or x + block_size > w:
                    continue

                block = y_ch[y:y + block_size, x:x + block_size]
                dct_block = cv2.dct(block)

                # 提取系数差异
                if dct_block.shape[0] > 4 and dct_block.shape[1] > 4:
                    coeff1 = abs(dct_block[3, 4])
                    coeff2 = abs(dct_block[4, 3])
                else:
                    mid = block_size // 2
                    coeff1 = abs(dct_block[mid, mid + 1]) if mid + 1 < block_size else abs(dct_block[mid, mid])
                    coeff2 = abs(dct_block[mid + 1, mid]) if mid + 1 < block_size else abs(dct_block[mid, mid])

                diff = abs(coeff1 - coeff2)
                dct_viz[y:y + block_size, x:x + block_size] = diff * 10

        # 归一化
        dct_viz = cv2.normalize(dct_viz, None, 0, 255, cv2.NORM_MINMAX)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('含水印图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(dct_viz, cmap='hot')
        plt.colorbar(label='水印强度')
        plt.title('DCT域分布')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("dct_visualization.png", dpi=150)
        plt.show()

    def compute_psnr(self, img1, img2):
        """计算PSNR"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_val = 255.0
        return 20 * math.log10(max_val / math.sqrt(mse))

    def compute_ssim(self, img1, img2):
        """计算SSIM"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    def evaluate_quality(self):
        """评估图像质量"""
        if self.src_img is None or self.wm_img is None:
            raise ValueError("需要原始和水印图像")

        psnr = self.compute_psnr(self.src_img, self.wm_img)
        ssim = self.compute_ssim(self.src_img, self.wm_img)

        print(f"图像质量评估:")
        print(f"PSNR: {psnr:.2f} dB (越高越好)")
        print(f"SSIM: {ssim:.4f} (接近1最好)")

        return {"PSNR": psnr, "SSIM": ssim}


def execute():
    # 创建处理器
    processor = AdvancedWatermarkProcessor()

    try:
        # 加载图像
        input_img = "input.jpg"
        processor.load_img(input_img)
        print(f"加载图像: {input_img}")

        # 设置水印
        wm_text = "机密文档2023"
        processor.set_watermark(wm_text)
        print(f"设置水印: {wm_text}")

        # 嵌入鲁棒水印
        print("\n嵌入鲁棒水印...")
        start = time.time()
        wm_image = processor.embed_robust_watermark()
        embed_time = time.time() - start
        saved_path = processor.save_image(wm_image, "watermarked_robust.png")
        print(f"嵌入完成, 耗时: {embed_time:.2f}秒")
        print(f"保存路径: {saved_path}")

        # 提取水印
        print("\n提取水印...")
        start = time.time()
        extracted = processor.extract_robust_watermark(wm_image)
        extract_time = time.time() - start
        print(f"提取耗时: {extract_time:.2f}秒")
        print(f"提取结果: {extracted}")

        # 质量评估
        processor.evaluate_quality()

        # 可视化
        print("\n可视化...")
        processor.visualize_dct()

        # 添加可见水印
        print("\n添加可见水印...")
        visible_wm = processor.add_visible_watermark("版权所有")
        saved_path = processor.save_image(visible_wm, "watermarked_visible.jpg")
        print(f"保存路径: {saved_path}")

        # 显示可见水印
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(visible_wm, cv2.COLOR_BGR2RGB))
        plt.title('可见水印')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("visible_watermark.png", dpi=150)
        plt.show()

        # 鲁棒性测试
        print("\n鲁棒性测试...")
        start = time.time()
        report = processor.test_robustness()
        test_time = time.time() - start
        print(f"测试完成, 耗时: {test_time:.2f}秒")

        # 打印报告
        print("\n鲁棒性测试报告:")
        print("-" * 70)
        print(f"{'测试':<15} | {'状态':<8} | {'结果'}")
        print("-" * 70)
        success = 0
        for test in report:
            status = test["状态"]
            if status == "成功":
                success += 1
            print(f"{test['测试']:<15} | {status:<8} | {test['结果']}")

        success_rate = success / len(report) * 100
        print("-" * 70)
        print(f"成功率: {success_rate:.2f}% ({success}/{len(report)})")

        # 可视化结果
        if report:
            processor.visualize_results(report)
        else:
            print("无测试结果可显示")

        print("\n处理完成!")

    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    execute()