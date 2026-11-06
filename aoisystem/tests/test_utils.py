"""common.utils 工具函数单元测试。"""
import os
import json
import tempfile
import shutil
import unittest

from common.utils import (
    extract_number_from_path,
    generate_thumbnail,
    load_json_file,
    save_json_file,
    ensure_dir,
    list_image_files,
    IMAGE_EXTENSIONS,
)


class TestExtractNumberFromPath(unittest.TestCase):
    """extract_number_from_path 路径数字提取测试。"""

    def test_extract_with_underscore(self):
        self.assertEqual(extract_number_from_path("images/test_imgs3.tif"), 3)

    def test_extract_without_underscore(self):
        self.assertEqual(extract_number_from_path("data/abc12.png"), 12)

    def test_extract_multi_digit(self):
        self.assertEqual(extract_number_from_path("data/img_123.jpg"), 123)

    def test_no_number_returns_none(self):
        self.assertIsNone(extract_number_from_path("data/abc.png"))

    def test_full_linux_path(self):
        path = "/home/aoi/pretest/images/test_imgs1.tif"
        self.assertEqual(extract_number_from_path(path), 1)

    def test_windows_path(self):
        path = r"D:\aoi\material\img_5.bmp"
        self.assertEqual(extract_number_from_path(path), 5)


class TestJsonFileOperations(unittest.TestCase):
    """load_json_file / save_json_file 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        path = os.path.join(self.tmpdir, "data.json")
        data = {"料号": "ABC123", "板宽": 100, "A面": [1, 2, 3]}
        self.assertTrue(save_json_file(path, data))
        loaded = load_json_file(path)
        self.assertEqual(loaded, data)

    def test_load_nonexistent_returns_default(self):
        self.assertEqual(load_json_file("nonexistent.json"), {})
        self.assertEqual(load_json_file("nonexistent.json", default=[]), [])

    def test_load_invalid_json_returns_default(self):
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json content}")
        self.assertEqual(load_json_file(path), {})

    def test_save_ensure_ascii_false(self):
        """验证中文内容不被转义。"""
        path = os.path.join(self.tmpdir, "cn.json")
        save_json_file(path, {"名称": "测试"})
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("测试", content)


class TestEnsureDir(unittest.TestCase):
    """ensure_dir 目录创建测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_new_dir(self):
        path = os.path.join(self.tmpdir, "new_folder")
        ensure_dir(path, mode=None)
        self.assertTrue(os.path.isdir(path))

    def test_existing_dir_no_error(self):
        path = os.path.join(self.tmpdir, "existing")
        os.makedirs(path)
        ensure_dir(path, mode=None)  # 不应抛异常
        self.assertTrue(os.path.isdir(path))

    def test_nested_dir(self):
        path = os.path.join(self.tmpdir, "a", "b", "c")
        ensure_dir(path, mode=None)
        self.assertTrue(os.path.isdir(path))


class TestListImageFiles(unittest.TestCase):
    """list_image_files 图片文件列举测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_files(self, names):
        for name in names:
            path = os.path.join(self.tmpdir, name)
            with open(path, "w") as f:
                f.write("dummy")

    def test_list_images_sorted(self):
        self._create_files(["img3.tif", "img1.tif", "img2.tif", "notes.txt", "readme.md"])
        result = list_image_files(self.tmpdir)
        self.assertEqual(len(result), 3)
        # 验证排序
        self.assertTrue(result[0].endswith("img1.tif"))
        self.assertTrue(result[-1].endswith("img3.tif"))

    def test_list_empty_dir(self):
        result = list_image_files(self.tmpdir)
        self.assertEqual(result, [])

    def test_list_nonexistent_dir(self):
        result = list_image_files(os.path.join(self.tmpdir, "no_such_dir"))
        self.assertEqual(result, [])

    def test_case_insensitive_extensions(self):
        self._create_files(["a.JPG", "b.TIF", "c.png"])
        result = list_image_files(self.tmpdir)
        self.assertEqual(len(result), 3)


class TestGenerateThumbnail(unittest.TestCase):
    """generate_thumbnail 缩略图生成测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generate_thumbnail_success(self):
        from PIL import Image
        img = Image.new("RGB", (800, 600), color="red")
        path = generate_thumbnail(img, "test_img.tif", self.tmpdir,
                                  scale_factor=8, jpeg_quality=80)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("test_img_thumb.jpg"))

    def test_generate_thumbnail_non_rgb(self):
        """RGBA 图片应自动转 RGB。"""
        from PIL import Image
        img = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
        path = generate_thumbnail(img, "alpha.png", self.tmpdir,
                                  scale_factor=4, jpeg_quality=70)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_generate_thumbnail_size_reduced(self):
        from PIL import Image
        img = Image.new("RGB", (1600, 1200), color="blue")
        path = generate_thumbnail(img, "big.tif", self.tmpdir,
                                  scale_factor=8, jpeg_quality=80)
        result = Image.open(path)
        self.assertEqual(result.size, (200, 150))


if __name__ == "__main__":
    unittest.main()
