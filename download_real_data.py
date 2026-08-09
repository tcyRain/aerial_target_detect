# -*- coding: utf-8 -*-
"""下载真实航拍数据集（可选，需要联网授权）。

默认下载 UCMerced_LandUse（21 类 x 100 张，256x256 航拍图，约 317MB），
下载后可用它替换/扩充合成数据做迁移学习。
"""
import os
import urllib.request


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{100.0 * done / total:.1f}% ({done // 1024 // 1024}MB / {total // 1024 // 1024}MB)",
                      end="", flush=True)
    print()


def main():
    url = "https://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "UCMerced_LandUse.zip")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("下载 UCMerced 航拍土地分类数据集 ->", dest)
    download(url, dest)
    print("完成。解压后每个类别一个文件夹，可直接迁移到分类训练。")
    print()
    print("检测方向真实数据参考（较大，按需下载）：")
    print("  VisDrone: https://github.com/VisDrone/VisDrone-Dataset")
    print("  UAVDT:    https://sites.google.com/site/daviddo0323/projects/uavdt")


if __name__ == "__main__":
    main()
