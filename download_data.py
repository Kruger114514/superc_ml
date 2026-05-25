"""
下载 SuperCon 超导数据集 (Hamidieh 2018)
来源: UCI ML Repository
内容: 21263 条记录，每条包含化学组分 + Tc + 81 个特征
"""
import os
import zipfile
import urllib.request

URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
DATA_DIR = "data/raw"

os.makedirs(DATA_DIR, exist_ok=True)
zip_path = os.path.join(DATA_DIR, "superconductivity.zip")

print(f"下载中: {URL}")
urllib.request.urlretrieve(URL, zip_path)
print(f"已保存到: {zip_path}")

print("\n解压中...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(DATA_DIR)
    print(f"解压完成。包含文件:")
    for name in z.namelist():
        print(f"  - {name}")

print("\n数据准备完成,下一步运行 baseline 脚本。")