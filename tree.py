import os
from pathlib import Path


def tree(dir_path: Path, prefix: str = ""):
    """递归打印目录树"""
    # 获取当前目录下的所有条目，并排序（文件夹在前，文件在后）
    entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    
    # 过滤掉隐藏文件/文件夹（以 . 开头的），如果不需要过滤可删掉这行
    entries = [e for e in entries if not e.name.startswith('.')]
    
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{entry.name}")
        
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            tree(entry, prefix + extension)


if __name__ == "__main__":
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).resolve().parent
    
    # 打印根目录名称
    print(f"{current_dir.name}/")
    
    # 开始打印树
    tree(current_dir)
