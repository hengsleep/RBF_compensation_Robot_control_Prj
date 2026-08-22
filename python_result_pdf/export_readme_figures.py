from pathlib import Path
import subprocess

# 精选出的 5 张核心 PDF
SELECTED_PDFS = [
    "Sim1_S=1 Tracking.pdf",
    "Sim1_S=3_S1=2_m=19 Tracking.pdf",
    "Sim2_S=3_S1=2_m=5 Position Tracking.pdf",
    "Sim2_S=3_S1=2_m=5 Speed Tracking.pdf",
    "Sim2_S=3_S1=2_m=5 Disturbance Estimation.pdf",
]

WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "readme_figures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 工作目录: {WORK_DIR}")
    print(f"📂 PNG 输出目录: {OUTPUT_DIR}\n")

    success_count = 0

    for pdf_name in SELECTED_PDFS:
        pdf_path = WORK_DIR / pdf_name

        if not pdf_path.exists():
            print(f"⚠️ 跳过未找到的文件: {pdf_name}")
            continue

        # 构造输出文件名 (去除 .pdf 后缀)
        output_prefix = OUTPUT_DIR / pdf_path.stem

        # 直接调用系统的 pdftoppm 工具转高精度 PNG (-r 300 表示 300 DPI)
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-f",
            "1",
            "-l",
            "1",
            str(pdf_path),
            str(output_prefix),
        ]

        try:
            subprocess.run(cmd, check=True)
            # pdftoppm 默认输出文件名为 prefix-1.png，将其重命名为干净的文件名
            generated_file = OUTPUT_DIR / f"{pdf_path.stem}-1.png"
            target_file = OUTPUT_DIR / f"{pdf_path.stem}.png"

            if generated_file.exists():
                generated_file.rename(target_file)

            print(f"✅ 成功导出: {target_file.name}")
            success_count += 1
        except Exception as e:
            print(f"❌ 转换失败 [{pdf_name}]: {e}")

    print(
        f"\n🎉 处理完成! 共导出 {success_count}/{len(SELECTED_PDFS)} 张图片。"
    )


if __name__ == "__main__":
    main()