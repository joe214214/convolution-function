import os
import subprocess
import modal

LOCAL_PROJECT_DIR = r"D:\python_project_graduate\convolution function\mpi"
REMOTE_PROJECT_DIR = "/root/app"

app = modal.App("mpi_conv_job")

image = (
    modal.Image.debian_slim()
    .apt_install([
        "openmpi-bin",
        "libopenmpi-dev",
    ])
    .pip_install([
        "numpy",
        "opencv-python-headless",  # 用无图形版的 OpenCV
        "numba",
        "mpi4py",
        "matplotlib",
    ])
    .add_local_dir(
        local_path=LOCAL_PROJECT_DIR,
        remote_path=REMOTE_PROJECT_DIR,
    )
)

@app.function(
    image=image,
    cpu=32,
    memory="32GiB",
    timeout=600,
)
def run_job():
    print("初始工作目录:", os.getcwd())
    print("容器里 /root 内容:", os.listdir("/root"))

    os.chdir(REMOTE_PROJECT_DIR)
    print("切换后工作目录:", os.getcwd())
    print("工程目录文件列表:", os.listdir("."))

    cmd = [
        "mpiexec",
        "--allow-run-as-root",
        "-n", "2",
        "python",
        "mpi_conv.py",
        "--synthetic", "2048", "2048",
        "--kernel-size", "7",
        "--stride", "1",
        "--padding", "same",
        "--px", "2", "--py", "1",
        "--csv", "results_modal.csv",
        # "--numba-disable",
        "--baseline", " 1.79",

    ]

    print(">>> running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ 运行完成，results_modal.csv 已生成")

    if os.path.exists("results_modal.csv"):
        with open("results_modal.csv", "r", encoding="utf-8") as f:
            data = f.read()
        print("results_modal.csv 预览 (前500字符):")
        print(data[:500])
        return data
    else:
        print("⚠ 没找到 results_modal.csv")
        return ""
