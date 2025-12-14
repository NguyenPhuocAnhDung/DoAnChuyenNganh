import platform
import psutil
import socket
import pandas as pd

# Thu thập thông tin hệ thống
config = {
    "OS": platform.platform(),
    "Processor": platform.processor(),
    "CPU Cores (Physical)": psutil.cpu_count(logical=False),
    "CPU Cores (Logical)": psutil.cpu_count(logical=True),
    "RAM (Total, GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    "Machine": platform.machine(),
    "Hostname": socket.gethostname(),
    "Python Version": platform.python_version()
}

# Tạo bảng và xuất ra file CSV
df = pd.DataFrame(config.items(), columns=["Property", "Value"])
df.to_csv("system_config.csv", index=False)

print("Đã xuất file cấu hình: system_config.csv")