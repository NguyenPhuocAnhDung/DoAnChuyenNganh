import pandas as pd
import os
import glob

# Chỉ cần trỏ đúng thư mục gốc
INPUT_ROOT_DIR = r"D:\DACN\dataset\raw\CICEVSE2024_Dataset\Network Traffic"

def find_attacker_auto():
    # Tự động tìm tất cả file csv
    search_path = os.path.join(INPUT_ROOT_DIR, "**", "*Aggressive-scan.csv")
    files = glob.glob(search_path, recursive=True)
    
    # Nếu không tìm thấy file cụ thể, lấy file csv bất kỳ
    if not files:
        files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**", "*.csv"), recursive=True)

    if not files:
        print(f"❌ Vẫn không tìm thấy file nào trong: {INPUT_ROOT_DIR}")
        return

    # Lấy file đầu tiên tìm được
    target_file = files[0]
    print(f"🕵️‍♂️ Đang phân tích file: {os.path.basename(target_file)}")
    print(f"📂 Đường dẫn: {target_file}")
    
    try:
        # Đọc 2 cột IP để thống kê
        df = pd.read_csv(target_file, usecols=['src_ip', 'dst_ip'])
        
        print("\n" + "="*40)
        print("🏆 TOP 5 IP GỬI NHIỀU NHẤT (Nghi phạm Attacker)")
        print("="*40)
        print(df['src_ip'].value_counts().head(5))
        
        print("\n" + "="*40)
        print("🎯 TOP 5 IP NHẬN NHIỀU NHẤT (Nạn nhân)")
        print("="*40)
        print(df['dst_ip'].value_counts().head(5))
        
    except ValueError:
        print("❌ File này không có cột 'src_ip' hoặc 'dst_ip'.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    find_attacker_auto()