import pandas as pd
import glob
import os
import shutil

def convert_all_xlsx_to_csv():
    """把所有xlsx转成csv"""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # 备份原xlsx文件（可选）
    backup_dir = os.path.join(data_dir, 'xlsx_backup')
    os.makedirs(backup_dir, exist_ok=True)

    xlsx_files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))

    print(f"找到 {len(xlsx_files)} 个xlsx文件\n")

    for xlsx_file in xlsx_files:
        filename = os.path.basename(xlsx_file)
        csv_filename = filename.replace('.xlsx', '.csv')
        csv_file = os.path.join(data_dir, csv_filename)

        print(f"正在转换: {filename}")

        try:
            df = pd.read_excel(xlsx_file)
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"  ✓ 已保存: {csv_filename} (形状: {df.shape})")

            # 备份原文件
            shutil.copy2(xlsx_file, os.path.join(backup_dir, filename))
            print(f"  ✓ 已备份到: xlsx_backup/{filename}")

        except Exception as e:
            print(f"  ✗ 转换失败: {e}")

        print()

    print("=" * 60)
    print("全部转换完成！")
    print("=" * 60)

    # 检查生成的文件
    print("\n生成的CSV文件:")
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, nrows=2)
        print(f"  {os.path.basename(csv_file)}: {df.shape[0]-1}行表头 + {pd.read_csv(csv_file).shape[0]}行数据, {df.shape[1]}列")

if __name__ == '__main__':
    convert_all_xlsx_to_csv()
