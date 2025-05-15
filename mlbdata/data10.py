import pandas as pd
import glob
import os

# ✅ 指定你的資料夾路徑（改成原始 CSV 檔所在位置）
folder_path = r"C:\Users\richc\OneDrive\桌面\專題\mlbdata"

# ✅ 找出所有符合命名規則的檔案
file_paths = glob.glob(os.path.join(folder_path, "mlb-player-stats-*.csv"))

# ✅ 檢查有沒有找到檔案
if not file_paths:
    print("⚠️ 沒有找到符合條件的 CSV 檔案，請檢查檔名或資料夾路徑")
else:
    print(f"✅ 找到 {len(file_paths)} 份檔案：")
    for f in file_paths:
        print(" -", os.path.basename(f))

# ✅ 讀取與加上年份欄位
all_dfs = []
for path in file_paths:
    filename = os.path.basename(path)
    try:
        year = int(filename.split('-')[-1].split('.')[0])
    except ValueError:
        print(f"⚠️ 無法從檔名擷取年份：{filename}")
        continue

    df = pd.read_csv(path)
    df["year"] = year
    all_dfs.append(df)

# ✅ 合併資料
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # 儲存
    combined_df.to_csv(os.path.join(folder_path, "mlb_player_stats_10years_combined.csv"), index=False)
    print("🎉 合併完成，共有資料列數：", len(combined_df))
else:
    print("❌ 沒有成功合併任何資料，請檢查檔案內容。")
