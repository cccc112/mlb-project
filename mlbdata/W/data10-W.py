import pandas as pd
import glob
import os

# 球隊名稱 → 縮寫對照表
team_name_to_abbr = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago White Sox": "CWS",
    "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",  
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYY",
    "New York Mets": "NYM",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH"
}

# 勝率資料夾
folder_path = r"C:\Users\richc\OneDrive\桌面\專題\mlbdata\W"
file_paths = glob.glob(os.path.join(folder_path, "W-*.csv"))

# 檢查檔案
if not file_paths:
    print("沒有找到符合條件的 CSV 檔案")
else:
    print(f"找到 {len(file_paths)} 份檔案：")
    for f in file_paths:
        print(" -", os.path.basename(f))

all_dfs = []
for path in file_paths:
    filename = os.path.basename(path)

    try:
        year = int(filename.split('-')[-1].split('.')[0])
    except ValueError:
        print(f"無法從檔名擷取年份：{filename}")
        continue

    try:
        df = pd.read_csv(path)

        # 刪除 'Season'
        if 'Season' in df.columns:
            df = df.drop('Season', axis=1)

        # 統一欄位名稱 Year → year
        if 'Year' in df.columns:
            df.rename(columns={'Year': 'year'}, inplace=True)

        # 加入年份欄位（若尚未存在）
        if 'year' not in df.columns:
            df['year'] = year

        # 將球隊名稱轉為縮寫
        if 'Team' in df.columns:
            df['Team'] = df['Team'].map(team_name_to_abbr).fillna(df['Team'])

        all_dfs.append(df)

    except Exception as e:
        print(f"錯誤發生於 {filename}: {e}")

# 合併儲存
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(folder_path, "mlb_player_stats_10years_combined-W.csv"), index=False)
    print("合併並轉換成功！總列數：", len(combined_df))
else:
    print("沒有成功合併任何資料。")
