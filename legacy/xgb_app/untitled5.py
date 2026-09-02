# eda.py —— 生成箱型图/小提琴图/分布/相关性 等 EDA 图
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------- 可按你数据修改的配置 -----------
# 数值列（建议与你的 utils.py 中 NUMERIC_COLS 一致或取其子集）
NUMERIC_COLS = [
    'Age','BMI','AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality',
    'SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL',
    'CholesterolTriglycerides','MMSE','FunctionalAssessment','ADL'
]
# 类别/二值列
CATEGORICAL_COLS = [
    'Gender','Ethnicity','EducationLevel','Smoking',
    'MemoryComplaints','BehavioralProblems','Confusion','Disorientation',
    'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness'
]
TARGET = 'Diagnosis'   # 0=未确诊 1=确诊（请与你数据一致）

# 中英对照（可按需添加）
CN = {
    'Age':'年龄', 'BMI':'体重指数', 'AlcoholConsumption':'饮酒', 'PhysicalActivity':'体力活动',
    'DietQuality':'饮食质量', 'SleepQuality':'睡眠质量', 'SystolicBP':'收缩压',
    'DiastolicBP':'舒张压','CholesterolTotal':'总胆固醇','CholesterolLDL':'低密度胆固醇',
    'CholesterolHDL':'高密度胆固醇','CholesterolTriglycerides':'甘油三酯',
    'MMSE':'简易精神状态检查','FunctionalAssessment':'功能评估','ADL':'日常生活能力',
    'Gender':'性别','Ethnicity':'族别','EducationLevel':'受教育水平','Smoking':'吸烟',
    'MemoryComplaints':'记忆抱怨','BehavioralProblems':'行为问题','Confusion':'意识混乱',
    'Disorientation':'定向障碍','PersonalityChanges':'人格改变','DifficultyCompletingTasks':'完成任务困难',
    'Forgetfulness':'健忘','Diagnosis':'阿尔茨海默症诊断'
}
# -------------------------------------------

def ensure_cols(df, cols):
    return [c for c in cols if c in df.columns]

def mkdir(p):
    os.makedirs(p, exist_ok=True); return p

def label_en_cn(name):
    return f"{name} / {CN.get(name, name)}"

def grouped_boxplot(df, cols, target, outdir, suffix='box'):
    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(6,4))
            data0 = df[df[target]==0][col].dropna()
            data1 = df[df[target]==1][col].dropna()
            ax.boxplot([data0, data1], labels=['0-No AD','1-AD'])
            ax.set_title(f"Boxplot: {label_en_cn(col)} by {label_en_cn(target)}")
            ax.set_ylabel(label_en_cn(col))
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{col}_{suffix}.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[boxplot] skip {col}: {e}")

def grouped_violin(df, cols, target, outdir, suffix='violin'):
    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(6,4))
            data0 = df[df[target]==0][col].dropna().values
            data1 = df[df[target]==1][col].dropna().values
            parts = ax.violinplot([data0, data1], showmeans=True, showextrema=True)
            ax.set_xticks([1,2]); ax.set_xticklabels(['0-No AD','1-AD'])
            ax.set_title(f"Violin: {label_en_cn(col)} by {label_en_cn(target)}")
            ax.set_ylabel(label_en_cn(col))
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{col}_{suffix}.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[violin] skip {col}: {e}")

def hist_kde(df, cols, outdir):
    for col in cols:
        try:
            x = df[col].dropna().values
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(x, bins=30, alpha=0.7, density=True)
            # 简单KDE（自写，避免依赖 seaborn）
            if len(x) > 10:
                from scipy.stats import gaussian_kde
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                kde = gaussian_kde(x)
                ax.plot(xs, kde(xs))
            ax.set_title(f"Distribution: {label_en_cn(col)}")
            ax.set_xlabel(label_en_cn(col)); ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{col}_dist.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[hist] skip {col}: {e}")

def cat_bar(df, cols, outdir):
    for col in cols:
        try:
            cnt = df[col].value_counts(dropna=False).sort_index()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar([str(k) for k in cnt.index], cnt.values)
            ax.set_title(f"Counts: {label_en_cn(col)}")
            ax.set_xlabel(label_en_cn(col)); ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{col}_bar.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[bar] skip {col}: {e}")

def corr_heatmap(df, cols, outdir, name='corr_heatmap.png'):
    cols = ensure_cols(df, cols)
    if len(cols) < 2: return
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6,len(cols)*0.5), max(5,len(cols)*0.5)))
    cax = ax.matshow(corr.values, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90); ax.set_yticklabels(cols)
    ax.set_title("Correlation Heatmap / 相关性热力图", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)

def missingness(df, outdir):
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6,len(miss)*0.25), 4))
    ax.bar(miss.index.astype(str), (miss*100).values)
    ax.set_ylabel("Missing Rate (%)")
    ax.set_title("Missingness by Column / 缺失率")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "missingness.png"), dpi=150)
    plt.close(fig)

def main(args):
    df = pd.read_csv(args.data_csv)
    outdir = mkdir(os.path.join(args.out_dir, "eda"))

    # 保障列名存在
    num_cols = ensure_cols(df, NUMERIC_COLS)
    cat_cols = ensure_cols(df, CATEGORICAL_COLS + [TARGET])
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)

    # 1) 数值分布 + 缺失率
    hist_kde(df, num_cols, outdir)
    missingness(df, outdir)

    # 2) 箱型图/小提琴图（按 Diagnosis 分组）
    if TARGET in df.columns:
        grouped_boxplot(df, num_cols, TARGET, outdir)
        grouped_violin(df, num_cols, TARGET, outdir)

    # 3) 类别/二值分布
    cat_cols_plot = [c for c in CATEGORICAL_COLS if c in df.columns]
    cat_bar(df, cat_cols_plot, outdir)

    # 4) 相关性热力图（数值列）
    corr_heatmap(df, num_cols, outdir)

    print(f"✅ EDA 图已输出到：{outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="alzheimers_disease_patient_data.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()
    main(args)
