import os, sys

# 1) 切到你的项目根目录（就是你截图里的这个）
os.chdir(r"D:\桌面\ad_risk_xgb_app_pro2\ad_risk_xgb_app_pro")
print("当前目录:", os.getcwd())
print("目录内容:", os.listdir())

# 2) 安装依赖（装过也可以再跑一遍，确保环境完整）
!{sys.executable} -m pip install --upgrade pip setuptools wheel
!{sys.executable} -m pip install -r requirements.txt

# 3) 训练：用同目录下的 CSV，产物写入 artifacts/
!{sys.executable} train.py --data_csv alzheimers_disease_patient_data.csv --artifacts_dir artifacts

# 4) 确认 artifacts 里已有模型/Scaler
import os
print("artifacts 内容:", os.listdir("artifacts"))

# 5) 启动 Flask 服务（如果端口被占用，稍后把 app.py 最后一行改成 port=5001 再跑）
!{sys.executable} app.py
