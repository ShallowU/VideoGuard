import requests
import time
url = "http://127.0.0.1:8000/process"
files = {"video": open("test-video/normal-littlechild.mp4", "rb")}
print("time now:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
t0 = time.time()
response = requests.post(url, files=files)
# print(response.json())
t1 = time.time()
print("time end:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(f"请求耗时: {t1 - t0:.2f}秒")