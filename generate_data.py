"""
纯Python数据生成脚本 (不依赖numpy)
"""
import random
import math
import os

os.makedirs("dataset", exist_ok=True)

# 设置随机种子
random.seed(42)

# 微污染物数据库
pollutants = [
    {"name": "BPA", "mw": 228, "charge": 0, "iogd": 3.32},
    {"name": "IBP", "mw": 206, "charge": -1, "iogd": 3.97},
    {"name": "SMX", "mw": 253, "charge": -1, "iogd": 0.89},
    {"name": "TCS", "mw": 289, "charge": -1, "iogd": 4.76},
    {"name": "ATZ", "mw": 215, "charge": 0, "iogd": 2.61},
    {"name": "NP", "mw": 220, "charge": 0, "iogd": 5.76},
    {"name": "CAF", "mw": 194, "charge": 0, "iogd": -0.07},
    {"name": "BA", "mw": 122, "charge": -1, "iogd": 1.87},
]

membranes = [
    {"name": "NF90", "mwco": 200, "jw": 12, "zeta": -15, "contact": 35},
    {"name": "NF270", "mwco": 400, "jw": 18, "zeta": -12, "contact": 25},
    {"name": "NF", "mwco": 300, "jw": 15, "zeta": -10, "contact": 40},
    {"name": "RO", "mwco": 100, "jw": 8, "zeta": -8, "contact": 45},
    {"name": "TFC-S", "mwco": 150, "jw": 10, "zeta": -20, "contact": 55},
]


def random_normal(mean, std):
    """Box-Muller transform for normal distribution"""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + std * z


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def generate_dataset(n_samples=3000):
    data = []
    
    for _ in range(n_samples):
        membrane = random.choice(membranes)
        pollutant = random.choice(pollutants)
        
        mwco = max(50, random_normal(membrane["mwco"], 30))
        jw = max(1, random_normal(membrane["jw"], 2))
        zeta = random_normal(membrane["zeta"], 3)
        contact = max(10, random_normal(membrane["contact"], 5))
        
        pollutant_mw = max(50, random_normal(pollutant["mw"], 10))
        pollutant_charge = max(-2, min(1, random_normal(pollutant["charge"], 0.2)))
        iogd = random_normal(pollutant["iogd"], 0.3)
        
        size_factor = sigmoid(0.02 * (pollutant_mw - mwco))
        charge_factor = sigmoid(0.5 * (pollutant_charge * zeta / 10))
        hydro_factor = sigmoid(0.3 * (iogd - 2))
        density_factor = sigmoid(0.05 * (mwco - 200))
        
        base_rejection = size_factor * 0.35 + charge_factor * 0.25 + hydro_factor * 0.15 + density_factor * 0.25
        rejection = base_rejection * 100 + random_normal(0, 5)
        rejection = max(0, min(99.9, rejection))
        
        data.append(f"{round(mwco,2)},{round(jw,2)},{round(pollutant_mw,2)},{round(zeta,2)},{round(pollutant_charge,2)},{round(contact,2)},{round(iogd,2)},{round(rejection,2)}")
    
    return data


# 生成数据
print("生成数据集...")
data = generate_dataset(3000)

# 写入CSV
header = "membrane_mwco,pure_water_flux,pollutant_mw,membrane_zeta,pollutant_charge,membrane_contact_angle,pollutant_iogd,rejection"
with open("dataset/membrane_dataset.csv", "w") as f:
    f.write(header + "\n")
    for row in data:
        f.write(row + "\n")

print("数据集已保存: dataset/membrane_dataset.csv")
print(f"共生成 {len(data)} 条数据")
