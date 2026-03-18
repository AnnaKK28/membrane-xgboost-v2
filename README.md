# 膜性能预测XGBoost大模型

基于XGBoost算法，预测膜对微量有机污染物的截留率。

## 特征说明

| 特征名 | 描述 | 单位 |
|--------|------|------|
| membrane_mwco | 膜截留分子量 | Da |
| pure_water_flux | 纯水通量 | LMH |
| pollutant_mw | 微量有机污染物分子量 | Da |
| membrane_zeta | 中性时膜表面zeta电位 | mV |
| pollutant_charge | 中性时污染物表面电荷 | - |
| membrane_contact_angle | 膜的水接触角 | ° |
| pollutant_iogd | 中性时污染物的IogD | - |

## 目标变量

- **rejection**: 膜对微量有机污染物的截留率 (%)

## 模型特点

- 多随机种子重复划分 (8个种子)
- 8:2 训练/测试集划分
- 5折交叉验证超参数优化
- RMSE和R²评价指标
- SHAP模型解释

## 安装依赖

```bash
pip install xgboost shap pandas numpy scikit-learn matplotlib seaborn
```

## 运行训练

```bash
python train_model.py
```

## 文件说明

- train_model.py - 主训练脚本
- generate_data.py - 数据生成脚本
- dataset/membrane_dataset_sample.csv - 数据集样本(500条)

## 参考文献

1. Xu et al., J. Membr. Sci. (2006)
2. Kiso et al., Water Res. (2011)
3. Yang et al., Chem. Eng. J. (2019)

