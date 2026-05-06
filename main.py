import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 生成模拟用户数据
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "view_count": np.random.randint(1, 100, n),
    "like_count": np.random.randint(0, 50, n),
    "comment_count": np.random.randint(0, 20, n),
    "share_count": np.random.randint(0, 10, n),
})

# 2. 构造标签（是否消费）
df["label"] = ((df["view_count"] * 0.1 +
                df["like_count"] * 0.3 +
                df["comment_count"] * 0.4 +
                df["share_count"] * 0.2) > 20).astype(int)

# 3. 特征与标签
X = df[["view_count", "like_count", "comment_count", "share_count"]]
y = df["label"]

# 4. 划分训练集和测试集
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 预测
y_pred = model.predict(X_test)

# 7. 评估
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 8. 可视化
plt.figure(figsize=(8,4))
plt.plot(y_test.values[:50], label="True")
plt.plot(y_pred[:50], label="Pred")
plt.legend()
plt.title("User Consumption Prediction")
plt.show()