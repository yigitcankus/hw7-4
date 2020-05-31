import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.over_sampling import ADASYN
import numpy as np

warnings.filterwarnings('ignore')

df = pd.read_csv("cleveland-0_vs_4.csv")
df["ca"] = df.ca.replace({'<null>':0})
df["ca"] = df["ca"].astype(np.int64)

df["thal"] = df.thal.replace({'<null>':0})
df["thal"] = df["thal"].astype(np.int64)

df["target"] = df.target.replace({'negative':0, "positive":1})

sns.countplot(df.target)
plt.show()
print("Kalp krizi olur oranı : %{:.2f}".format(sum(df.target)/len(df.target)*100))
print("Kalp krizi olmaz oranı   : %{:.2f}".format((len(df.target)-sum(df.target))/len(df.target)*100))

X = df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y = df["target"]

X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=112)

logreg_model = LogisticRegression()
logreg_model.fit(X_eğitim, y_eğitim)
tahmin_eğitim = logreg_model.predict(X_eğitim)
tahmin_test = logreg_model.predict(X_test)
hata_matrisi_eğitim = confusion_matrix(y_eğitim, tahmin_eğitim)
hata_matrisi_test = confusion_matrix(y_test, tahmin_test)

print("Veriye herhangi bir şey eklenmeden önceki hali--")
print("Modelin doğruluk değeri : ",  logreg_model.score(X_test, y_test))
print("Eğitim veri kümesi")
print(classification_report(y_eğitim,tahmin_eğitim) )
print("Test veri kümesi")
print(classification_report(y_test,tahmin_test))

# Modelimizin doğruluk değeri : 0.94
# Hassasiyeti : 1.00
# Duyarlılık : 0.50
# F1 değeri : 0.67

#Çok az verimiz olduğu için garip değerler çıkabiliyor. Veri sayımızı arttırmamız lazim.

saglikli_df = df[df.target == 0]
hastalikli_df = df[df.target == 1]

hastalikli_df = resample(hastalikli_df,
                                     replace = True,
                                     n_samples = len(saglikli_df),
                                     random_state = 111)

artırılmıs_df = pd.concat([saglikli_df, hastalikli_df])
# print(artırılmıs_df.target.value_counts())


X = artırılmıs_df.drop('target', axis=1)
y = artırılmıs_df['target']

X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=112)

logreg_model = LogisticRegression()
logreg_model.fit(X_eğitim, y_eğitim)
tahmin_eğitim = logreg_model.predict(X_eğitim)
tahmin_test = logreg_model.predict(X_test)
hata_matrisi_eğitim = confusion_matrix(y_eğitim, tahmin_eğitim)
hata_matrisi_test = confusion_matrix(y_test, tahmin_test)

print("Modelin azınlık tarafı arttırıldıktan sonraki hali--")
print("Modelin doğruluk değeri : ",  logreg_model.score(X_test, y_test))
print("Eğitim veri kümesi")
print(classification_report(y_eğitim,tahmin_eğitim) )
print("Test veri kümesi")
print(classification_report(y_test,tahmin_test))

# Verileri arttırdıktan sonra

# Modelimizin doğruluk değeri : 0.95
# Hassasiyeti : 0.92
# Duyarlılık : 1.00
# F1 değeri : 0.96

#modelimiz iyiye gidiyor bir de ADASYN sentetik örnek arttırmasını deneyelim.

ad = ADASYN()
X_adasyn, y_adasyn = ad.fit_sample(X, y)

X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X_adasyn, y_adasyn, test_size=0.20, random_state=111, stratify = y)
logreg_model = LogisticRegression()
logreg_model.fit(X_eğitim, y_eğitim)

tahmin_eğitim = logreg_model.predict(X_eğitim)
tahmin_test = logreg_model.predict(X_test)
hata_matrisi_eğitim = confusion_matrix(y_eğitim, tahmin_eğitim)
hata_matrisi_test = confusion_matrix(y_test, tahmin_test)

print("Adasyn uygulandıktan sonra--")
print("Modelin doğruluk değeri : ",  logreg_model.score(X_test, y_test))
print("Eğitim veri kümesi")
print(classification_report(y_eğitim,tahmin_eğitim) )
print("Test veri kümesi")
print(classification_report(y_test,tahmin_test) )

# Modelimizin doğruluk değeri : 0.969
# Hassasiyeti : 0.94
# Duyarlılık : 1.00
# F1 değeri : 0.97

#modelimizin en iyi hali bu oldu.










