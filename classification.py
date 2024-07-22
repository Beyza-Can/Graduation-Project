import pandas as pd

# Veri setini oku (veri setinin adı ve yolunu değiştirin)
df = pd.read_csv('Cancer.csv')

# İlk sütunu (Patient Id) drop et
df = df.drop(columns=['Patient Id'])

# Son sütundaki kategorik değerleri numerik değerlere dönüştür
df['Level'] = df['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Sonucu göster (isteğe bağlı olarak)
print(df.head())  # İlk beş satırı yazdırır

# Eğer dönüşümü kaydetmek istiyorsanız:
df.to_csv('converted_dataset.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri setini oku (veri setinin adı ve yolunu değiştirin)
df = pd.read_csv('converted_dataset.csv')

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = df.drop(columns=['Level'])  # Level sütununu bağımsız değişkenlerden çıkar
y = df['Level']  # Level sütunu bağımlı değişken olarak al

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendir (Opsiyonel olarak)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression modeli oluşturma ve eğitme
logreg = LogisticRegression(C=0.001, solver='liblinear',class_weight='balanced')
logreg.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred_logreg = logreg.predict(X_test)

# Sınıflandırma raporu gösterme
print("Logistic Regression modeli için sınıflandırma raporu:")
print(classification_report(y_test, y_pred_logreg))

print("Logistic Regression modeli için test seti doğruluğu:", accuracy_score(y_test, y_pred_logreg))


from sklearn.model_selection import cross_val_score


# Cross-validation ile model performansını değerlendirme
scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("Cross-validation doğruluk skorları:", scores)
print("Ortalama cross-validation doğruluk skoru:", scores.mean())

# Tüm sütunlar için pairplot oluşturma
#sns.pairplot(df, hue='Level', diag_kind='kde')
#plt.show()

# Örnek veri setini yükleme
data =  pd.read_csv('converted_dataset.csv')


df = pd.DataFrame(data)

# Level sütununu kategorik olarak değiştirme
df['Level'] = df['Level'].replace({0: 'low', 1: 'medium', 2: 'high'})

# Veriyi görselleştirme için seaborn kütüphanesini kullanma
import seaborn as sns
import matplotlib.pyplot as plt

# Verinizin özellikleri ve Level sütunu
features = ['Age', 'Gender', 'AirPollution', 'Alcoholuse', 'DustAllergy', 
            'OccuPationalHazards', 'GeneticRisk', 'chronicLungDisease', 
            'BalancedDiet', 'Obesity', 'Smoking', 'PassiveSmoker', 'ChestPain', 
            'CoughingofBlood', 'Fatigue', 'WeightLoss', 'ShortnessofBreath', 
            'Wheezing', 'SwallowingDifficulty', 'ClubbingofFingerNails', 
            'FrequentCold', 'DryCough', 'Snoring', 'Level']

# Korelasyon matrisini oluşturun
corr_matrix = data[features].corr()



#DATA VISUALİZATION

# Heatmap çizin
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Level ve Diğer Özellikler Arasındaki Korelasyon')
plt.show()

# Level dağılımını gösteren countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='Level', data=df)
plt.title('Level Sınıf Dağılımı')
plt.show()

# Age dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Yaş Dağılımı')
plt.show()

# Gender ve Level arasındaki ilişkiyi gösteren bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Level', data=df)
plt.title('Cinsiyet ve Kanser Riski Seviyesi Arasındaki İlişki')
plt.show()

# AirPollution ve Level arasındaki ilişkiyi gösteren bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='AirPollution', y='Level', data=df)
plt.title('Hava Kirliliği ve Kanser Riski Seviyesi Arasındaki İlişki')
plt.show()

# Sınıflandırma raporu gösterme
print("Logistic Regression modeli için sınıflandırma raporu:")
print(classification_report(y_test, y_pred_logreg))

print("Logistic Regression modeli için test seti doğruluğu:", accuracy_score(y_test, y_pred_logreg))

# Cross-validation ile model performansını değerlendirme
scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print("Cross-validation doğruluk skorları:", scores)
print("Ortalama cross-validation doğruluk skoru:", scores.mean())

# Karışıklık matrisini oluşturma ve görselleştirme
cm = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Karışıklık Matrisi')
plt.show()

def predict_cancer_risk(features):
    features_scaled = scaler.transform([features])
    prediction = logreg.predict(features_scaled)
    return prediction[0]

# Kullanıcıdan input alarak tahmin yapma
#if __name__ == "__main__":
#    user_input = [int(input(f"{col}: ")) for col in X.columns]
#    risk_level = predict_cancer_risk(user_input)
#    risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
#    print(f"Tahmin edilen kanser riski: {risk_map[risk_level]}")
