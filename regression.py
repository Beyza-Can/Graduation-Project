import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#  veri setini yükle
df = pd.read_csv('converted_dataset.csv')

# sütunları kontrol et
print("Veri setindeki sütunlar:")
print(df.columns)

# Korelasyon değerlerini kullan ve  percentage sütununu oluştur
correlations = {
    'Age': 0.06, 'Gender': -0.17, 'AirPollution': 0.64, 'Alcoholuse': 0.72,
    'DustAllergy': 0.71, 'OccuPationalHazards': 0.67, 'GeneticRisk': 0.70,
    'chronicLungDisease': 0.61, 'BalancedDiet': -0.70, 'Obesity': 0.83,
    'Smoking': 0.52, 'PassiveSmoker': 0.70, 'ChestPain': 0.64,
    'CoughingofBlood': 0.78, 'Fatigue': 0.62, 'WeightLoss': 0.35,
    'ShortnessofBreath': 0.50, 'Wheezing': 0.25, 'SwallowingDifficulty': 0.25,
    'ClubbingofFingerNails': 0.28, 'FrequentCold': 0.44, 'DryCough': 0.38,
    'Snoring': 0.29
}

# Percentage sütununu oluştur
percentage = sum(df[feature] * correlation for feature, correlation in correlations.items())
df['Percentage'] = percentage

# Sonucu göster
print(df[['Percentage']].head())

# Veri setini kaydet
df.to_csv('dataset_with_percentage.csv', index=False)

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = df.drop(columns=['Percentage','Level']) 
 # Percentage sütununu bağımsız değişkenlerden çıkar
y = df['Percentage'] 
 # Percentage sütunu bağımlı değişken olarak al

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso Regression modeli oluşturma ve eğitme
lasso_model = Lasso(alpha=0.5)  # alpha, regülarizasyon yapan parametre
lasso_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
lasso_y_pred = lasso_model.predict(X_test)
# Regresyon metriklerini yazdır (Lasso Regression)
print("Lasso Regression modeli için regresyon metrikleri:")
print(f"Ortalama Kare Hatası (MSE): {mean_squared_error(y_test, lasso_y_pred):.2f}")
print(f"R-Kare Skoru (R^2): {r2_score(y_test, lasso_y_pred):.2f}")
print(f"Ortalama Mutlak Hata (MAE): {mean_absolute_error(y_test, lasso_y_pred):.2f}")

# Doğruluk yüzdesi hesaplama
def accuracy_percentage(y_true, y_pred):
    differences = abs(y_true - y_pred)
    percentage_differences = 100 - (differences / y_true) * 100
    accuracy = percentage_differences.mean()
    return accuracy

accuracy = accuracy_percentage(y_test, lasso_y_pred)
print(f"Modelin doğruluk yüzdesi: {accuracy:.2f}%")

def predict_percentage(features):
    prediction = lasso_model.predict([features])
    return prediction[0]

# Özellik isimlerini kontrol et
#print("Özellik isimleri:")
#print(X.columns)