import joblib
import pandas as pd

# Modeli yükle
model = joblib.load('classification_model.pkl')

# Test verisi oluştur (örnek olarak, verisetinizden bir örnek alabilirsiniz)
test_data = pd.DataFrame({
    'Age': [40],
    'Gender': [1],
    'AirPollution': [3],
    'AlcoholUse': [0],
    'DustAllergy': [5],
    'OccupationalHazards': [4],
    'GeneticRisks': [5],
    'ChronicLungDisease': [5],
    'BalancedDiet': [6],
    'Obesity': [10],
    'Smoking': [0],
    'PassiveSmoker': [10],
    'ChestPain': [5],
    'CoughingOfBlood': [0],
    'Fatigue': [5],
    'WeightLoss': [0],
    'ShortnessOfBreath': [8],
    'Wheezing': [7],
    'SwallowingDifficulty': [5],
    'ClubbingOfFingerNails': [2],
    'FrequentCold': [2],
    'DryCough': [6],
    'Snoring': [8]
})

# Model ile tahmin yap
prediction = model.predict(test_data)[0]

# Tahmin sonucunu ekrana yazdır
print(f'Tahmin edilen sınıf: {prediction}')
