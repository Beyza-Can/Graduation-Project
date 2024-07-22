import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def perform_regression(df):
    # Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
    X = df.drop(columns=['Percentage', 'Level'])  # Percentage sütununu bağımsız değişkenlerden çıkar
    y_percentage = df['Percentage']  # Percentage sütunu bağımlı değişken olarak al

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train_percentage, y_test_percentage = train_test_split(X, y_percentage, test_size=0.2, random_state=42)

    # Lasso Regression modeli oluşturma ve eğitme
    lasso_model = Lasso(alpha=0.5)  # alpha, regülarizasyon parametresidir
    lasso_model.fit(X_train, y_train_percentage)

    # Test seti üzerinde tahmin yap
    lasso_y_pred = lasso_model.predict(X_test)

    # Regresyon metriklerini yazdır (Lasso Regression)
    print("Lasso Regression modeli için regresyon metrikleri:")
    print(f"Ortalama Kare Hatası (MSE): {mean_squared_error(y_test_percentage, lasso_y_pred):.2f}")
    print(f"R-Kare Skoru (R^2): {r2_score(y_test_percentage, lasso_y_pred):.2f}")
    print(f"Ortalama Mutlak Hata (MAE): {mean_absolute_error(y_test_percentage, lasso_y_pred):.2f}")

    # Cross-validation ile model performansını değerlendirme
    scores = cross_val_score(lasso_model, X_train, y_train_percentage, cv=5)
    print("Cross-validation doğruluk skorları:", scores)
    print("Ortalama cross-validation doğruluk skoru:", scores.mean())

    return lasso_model

def perform_classification(df):
    # Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
    X = df.drop(columns=['Percentage', 'Level'])  # Level ve Percentage sütunlarını bağımsız değişkenlerden çıkar
    y_classification = df['Level']  # Level sütunu bağımlı değişken olarak al

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification, test_size=0.2, random_state=42)

    # Verileri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression modeli oluşturma ve eğitme
    logreg = LogisticRegression(C=0.001, solver='liblinear', class_weight='balanced')
    logreg.fit(X_train_scaled, y_train_classification)

    # Test seti üzerinde tahmin yapma
    y_pred_logreg = logreg.predict(X_test_scaled)

    # Sınıflandırma raporu gösterme
    print("Logistic Regression modeli için sınıflandırma raporu:")
    print(classification_report(y_test_classification, y_pred_logreg))

    print("Logistic Regression modeli için test seti doğruluğu:", accuracy_score(y_test_classification, y_pred_logreg))

    # Cross-validation ile model performansını değerlendirme
    scores = cross_val_score(logreg, X_train_scaled, y_train_classification, cv=5)
    print("Cross-validation doğruluk skorları:", scores)
    print("Ortalama cross-validation doğruluk skoru:", scores.mean())

    return logreg

df = pd.read_csv('converted_dataset.csv')


X = df.drop(columns=['Level'])  # Level sütununu bağımsız değişkenlerden çıkar
y = df['Level']  # Level sütunu bağımlı değişken olarak al

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression modeli oluşturma ve eğitme
logreg = LogisticRegression(C=0.001, solver='liblinear',class_weight='balanced')
logreg.fit(X_train, y_train)
def predict_cancer_risk(features):
    features_scaled = scaler.transform([features])
    prediction = logreg.predict(features_scaled)
    return prediction[0]

if __name__ == "__main__":
    # Önceden oluşturulmuş ve kaydedilmiş veri setini yükle
    df = pd.read_csv('dataset_with_percentage.csv')

    # Regression işlemlerini gerçekleştir
    lasso_model = perform_regression(df)

    # Classification işlemlerini gerçekleştir
    logreg = perform_classification(df)
