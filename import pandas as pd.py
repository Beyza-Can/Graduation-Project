def validate_data(data):
    """
    Verilerin doğruluğunu kontrol eder.
    Yaşın negatif olmaması, zorunlu alanların dolu olması gibi temel kontrolleri yapar.
    """
    # Yaşın negatif olmaması
    if data['Age'] < 0:
        return False
    # Cinsiyetin 0 (Kadın) veya 1 (Erkek) olması
    if data['Gender'] not in [0, 1]:
        return False
    # Diğer özelliklerin belirli bir aralıkta olması (0-5)
    for key in data:
        if key not in ['Age', 'Gender'] and (data[key] < 0 or data[key] > 5):
            return False
    return True
def test_validate_numerical_data():
    """
    validate_data fonksiyonunu test eder.
    """
    valid_data = {'Age': 45, 'Gender': 1, 'AirPollution': 2, 'Alcoholuse': 1, 
                  'DustAllergy': 0, 'OccuPationalHazards': 1, 'GeneticRisk': 2, 
                  'chronicLungDisease': 0, 'BalancedDiet': 1, 'Obesity': 3, 'Smoking': 1, 
                  'PassiveSmoker': 0, 'ChestPain': 0, 'CoughingofBlood': 0, 'Fatigue': 0, 
                  'WeightLoss': 0, 'ShortnessofBreath': 0, 'Wheezing': 0, 'SwallowingDifficulty': 0,
                    'ClubbingofFingerNails': 0, 'FrequentCold': 0, 'DryCough': 0, 'Snoring': 0}
    
    invalid_data_negative_age = {'Age': -1, 'Gender': 1, 'AirPollution': 2, 'Alcoholuse': 1, 
                                 'DustAllergy': 0, 'OccuPationalHazards': 1, 'GeneticRisk': 2, 
                                 'chronicLungDisease': 0, 'BalancedDiet': 1, 'Obesity': 3, 
                                 'Smoking': 1, 'PassiveSmoker': 0, 'ChestPain': 0, 
                                 'CoughingofBlood': 0, 'Fatigue': 0, 'WeightLoss': 0, 
                                 'ShortnessofBreath': 0, 'Wheezing': 0, 'SwallowingDifficulty': 0,
                                   'ClubbingofFingerNails': 0, 'FrequentCold': 0, 'DryCough': 0, 
                                   'Snoring': 0}
    
    invalid_data_gender = {'Age': 45, 'Gender': 2, 'AirPollution': 2, 'Alcoholuse': 1, 
                           'DustAllergy': 0, 'OccuPationalHazards': 1, 'GeneticRisk': 2, 
                           'chronicLungDisease': 0, 'BalancedDiet': 1, 'Obesity': 3, 
                           'Smoking': 1, 'PassiveSmoker': 0, 'ChestPain': 0, 'CoughingofBlood': 0, 
                           'Fatigue': 0, 'WeightLoss': 0, 'ShortnessofBreath': 0, 'Wheezing': 0, 
                           'SwallowingDifficulty': 0, 'ClubbingofFingerNails': 0, 'FrequentCold': 0,
                             'DryCough': 0, 'Snoring': 0}
    
    assert validate_data(valid_data) == True
    assert validate_data(invalid_data_negative_age) == False
    assert validate_data(invalid_data_gender) == False
