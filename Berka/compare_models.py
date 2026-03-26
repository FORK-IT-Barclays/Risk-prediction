import joblib

m1 = joblib.load(r'Output_Artifacts\behavioral_engine.pkl')
m2_dict = joblib.load(r'Output_Artifacts\behavioral_engine_v2.pkl')
m2 = m2_dict['model']

print('=== M1 PARAMS ===')
print(m1.get_params())
print('\n=== M2 PARAMS ===')
print(m2.get_params())

print('\n=== IMPORTANCES ===')
try:
    print('M1 Importances:', m1.feature_importances_)
    print('M2 Importances:', m2.feature_importances_)
except Exception as e:
    print('Error getting importances:', e)
