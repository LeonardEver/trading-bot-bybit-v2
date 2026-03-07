# ml/check_features.py
import joblib
from pathlib import Path

# Resolve caminho absoluto do modelo independente de onde você executar o script
MODEL_PATH = Path(__file__).parent / "model_lgb.pkl"
model = joblib.load(MODEL_PATH)

# Tenta extrair os nomes das features do modelo
if hasattr(model, "booster_"):  # se for LGBMClassifier
    feature_names = model.booster_.feature_name()
elif hasattr(model, "feature_name"):  # se for Booster direto
    feature_names = model.feature_name()
else:
    feature_names = getattr(model, "feature_name_", None)

print("Features usadas no treino:")
print(feature_names)
