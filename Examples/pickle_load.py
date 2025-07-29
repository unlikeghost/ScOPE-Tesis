import pickle
from scope.model import ScOPE

# Cargar pickle
filename: str = "test_optimization_results_20250729_120413"
pickle_file = f"results/{filename}.pkl"

with open(pickle_file, 'rb') as f:
    results = pickle.load(f)


# Mostrar resultados
print("=== RESULTADOS CARGADOS ===")
print(f"Best score: {results['best_value']:.4f}")
print("\nBest parameters:")

for param, value in results['best_params'].items():
    if param == 'string_separator':
        value = repr(value)
    print(f"  {param}: {value}")


# Ver trials
df = results['trials_dataframe']
print(f"\nTotal trials: {len(df)}")
print("\nTop 5 trials:")
print(df.nlargest(5, 'value')[['number', 'value']].to_string())


model_params = {}
for param, value in results['best_params'].items():
    
    if param.startswith("pd_"):
        param = param.replace("pd_", "")
        
    elif param.startswith("ot_"):
        param = param.replace("ot_", "")
    
    model_params[param] = value

# Cargar modelo
model = ScOPE(
    **model_params
)
print(model)