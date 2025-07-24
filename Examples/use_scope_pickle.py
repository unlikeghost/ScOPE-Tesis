import pickle

# Cargar pickle
pickle_file = "/home/unlikeghost/Dev/Personal/ScOPE/aaa/clintox_experiment_v1_results_20250724_163719.pkl"

with open(pickle_file, 'rb') as f:
    results = pickle.load(f)

# Mostrar resultados
print("=== RESULTADOS CARGADOS ===")
print(f"Best score: {results['best_value']:.4f}")
print("\nBest parameters:")
for param, value in results['best_params'].items():
    if param == 'str_separator':
        value = repr(value)
    print(f"  {param}: {value}")

# Ver trials
df = results['trials_dataframe']
print(f"\nTotal trials: {len(df)}")
print("\nTop 5 trials:")
print(df.nlargest(5, 'value')[['number', 'value']].to_string())