import numpy as np
from scope.utils.param_search.bayesian import optimize_scope_model, analyze_optimization_results

print("=== SCOPE OPTIMIZATION EXAMPLE ===\n")

np.random.seed(42)

X_validation = [f"example text {i}" for i in range(100)]
y_validation = np.random.choice(['class_0', 'class_1'], size=100).tolist()

kw_samples_validation = []
for _ in range(100):
    kw_sample = {
        'class_0': [f"example 0 {i}" for i in range(3)],
        'class_1': [f"example 1 {i}" for i in range(3)]
    }
    kw_samples_validation.append(kw_sample)

print(f"Validation: {len(X_validation)} samples")
print(f"Classes: {set(y_validation)}")

# Execute optimization
study, optimizer = optimize_scope_model(
    X_validation=X_validation,
    y_validation=y_validation,
    kw_samples_validation=kw_samples_validation,
    n_trials=20,
    timeout=600,
    target_metric='auc_roc'
)

# Analyze results
df_results = analyze_optimization_results(study)

print("\n=== FINAL OPTIMIZED MODEL ===")
print("Optimal configuration found:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

# Create model with optimal configuration for future use
final_optimized_model = optimizer.create_model_from_params(study.best_params)
print("\nOptimized model ready to use!")
print("Usage: optimized_model.__forward__(sample, kw_sample)")

# Save results
# save_optimization_results(study, "scope_optimization_example.pkl")