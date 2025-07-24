from scope.param_search import ScOPEOptimizerBayesian

# Crear optimizador
optimizer = ScOPEOptimizerBayesian(
    study_name="clintox_experiment_v1",
    output_path="./aaa",
    n_trials=1000,
    target_metric='combined',
    cv_folds=5
)


# DATOS CON MÁS MUESTRAS Y VARIEDAD - 12 de cada clase para CV=3
x_validation = [
    # Clase 0 (12 muestras)
    "molecule toxic heavy metal lead", "compound dangerous poison arsenic", 
    "chemical harmful mercury substance", "element toxic cadmium dangerous",
    "poison lethal cyanide compound", "toxic substance benzene harmful",
    "dangerous chemical formaldehyde", "harmful compound asbestos fiber",
    "toxic metal chromium dangerous", "poison substance strychnine lethal",
    "harmful chemical dioxin toxic", "dangerous compound pesticide toxic",
    
    # Clase 1 (12 muestras)  
    "safe molecule water oxygen", "harmless compound sugar glucose",
    "beneficial substance vitamin C", "safe chemical sodium chloride",
    "harmless element calcium safe", "beneficial compound protein amino",
    "safe substance carbohydrate energy", "harmless chemical citric acid",
    "beneficial molecule antioxidant", "safe compound fiber cellulose",
    "harmless substance mineral zinc", "beneficial chemical enzyme natural"
]

y_validation = [0]*12 + [1]*12  # 12 de cada clase

kw_samples_validation = [
    {
        "sample_1": ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"], 
        "sample_2": ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(24)  # 24 elementos
]

# optimizer.parameter_space.model_types = ["ot"]

# Ejecutar optimización
study = optimizer.optimize(x_validation, y_validation, kw_samples_validation)

optimizer.save_complete_analysis(top_n=500)
