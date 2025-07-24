from scope import ScOPE

model =  ScOPE(
    compressor_name="gzip",
    compression_distance_function='ncd',
    use_best_sigma=True,
    string_separator=' ',
    get_softmax=True,
    
    model_type="ot",
    use_matching_method=True,
    matching_method_name="dice"
)

sample= ["hola"]

kw_samples = {
    "sample_1": ["hola", "holi"],
    "sample_2": ["adios", "adiosito"]
}
    
print(list(model(sample, kw_samples)))


for distance in ["cosine", "euclidean", "manhattan", "chebyshev", "canberra", "minkowski", "braycurtis", "hamming", "correlation", "dot_product"]:
    model = ScOPE(
        compressor_name="gzip",
        compression_distance_function='ncd',
        use_best_sigma=True,
        string_separator=' ',
        get_softmax=True,
        
        model_type="pd",
        distance_metric=distance,
        use_prototypes=True
    )

    print(list(model(sample, kw_samples)))