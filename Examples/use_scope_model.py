from scope import ScOPE

print("holi")
model =  ScOPE(
    compressor_name="gzip",
    compression_distance_function='ncd',
    use_best_sigma=True,
    string_separator=' ',
    use_softmax=True,
    
    model_type="ot",
    use_matching_method=True,
    matching_method_name="dice"
)

sample= ["holi"]

kw_samples = {
    "sample_1": ["holi", "hola"],
    "sample_2": ["adios", "adiosito"]
}
    
print(list(model(sample, kw_samples)))


for distance in ["cosine", "euclidean", "manhattan", "chebyshev", "canberra", "minkowski", "braycurtis", "hamming", "correlation", "dot_product"]:
    model = ScOPE(
        compressor_name="gzip",
        compression_distance_function='ncd',
        use_best_sigma=True,
        string_separator=' ',
        
        model_type="pd",
        distance_metric=distance,
        use_prototypes=True
    )

    print(list(model(sample, kw_samples)))
    break


