from scope import ScOPE

    
model =  ScOPE(
    compressor="gzip",
    name_distance_function='ncd',
    use_best_sigma=True,
    str_separator=' ',
    use_matching_method=True
)

sample= ["hola"]

kw_samples = {
    "sample_1": ["hola"],
    "sample_2": ["adios"]
}
    
print(list(model(sample, kw_samples)))