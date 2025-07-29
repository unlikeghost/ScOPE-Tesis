from scope.compression import CompressionMatrixFactory

test_samples = {
    'class_0': ['Hola'],
    'class_1': ['Adios']
}
test_sample = 'Hola'

factory = CompressionMatrixFactory(
    compression_metric="ncd", concat_value="", compressor_name="gzip"
)

result = factory(test_sample, test_samples, get_sigma=True)
print(result)

print(factory)