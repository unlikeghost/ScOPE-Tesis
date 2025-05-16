import gzip
from scope.compressors.base_compressor import BaseCompressor


class GZIPCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_module=gzip,
            compressor_name="gzip"
        )


if __name__ == '__main__':
    compressor = GZIPCompressor()
    print(compressor)
    print(compressor(sequence='Hola'))
    print(compressor.get_compressor)
