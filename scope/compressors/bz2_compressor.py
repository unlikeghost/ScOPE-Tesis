import bz2
from scope.compressors.base_compressor import BaseCompressor


class BZ2Compressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_module=bz2,
            compressor_name="bz2"
        )


if __name__ == '__main__':
    compressor = BZ2Compressor()
    print(compressor)
    print(compressor(sequence='Hola'))
