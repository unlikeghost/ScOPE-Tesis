from zstandard import ZstdCompressor

from scope.compressors.base_compressor import BaseCompressor


class ZStandardCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_module=ZstdCompressor(),
            compressor_name="ZStandard"
        )


if __name__ == '__main__':
    compressor = ZStandardCompressor()
    print(compressor)
    print(compressor(sequence='Hola'))
    print(compressor.get_compressor)
