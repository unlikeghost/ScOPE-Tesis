from scope.compression import METRIC_STRATEGIES, MetricType
from scope.compression.metrics import NCDStrategy
from scope.compression import get_compressor



compressor = get_compressor(
    name="gzip",
)

class Factory:
    def __init__(self, compressor):
        self.compressor = compressor

    def _get_compress_len_(self, text: str) -> int:
        a, b, compressed_len = self.compressor(text)
        return compressed_len

fact = Factory(compressor=compressor)

ncd = NCDStrategy(factory=fact)

text: str = "Hola"

valor = ncd.compute(x1=text, x2=text, concat_str="")

print(valor)