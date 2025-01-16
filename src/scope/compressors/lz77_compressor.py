"""
https://github.com/manassra/LZ77-Compressor
"""
from typing import Union
from bitarray import bitarray
from scope.compressors.base_compressor import BaseCompressor

__all__ = [
    'LZ77Compressor',
]


class LZ77:
    MAX_WINDOW_SIZE: int = 400

    def __init__(self, window_size: int = 20):
        self.window_size: float = min(window_size, self.MAX_WINDOW_SIZE)
        self.lookahead_buffer_size: int = 15

    def find_longest_match(self, data, current_position):
        """
        Finds the longest match to a substring starting at the current_position
        in the lookahead buffer from the history window
        """
        end_of_buffer = min(current_position + self.lookahead_buffer_size, len(data) + 1)

        best_match_distance = -1
        best_match_length = -1

        # Optimization: Only consider substrings of length 2 and greater, and just
        # output any substring of length 1 (8 bits uncompressed is better than 13 bits
        # for the flag, distance, and length)
        for j in range(current_position + 2, end_of_buffer):

            start_index = max(0, current_position - self.window_size)
            substring = data[current_position:j]

            for i in range(start_index, current_position):

                repetitions = len(substring) // (current_position - i)

                last = len(substring) % (current_position - i)

                matched_string = data[i:current_position] * repetitions + data[i:i + last]

                if matched_string == substring and len(substring) > best_match_length:
                    best_match_distance = current_position - i
                    best_match_length = len(substring)

        if best_match_distance > 0 and best_match_length > 0:
            return best_match_distance, best_match_length
        return None

    def compress(self, text: Union[str, bytes], verbose: bool = False, **kwargs) -> bytes:

        i = 0
        output_buffer = bitarray(endian='big')

        if not isinstance(text, str) and not isinstance(text, bytes):
            raise TypeError("Text must be a string or bytes")

        data: bytes = text.encode('utf-8') if isinstance(text, str) else text

        while i < len(data):
            match = self.find_longest_match(data, i)
            if match:
                # Add 1 bit flag, followed by 12 bit for distance, and 4 bit for the length
                # of the match
                (bestMatchDistance, bestMatchLength) = match

                output_buffer.append(True)
                output_buffer.frombytes(bytes([bestMatchDistance >> 4]))
                output_buffer.frombytes(bytes([((bestMatchDistance & 0xf) << 4) | bestMatchLength]))

                if verbose:
                    print("<1, %i, %i>" % (bestMatchDistance, bestMatchLength), end='')

                i += bestMatchLength

            else:
                # No useful match was found. Add 0 bit flag, followed by 8 bit for the character
                output_buffer.append(False)
                output_buffer.frombytes(bytes([data[i]]))

                if verbose:
                    print("<0, %s>" % data[i], end='')

                i += 1

        # fill the buffer with zeros if the number of bits is not a multiple of 8
        output_buffer.fill()

        return output_buffer.tobytes()


class LZ77Compressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_module=LZ77(),
            compressor_name="LZ77"
        )


if __name__ == '__main__':
    compressor = LZ77Compressor()
    print(compressor)
    print(compressor(sequence='Hola!!'))
    print(compressor.get_compressor)
