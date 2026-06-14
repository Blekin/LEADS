import random
import math
import zlib
import itertools
from typing import List, Tuple


# Precompute FSM states (24 4-to-4 permutations)

def build_screened():
    input_base = ['A', 'C', 'G', 'T']
    arrow = [[i, j] for i in input_base for j in input_base]  # 16 (in, out) pairs

    qualified = []
    for rule in itertools.permutations(arrow, 4):
        symbol1 = [j[0] for j in rule]
        symbol2 = [j[1] for j in rule]
        if len(set(symbol1)) == 4 and len(set(symbol2)) == 4:
            qualified.append(rule)

    l_dict = []
    screened = []
    for rule in qualified:
        mapping = {}
        for p in rule:
            mapping[p[0]] = p[1]
        if mapping not in l_dict:
            l_dict.append(mapping)
            screened.append(rule)

    return screened


SCREENED = build_screened()


# NSC / FiniteSigma

class FiniteSigma:
    """
    Basic unit of NSC:
    - For each seed and a fixed-length segment:
        - At most one FSM transition is allowed per position.
        - If after a single transition the 4-homopolymer constraint is still violated,
          the seed is considered invalid for this segment.
    """
    SCREENED = SCREENED

    def __init__(self):
        self.screened = FiniteSigma.SCREENED
        self.s1 = self.screened[0]
        self.s2 = self.screened[0]
        self.s3 = self.screened[0]
        self.s4 = self.screened[0]
        self.recorded_seq = ""
        self.tran_point: List[int] = []
        self._gene = None
        self.failed: bool = False

    @staticmethod
    def check(seq: str) -> bool:
        """Sequence constraint: no 4-homopolymer runs."""
        constraints = ('AAAA', 'CCCC', 'GGGG', 'TTTT')
        return not any(c in seq for c in constraints)

    def generator(self, seed: int):
        rng = random.Random(seed)
        while True:
            yield rng.randint(0, 23)

    def transition(self):
        """Pick 4 indices from the random stream and update the permutations used by s1..s4."""
        s1_idx = next(self._gene)
        s2_idx = next(self._gene)
        s3_idx = next(self._gene)
        s4_idx = next(self._gene)
        self.s1 = self.screened[s1_idx]
        self.s2 = self.screened[s2_idx]
        self.s3 = self.screened[s3_idx]
        self.s4 = self.screened[s4_idx]

    def calculate(self, pre_base: str, cur_base: str) -> str:
        """Forward operation: original base -> recorded base."""
        if pre_base == 'A':
            table = self.s1
        elif pre_base == 'C':
            table = self.s2
        elif pre_base == 'G':
            table = self.s3
        elif pre_base == 'T':
            table = self.s4
        else:
            raise ValueError(f"Invalid pre_base: {pre_base}")
        for src, dst in table:
            if cur_base == src:
                return dst
        raise ValueError(f"Unexpected base mapping: {cur_base} with pre_base {pre_base}")

    def decalculate(self, nex_base: str, pre_base: str) -> str:
        """Reverse operation: recorded base -> original base."""
        if pre_base == 'A':
            table = self.s1
        elif pre_base == 'C':
            table = self.s2
        elif pre_base == 'G':
            table = self.s3
        elif pre_base == 'T':
            table = self.s4
        else:
            raise ValueError(f"Invalid pre_base: {pre_base}")
        for src, dst in table:
            if nex_base == dst:
                return src
        raise ValueError(f"Unexpected inverse mapping: {nex_base} with pre_base {pre_base}")

    def record(self, inp_seq: str, seed: int):
        """
        Transform a segment using the given seed.
        Rules:
        - At most one transition() call is allowed per position.
        - If the constraint is still violated after a transition, mark self.failed = True.
        """
        self.failed = False
        if not inp_seq:
            self.recorded_seq = ""
            self.tran_point = []
            return

        self._gene = self.generator(seed)
        self.s1 = self.screened[0]
        self.s2 = self.screened[0]
        self.s3 = self.screened[0]
        self.s4 = self.screened[0]

        out_seq: List[str] = []
        transition_point: List[int] = []

        # First position: pre_base is fixed to 'A'
        first_record = self.calculate('A', inp_seq[0])
        out_seq.append(first_record)
        if not self.check(''.join(out_seq)):
            self.failed = True
            self.recorded_seq = ""
            self.tran_point = []
            return

        # Subsequent positions
        for i in range(1, len(inp_seq)):
            pre_base = out_seq[-1]

            # First attempt: do not switch FSM
            record_base = self.calculate(pre_base, inp_seq[i])
            out_seq.append(record_base)

            if self.check(''.join(out_seq)):
                continue  # success

            # Otherwise, roll back the current base, perform one FSM transition, and retry
            out_seq.pop()
            self.transition()

            record_base = self.calculate(pre_base, inp_seq[i])
            out_seq.append(record_base)

            if not self.check(''.join(out_seq)):
                # Still violates constraint after one transition: this seed is invalid
                self.failed = True
                self.recorded_seq = ""
                self.tran_point = []
                return

            # Success: record a transition point (at most one per position)
            transition_point.append(i)

        self.recorded_seq = ''.join(out_seq)
        self.tran_point = transition_point

    def recover(self, rec_seq: str, tran_point: List[int], seed: int) -> str:
        """Recover the original sequence from recorded_seq + tran_point + seed."""
        if not rec_seq:
            return ""

        tran_point = list(tran_point)

        # No transition points: the whole segment uses the initial FSM; direct inverse is possible
        if not tran_point:
            self.s1 = self.screened[0]
            self.s2 = self.screened[0]
            self.s3 = self.screened[0]
            self.s4 = self.screened[0]
            decoded = []
            for idx, base in enumerate(rec_seq):
                pre = 'A' if idx == 0 else rec_seq[idx - 1]
                decoded.append(self.decalculate(base, pre))
            return ''.join(decoded)

        # Transition points exist: replay the random sequence to recover the FSM state at each transition
        len_trans = len(tran_point)
        gene = self.generator(seed)
        all_states = [next(gene) for _ in range(4 * len_trans)]

        # First segment (before the first transition point)
        first_part_encoded = rec_seq[:tran_point[0]]
        self.s1 = self.screened[0]
        self.s2 = self.screened[0]
        self.s3 = self.screened[0]
        self.s4 = self.screened[0]

        first_part_decoded = []
        for idx, base in enumerate(first_part_encoded):
            pre = 'A' if idx == 0 else first_part_encoded[idx - 1]
            first_part_decoded.append(self.decalculate(base, pre))

        # Decode the remaining segments backwards
        remain_seq = []
        rec_tail = rec_seq

        while tran_point:
            # Restore the FSM at the time of this transition
            self.s1 = self.screened[all_states[-4]]
            self.s2 = self.screened[all_states[-3]]
            self.s3 = self.screened[all_states[-2]]
            self.s4 = self.screened[all_states[-1]]
            all_states = all_states[:-4]

            cut_index = tran_point[-1]

            while len(rec_tail) > cut_index:
                if len(rec_tail) == 1:
                    pre_base = 'A'
                else:
                    pre_base = rec_tail[-2]
                cur_base = self.decalculate(rec_tail[-1], pre_base)
                remain_seq.append(cur_base)
                rec_tail = rec_tail[:-1]

            tran_point = tran_point[:-1]

        recovered_seq = first_part_decoded + remain_seq[::-1]
        return ''.join(recovered_seq)


# Elias-gamma encoding/decoding

def gamma_encode(n: int) -> List[str]:
    """Elias-gamma encoding: n >= 1 -> list of bits ['0','1',...]."""
    if n <= 0:
        raise ValueError("gamma_encode expects n >= 1")
    b = bin(n)[2:]  # binary without '0b'
    L = len(b)
    return ['0'] * (L - 1) + ['1'] + list(b[1:])


def gamma_decode(bits: List[str], offset: int) -> Tuple[int, int]:
    """Elias-gamma decoding: starting from bits[offset], returns (n, new_offset)."""
    L = 0
    while offset + L < len(bits) and bits[offset + L] == '0':
        L += 1
    if offset + L >= len(bits):
        raise ValueError("gamma_decode: truncated code")
    offset += L
    if bits[offset] != '1':
        raise ValueError("gamma_decode: malformed code")
    offset += 1
    payload_bits = ['1']
    for _ in range(L):
        if offset >= len(bits):
            raise ValueError("gamma_decode: truncated payload")
        payload_bits.append(bits[offset])
        offset += 1
    n = int(''.join(payload_bits), 2)
    return n, offset


# Metadata encoding/decoding

def encode_chunk_metadata_gamma(seed: int, tran_points: List[int],
                                L: int, num_seeds: int) -> List[str]:
    """
    Elias-gamma encode the metadata for a single chunk:
    - seed in [0, num_seeds-1]   -> gamma(seed+1)
    - K = len(tran_points)       -> gamma(K+1)
    - If K > 0:
        p1                       -> gamma(p1+1)
        For j>=2: gap_j = p_j - p_{j-1} (>0) -> gamma(gap_j)
    """
    if not (0 <= seed < num_seeds):
        raise ValueError("seed out of range")

    bits: List[str] = []
    bits.extend(gamma_encode(seed + 1))

    K = len(tran_points)
    bits.extend(gamma_encode(K + 1))

    if K > 0:
        tps = sorted(tran_points)
        # p1
        bits.extend(gamma_encode(tps[0] + 1))
        # gaps
        last = tps[0]
        for j in range(1, K):
            gap = tps[j] - last
            if gap <= 0:
                raise ValueError("transition points must be strictly increasing")
            bits.extend(gamma_encode(gap))
            last = tps[j]

    return bits


def decode_chunk_metadata_gamma(bits: List[str], offset: int) -> Tuple[int, List[int], int]:
    """
    Read (seed, tran_points) of one chunk from the bit stream, returns (seed, tran_points, new_offset).
    """
    seed1, offset = gamma_decode(bits, offset)
    seed = seed1 - 1

    K1, offset = gamma_decode(bits, offset)
    K = K1 - 1

    tran_points: List[int] = []
    if K > 0:
        p1_1, offset = gamma_decode(bits, offset)
        p1 = p1_1 - 1
        tran_points.append(p1)
        last = p1
        for _ in range(1, K):
            gap, offset = gamma_decode(bits, offset)
            last = last + gap
            tran_points.append(last)

    return seed, tran_points, offset


def encode_all_metadata_gamma_with_tail(chunk_info: List[Tuple[int, List[int]]],
                                        L: int,
                                        num_seeds: int,
                                        tail_len: int) -> List[str]:
    """
    Concatenate all chunk metadata into a single bit stream:
    - First encode tail_len: gamma(tail_len + 1)
    - Then encode the number of chunks: gamma(num_chunks + 1)
    - Then encode (seed, tran_points) for each chunk sequentially
    """
    bits: List[str] = []
    bits.extend(gamma_encode(tail_len + 1))
    bits.extend(gamma_encode(len(chunk_info) + 1))
    for seed, tp in chunk_info:
        bits.extend(encode_chunk_metadata_gamma(seed, tp, L, num_seeds))
    return bits


def decode_all_metadata_gamma_with_tail(bits: List[str]) -> Tuple[int, int, List[Tuple[int, List[int]]]]:
    """
    Decode from the bit stream:
    - tail_len
    - num_chunks
    - chunk_info list
    """
    offset = 0
    tail1, offset = gamma_decode(bits, offset)
    tail_len = tail1 - 1

    num_chunks1, offset = gamma_decode(bits, offset)
    num_chunks = num_chunks1 - 1

    chunk_info: List[Tuple[int, List[int]]] = []
    for _ in range(num_chunks):
        seed, tp, offset = decode_chunk_metadata_gamma(bits, offset)
        chunk_info.append((seed, tp))

    return tail_len, num_chunks, chunk_info


# bit / byte / DNA conversion

def bits_to_bytes(bits: List[str]) -> bytes:
    """Bit list -> bytes, zero-padded to an integer number of bytes at the end."""
    byte_stream = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            chunk = chunk + ['0'] * (8 - len(chunk))
        byte_value = int(''.join(chunk), 2)
        byte_stream.append(byte_value)
    return bytes(byte_stream)


def bytes_to_bits(data: bytes) -> List[str]:
    bits: List[str] = []
    for b in data:
        bits.extend(f"{b:08b}")
    return bits


def bits_to_dna(bits: List[str]) -> str:
    """
    Map bits to DNA:
    - 00 -> A
    - 01 -> C
    - 10 -> G
    - 11 -> T
    The end is zero-padded to an even number of bits.
    """
    mapping = {
        '00': 'A',
        '01': 'C',
        '10': 'G',
        '11': 'T',
    }
    if len(bits) % 2 != 0:
        bits = bits + ['0']  # pad with a single '0' to make length even
    dna = []
    for i in range(0, len(bits), 2):
        pair = ''.join(bits[i:i+2])
        dna.append(mapping[pair])
    return ''.join(dna)


def dna_to_bits(dna: str) -> List[str]:
    """
    DNA -> bit stream (for metadata):
    - A -> 00
    - C -> 01
    - G -> 10
    - T -> 11
    """
    mapping = {
        'A': '00',
        'C': '01',
        'G': '10',
        'T': '11',
    }
    bits: List[str] = []
    for base in dna:
        bits.extend(list(mapping[base]))
    return bits


def bytes_to_dna(data: bytes) -> str:
    """bytes -> DNA (each byte -> 8 bits -> 4 bases)."""
    bits = bytes_to_bits(data)
    return bits_to_dna(bits)


def dna_to_bytes(dna: str) -> bytes:
    """DNA -> bytes, assuming the bit length is a multiple of 8."""
    bits = dna_to_bits(dna)
    if len(bits) % 8 != 0:
        raise ValueError("dna_to_bytes: bit length not multiple of 8")
    byte_stream = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        byte_value = int(''.join(chunk), 2)
        byte_stream.append(byte_value)
    return bytes(byte_stream)


# Encoding: original DNA -> recorded_dna + metadata_dna

def transform_dna_sequence(
    dna_seq: str,
    NUM_SEEDS: int = 1024,
    CHUNK_LEN: int = 1024,
    check_some_recover: bool = False,
):
    """
    Input:
        dna_seq: original DNA sequence (containing only A/C/G/T)
        NUM_SEEDS: number of seeds to try per chunk (0..NUM_SEEDS-1)
        CHUNK_LEN: length of each chunk
        check_some_recover: if True, perform a recovery check on the first few chunks

    Output:
        recorded_dna: concatenation of all recorded chunks plus the unmodified tail
        metadata_dna: DNA representation of the compressed metadata
    """
    total_len = len(dna_seq)
    num_chunks = total_len // CHUNK_LEN
    effective_len = num_chunks * CHUNK_LEN
    tail_len = total_len - effective_len

    head_seq = dna_seq[:effective_len]
    tail_seq = dna_seq[effective_len:]  # tail that is not converted

    chunk_info: List[Tuple[int, List[int]]] = []
    recorded_chunks: List[str] = []

    for ci in range(num_chunks):
        seq = head_seq[ci * CHUNK_LEN:(ci + 1) * CHUNK_LEN]

        best_seed = None
        best_tp: List[int] = []
        best_recorded = ""
        best_cost = None

        for seed in range(NUM_SEEDS):
            fs = FiniteSigma()
            fs.record(seq, seed)
            if fs.failed:
                continue

            cost = len(fs.tran_point)
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_seed = seed
                best_tp = fs.tran_point
                best_recorded = fs.recorded_seq
                if best_cost == 0:
                    break  # optimal already found

        if best_seed is None:
            raise RuntimeError(f"No valid seed found for chunk {ci}")

        chunk_info.append((best_seed, best_tp))
        recorded_chunks.append(best_recorded)

        # Optional: verify recovery for the first few chunks
        if check_some_recover and ci < 5:
            fs = FiniteSigma()
            fs.record(seq, best_seed)
            assert not fs.failed
            recovered = fs.recover(fs.recorded_seq, best_tp, best_seed)
            assert recovered == seq, f"Recover mismatch at chunk {ci}"

    # Overall transformed DNA = all recorded chunks + unmodified tail
    recorded_dna = ''.join(recorded_chunks) + tail_seq

    # Encode metadata as bit stream (includes tail_len + chunk_info)
    metadata_bits = encode_all_metadata_gamma_with_tail(
        chunk_info, CHUNK_LEN, NUM_SEEDS, tail_len
    )

    # bits -> bytes -> zlib compression
    metadata_bytes_uncompressed = bits_to_bytes(metadata_bits)
    metadata_bytes_compressed = zlib.compress(metadata_bytes_uncompressed)

    # Map compressed bytes to DNA
    metadata_dna = bytes_to_dna(metadata_bytes_compressed)

    # Print a brief redundancy summary
    compressed_size_bits = len(metadata_bytes_compressed) * 8
    compressed_nt = math.ceil(compressed_size_bits / 2)
    redundancy = compressed_nt / effective_len if effective_len > 0 else 0.0
    print("******Transform Summary******")
    print(f"Input length (nt)     = {total_len}")
    print(f"Effective length (nt) = {effective_len}")
    print(f"Tail length (nt)      = {tail_len}")
    print(f"Chunks                = {num_chunks}")
    print(f"Recorded DNA length   = {len(recorded_dna)}")
    print(f"Metadata DNA length   = {len(metadata_dna)}")
    print(f"Metadata redundancy   ~ {redundancy:.6f} (per nt of payload)")

    return recorded_dna, metadata_dna


# Decoding: recorded_dna + metadata_dna -> original DNA

def recover_dna_sequence(
    recorded_dna: str,
    metadata_dna: str,
    CHUNK_LEN: int = 1024,
) -> str:
    """
    Input:
        recorded_dna: transformed DNA (including tail) output during encoding
        metadata_dna: metadata DNA output during encoding
    Output:
        original_dna: fully recovered original DNA sequence
    """
    # 1. metadata_dna -> bytes -> decompress -> bit stream
    metadata_bytes_compressed = dna_to_bytes(metadata_dna)
    metadata_bytes_uncompressed = zlib.decompress(metadata_bytes_compressed)
    metadata_bits = bytes_to_bits(metadata_bytes_uncompressed)

    # 2. Parse tail_len + chunk_info
    tail_len, num_chunks, chunk_info = decode_all_metadata_gamma_with_tail(metadata_bits)

    effective_len = num_chunks * CHUNK_LEN
    total_len = effective_len + tail_len

    if len(recorded_dna) != total_len:
        raise ValueError(
            f"recorded_dna length mismatch: got {len(recorded_dna)}, expected {total_len}"
        )

    head_rec = recorded_dna[:effective_len]
    tail_seq = recorded_dna[effective_len:]

    # 3. For each chunk, recover using the corresponding seed and transition points
    recovered_chunks: List[str] = []
    for ci in range(num_chunks):
        rec_chunk = head_rec[ci * CHUNK_LEN:(ci + 1) * CHUNK_LEN]
        seed, tp = chunk_info[ci]
        fs = FiniteSigma()
        original_chunk = fs.recover(rec_chunk, tp, seed)
        recovered_chunks.append(original_chunk)

    original_head = ''.join(recovered_chunks)
    original_dna = original_head + tail_seq  # tail is appended unchanged

    return original_dna


# Self-check example (optional)

if __name__ == "__main__":

    with open('your_input.txt','r') as f:
        original = f.read()

    print("Encoding...")
    recorded_dna, metadata_dna = transform_dna_sequence(
        original,
        NUM_SEEDS=16384,   # you can adjust this, e.g., 1024/4096
        CHUNK_LEN=1024,
        check_some_recover=False
    )
    
    with open('recorded_dna.txt', 'w') as fx:
        fx.write(recorded_dna)

    with open('metadata_dna.txt', 'w') as fx:
        fx.write(metadata_dna)

    print("Decoding...")
    recovered = recover_dna_sequence(
        recorded_dna,
        metadata_dna,
        CHUNK_LEN=1024,
    )

    print("Original length :", len(original))
    print("Recovered length:", len(recovered))
    print("Identical?      :", original == recovered)