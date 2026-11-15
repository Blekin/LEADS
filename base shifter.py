#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import random
import zlib
from pathlib import Path
from typing import List, Optional, Tuple


def build_screened():
    input_base = ["A", "C", "G", "T"]
    arrows = [[i, j] for i in input_base for j in input_base]
    qualified = []
    for rule in itertools.permutations(arrows, 4):
        symbol1 = [pair[0] for pair in rule]
        symbol2 = [pair[1] for pair in rule]
        if len(set(symbol1)) == 4 and len(set(symbol2)) == 4:
            qualified.append(rule)

    screened = []
    seen = []
    for rule in qualified:
        mapping = {pair[0]: pair[1] for pair in rule}
        if mapping not in seen:
            seen.append(mapping)
            screened.append(rule)
    return screened


SCREENED = build_screened()


class FiniteSigma:
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
        self.failed = False

    @staticmethod
    def check(seq: str) -> bool:
        return not any(motif in seq for motif in ("AAAA", "CCCC", "GGGG", "TTTT"))

    @staticmethod
    def gc_ok(seq: str, gc_min: float, gc_max: float) -> bool:
        if not seq:
            return False
        if not (0.0 <= gc_min <= gc_max <= 1.0):
            raise ValueError("Invalid GC range")
        gc_count = seq.count("G") + seq.count("C")
        min_gc_count = math.ceil(gc_min * len(seq))
        max_gc_count = math.floor(gc_max * len(seq))
        return min_gc_count <= gc_count <= max_gc_count

    def generator(self, seed: int):
        rng = random.Random(seed)
        while True:
            yield rng.randint(0, 23)

    def transition(self):
        self.s1 = self.screened[next(self._gene)]
        self.s2 = self.screened[next(self._gene)]
        self.s3 = self.screened[next(self._gene)]
        self.s4 = self.screened[next(self._gene)]

    def calculate(self, pre_base: str, cur_base: str) -> str:
        tables = {"A": self.s1, "C": self.s2, "G": self.s3, "T": self.s4}
        if pre_base not in tables:
            raise ValueError(f"Invalid pre_base: {pre_base}")
        for src, dst in tables[pre_base]:
            if cur_base == src:
                return dst
        raise ValueError(f"Unexpected base mapping: {cur_base} with pre_base {pre_base}")

    def decalculate(self, next_base: str, pre_base: str) -> str:
        tables = {"A": self.s1, "C": self.s2, "G": self.s3, "T": self.s4}
        if pre_base not in tables:
            raise ValueError(f"Invalid pre_base: {pre_base}")
        for src, dst in tables[pre_base]:
            if next_base == dst:
                return src
        raise ValueError(f"Unexpected inverse mapping: {next_base} with pre_base {pre_base}")

    def record(self, inp_seq: str, seed: int):
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

        first_record = self.calculate("A", inp_seq[0])
        out_seq.append(first_record)
        if not self.check("".join(out_seq)):
            self.failed = True
            self.recorded_seq = ""
            self.tran_point = []
            return

        for i in range(1, len(inp_seq)):
            pre_base = out_seq[-1]
            record_base = self.calculate(pre_base, inp_seq[i])
            out_seq.append(record_base)

            if self.check("".join(out_seq)):
                continue

            out_seq.pop()
            self.transition()
            record_base = self.calculate(pre_base, inp_seq[i])
            out_seq.append(record_base)

            if not self.check("".join(out_seq)):
                self.failed = True
                self.recorded_seq = ""
                self.tran_point = []
                return

            transition_point.append(i)

        self.recorded_seq = "".join(out_seq)
        self.tran_point = transition_point

    def recover(self, rec_seq: str, tran_point: List[int], seed: int) -> str:
        if not rec_seq:
            return ""

        tran_point = list(tran_point)
        if not tran_point:
            self.s1 = self.screened[0]
            self.s2 = self.screened[0]
            self.s3 = self.screened[0]
            self.s4 = self.screened[0]
            decoded = []
            for idx, base in enumerate(rec_seq):
                pre = "A" if idx == 0 else rec_seq[idx - 1]
                decoded.append(self.decalculate(base, pre))
            return "".join(decoded)

        gene = self.generator(seed)
        all_states = [next(gene) for _ in range(4 * len(tran_point))]

        first_part_encoded = rec_seq[:tran_point[0]]
        self.s1 = self.screened[0]
        self.s2 = self.screened[0]
        self.s3 = self.screened[0]
        self.s4 = self.screened[0]

        first_part_decoded = []
        for idx, base in enumerate(first_part_encoded):
            pre = "A" if idx == 0 else first_part_encoded[idx - 1]
            first_part_decoded.append(self.decalculate(base, pre))

        remain_seq = []
        rec_tail = rec_seq
        while tran_point:
            self.s1 = self.screened[all_states[-4]]
            self.s2 = self.screened[all_states[-3]]
            self.s3 = self.screened[all_states[-2]]
            self.s4 = self.screened[all_states[-1]]
            all_states = all_states[:-4]

            cut_index = tran_point[-1]
            while len(rec_tail) > cut_index:
                pre_base = "A" if len(rec_tail) == 1 else rec_tail[-2]
                cur_base = self.decalculate(rec_tail[-1], pre_base)
                remain_seq.append(cur_base)
                rec_tail = rec_tail[:-1]
            tran_point = tran_point[:-1]

        return "".join(first_part_decoded + remain_seq[::-1])


def gamma_encode(n: int) -> List[str]:
    if n <= 0:
        raise ValueError("gamma_encode expects n >= 1")
    binary = bin(n)[2:]
    length = len(binary)
    return ["0"] * (length - 1) + ["1"] + list(binary[1:])


def gamma_decode(bits: List[str], offset: int) -> Tuple[int, int]:
    zeros = 0
    while offset + zeros < len(bits) and bits[offset + zeros] == "0":
        zeros += 1
    if offset + zeros >= len(bits):
        raise ValueError("gamma_decode: truncated code")
    offset += zeros
    if bits[offset] != "1":
        raise ValueError("gamma_decode: malformed code")
    offset += 1
    payload_bits = ["1"]
    for _ in range(zeros):
        if offset >= len(bits):
            raise ValueError("gamma_decode: truncated payload")
        payload_bits.append(bits[offset])
        offset += 1
    return int("".join(payload_bits), 2), offset


def encode_chunk_metadata_gamma(
    seed: int,
    tran_points: List[int],
    num_seeds: int,
) -> List[str]:
    if not (0 <= seed < num_seeds):
        raise ValueError("seed out of range")

    bits: List[str] = []
    bits.extend(gamma_encode(seed + 1))
    bits.extend(gamma_encode(len(tran_points) + 1))

    if tran_points:
        points = sorted(tran_points)
        bits.extend(gamma_encode(points[0] + 1))
        previous = points[0]
        for point in points[1:]:
            gap = point - previous
            if gap <= 0:
                raise ValueError("transition points must be strictly increasing")
            bits.extend(gamma_encode(gap))
            previous = point
    return bits


def decode_chunk_metadata_gamma(
    bits: List[str],
    offset: int,
) -> Tuple[int, List[int], int]:
    seed_value, offset = gamma_decode(bits, offset)
    seed = seed_value - 1
    count_value, offset = gamma_decode(bits, offset)
    count = count_value - 1

    transition_points: List[int] = []
    if count > 0:
        first_value, offset = gamma_decode(bits, offset)
        first_point = first_value - 1
        transition_points.append(first_point)
        previous = first_point
        for _ in range(1, count):
            gap, offset = gamma_decode(bits, offset)
            previous += gap
            transition_points.append(previous)
    return seed, transition_points, offset


def encode_all_metadata_gamma_with_tail(
    chunk_info: List[Tuple[int, List[int]]],
    num_seeds: int,
    tail_len: int,
) -> List[str]:
    bits: List[str] = []
    bits.extend(gamma_encode(tail_len + 1))
    bits.extend(gamma_encode(len(chunk_info) + 1))
    for seed, transition_points in chunk_info:
        bits.extend(encode_chunk_metadata_gamma(seed, transition_points, num_seeds))
    return bits


def decode_all_metadata_gamma_with_tail(
    bits: List[str],
) -> Tuple[int, int, List[Tuple[int, List[int]]]]:
    offset = 0
    tail_value, offset = gamma_decode(bits, offset)
    tail_len = tail_value - 1
    chunk_value, offset = gamma_decode(bits, offset)
    num_chunks = chunk_value - 1

    chunk_info: List[Tuple[int, List[int]]] = []
    for _ in range(num_chunks):
        seed, transition_points, offset = decode_chunk_metadata_gamma(bits, offset)
        chunk_info.append((seed, transition_points))
    return tail_len, num_chunks, chunk_info


def bits_to_bytes(bits: List[str]) -> bytes:
    output = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            chunk += ["0"] * (8 - len(chunk))
        output.append(int("".join(chunk), 2))
    return bytes(output)


def bytes_to_bits(data: bytes) -> List[str]:
    bits: List[str] = []
    for value in data:
        bits.extend(f"{value:08b}")
    return bits


def bits_to_dna(bits: List[str]) -> str:
    mapping = {"00": "A", "01": "C", "10": "G", "11": "T"}
    if len(bits) % 2:
        bits = bits + ["0"]
    return "".join(mapping["".join(bits[i:i + 2])] for i in range(0, len(bits), 2))


def dna_to_bits(dna: str) -> List[str]:
    mapping = {"A": "00", "C": "01", "G": "10", "T": "11"}
    bits: List[str] = []
    for base in dna:
        if base not in mapping:
            raise ValueError(f"Invalid DNA base: {base}")
        bits.extend(mapping[base])
    return bits


def bytes_to_dna(data: bytes) -> str:
    return bits_to_dna(bytes_to_bits(data))


def dna_to_bytes(dna: str) -> bytes:
    bits = dna_to_bits(dna)
    if len(bits) % 8:
        raise ValueError("dna_to_bytes: bit length not multiple of 8")
    return bits_to_bytes(bits)


def transform_dna_sequence(
    dna_seq: str,
    num_seeds: int = 1024,
    chunk_len: int = 1024,
    check_some_recover: bool = False,
    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,
) -> Tuple[str, str]:
    if not dna_seq:
        raise ValueError("Input DNA sequence is empty")
    if set(dna_seq) - set("ACGT"):
        raise ValueError("Input DNA sequence must contain only A, C, G, and T")
    if chunk_len <= 0 or num_seeds <= 0:
        raise ValueError("chunk_len and num_seeds must be positive")
    if (gc_min is None) != (gc_max is None):
        raise ValueError("gc_min and gc_max must be provided together")
    if gc_min is not None and not (0.0 <= gc_min <= gc_max <= 1.0):
        raise ValueError("Invalid GC range")

    total_len = len(dna_seq)
    num_chunks = total_len // chunk_len
    effective_len = num_chunks * chunk_len
    tail_len = total_len - effective_len
    head_seq = dna_seq[:effective_len]
    tail_seq = dna_seq[effective_len:]

    chunk_info: List[Tuple[int, List[int]]] = []
    recorded_chunks: List[str] = []

    for chunk_index in range(num_chunks):
        seq = head_seq[chunk_index * chunk_len:(chunk_index + 1) * chunk_len]
        best_seed = None
        best_transition_points: List[int] = []
        best_recorded = ""
        best_cost = None

        for seed in range(num_seeds):
            finite_sigma = FiniteSigma()
            finite_sigma.record(seq, seed)
            if finite_sigma.failed:
                continue
            if gc_min is not None and not FiniteSigma.gc_ok(
                finite_sigma.recorded_seq, gc_min, gc_max
            ):
                continue

            cost = len(finite_sigma.tran_point)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_seed = seed
                best_transition_points = finite_sigma.tran_point
                best_recorded = finite_sigma.recorded_seq
                if best_cost == 0:
                    break

        if best_seed is None:
            range_text = ""
            if gc_min is not None:
                range_text = f" with GC range [{gc_min}, {gc_max}]"
            raise RuntimeError(
                f"No valid seed found for chunk {chunk_index}{range_text}"
            )

        chunk_info.append((best_seed, best_transition_points))
        recorded_chunks.append(best_recorded)

        if check_some_recover and chunk_index < 5:
            finite_sigma = FiniteSigma()
            finite_sigma.record(seq, best_seed)
            if finite_sigma.failed:
                raise RuntimeError(f"Recovery check setup failed for chunk {chunk_index}")
            recovered = finite_sigma.recover(
                finite_sigma.recorded_seq,
                best_transition_points,
                best_seed,
            )
            if recovered != seq:
                raise RuntimeError(f"Recovery mismatch for chunk {chunk_index}")

    recorded_dna = "".join(recorded_chunks) + tail_seq
    metadata_bits = encode_all_metadata_gamma_with_tail(
        chunk_info,
        num_seeds,
        tail_len,
    )
    metadata_bytes = zlib.compress(bits_to_bytes(metadata_bits))
    metadata_dna = bytes_to_dna(metadata_bytes)

    print("===== Transform Summary =====")
    print(f"Input length (nt)     = {total_len}")
    print(f"Effective length (nt) = {effective_len}")
    print(f"Tail length (nt)      = {tail_len}")
    print(f"Chunks                = {num_chunks}")
    print(f"Recorded DNA length   = {len(recorded_dna)}")
    print(f"Metadata DNA length   = {len(metadata_dna)}")
    if gc_min is not None:
        print(f"GC constraint         = [{gc_min:.3f}, {gc_max:.3f}]")
    if effective_len:
        print(f"Metadata redundancy   = {len(metadata_dna) / effective_len:.6f}")

    return recorded_dna, metadata_dna


def recover_dna_sequence(
    recorded_dna: str,
    metadata_dna: str,
    chunk_len: int = 1024,
) -> str:
    metadata_bytes = zlib.decompress(dna_to_bytes(metadata_dna))
    metadata_bits = bytes_to_bits(metadata_bytes)
    tail_len, num_chunks, chunk_info = decode_all_metadata_gamma_with_tail(metadata_bits)

    effective_len = num_chunks * chunk_len
    total_len = effective_len + tail_len
    if len(recorded_dna) != total_len:
        raise ValueError(
            f"recorded_dna length mismatch: got {len(recorded_dna)}, expected {total_len}"
        )

    head_recorded = recorded_dna[:effective_len]
    tail_seq = recorded_dna[effective_len:]
    recovered_chunks: List[str] = []

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_len
        end = start + chunk_len
        rec_chunk = head_recorded[start:end]
        seed, transition_points = chunk_info[chunk_index]
        finite_sigma = FiniteSigma()
        recovered_chunks.append(
            finite_sigma.recover(rec_chunk, transition_points, seed)
        )

    return "".join(recovered_chunks) + tail_seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--recorded-output", type=Path, default=Path("recorded_dna.txt"))
    parser.add_argument("--metadata-output", type=Path, default=Path("metadata_dna.txt"))
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--num-seeds", type=int, default=16384)
    parser.add_argument("--gc-min", type=float, default=0.4)
    parser.add_argument("--gc-max", type=float, default=0.6)
    parser.add_argument("--check-recover", action="store_true")
    args = parser.parse_args()

    dna_input = "".join(args.input.read_text().split())
    recorded_dna, metadata_dna = transform_dna_sequence(
        dna_input,
        num_seeds=args.num_seeds,
        chunk_len=args.chunk_len,
        check_some_recover=args.check_recover,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
    )

    args.recorded_output.write_text(recorded_dna)
    args.metadata_output.write_text(metadata_dna)

    recovered = recover_dna_sequence(
        recorded_dna,
        metadata_dna,
        chunk_len=args.chunk_len,
    )

    print("===== Recovery Summary =====")
    print(f"Recovered length      = {len(recovered)}")
    print(f"Identical              = {recovered == dna_input}")
    print(f"Recorded output        = {args.recorded_output}")
    print(f"Metadata output        = {args.metadata_output}")


if __name__ == "__main__":
    main()
