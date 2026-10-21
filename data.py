import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SYNTHETIC RESEARCH DATA ONLY — NOT FOR FORENSIC CASEWORK
# ============================================================

N_CASES = 10_000
N_POSITIVE = 5_000
SEED = 20261016
THETA = 0.01
PRIOR_PROBABILITY = 0.5

CODIS_CORE_20 = [
    "CSF1PO", "D3S1358", "D5S818", "D7S820", "D8S1179",
    "D13S317", "D16S539", "D18S51", "D21S11", "FGA",
    "TH01", "TPOX", "vWA", "D1S1656", "D2S441",
    "D2S1338", "D10S1248", "D12S391", "D19S433",
    "D22S1045",
]

MUTATION_RATE = {locus: 0.003 for locus in CODIS_CORE_20}
OUTPUT_CSV = "synthetic_paternity_10000.csv"
FREQUENCY_FILE = "allele_frequencies.json"
POPULATION = "Caucasian"


def load_frequencies(path, population):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if population not in raw:
        raise ValueError(f"Missing population: {population}")

    frequencies = {}

    for locus in CODIS_CORE_20:
        if locus not in raw[population]:
            raise ValueError(f"Missing frequency table: {locus}")

        table = {
            float(allele): float(prob)
            for allele, prob in raw[population][locus].items()
        }

        if not table or any(
            not math.isfinite(p) or p < 0 for p in table.values()
        ):
            raise ValueError(f"Invalid frequencies at {locus}")

        total = sum(table.values())
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(
                f"Frequencies at {locus} sum to {total}, not 1"
            )

        if any(not math.isfinite(a) for a in table):
            raise ValueError(f"Invalid allele value at {locus}")

        frequencies[locus] = table

    return frequencies


def genotype_distribution(freq, theta):
    """Normalized unordered genotype prior under the chosen model."""
    if not 0 <= theta <= 1:
        raise ValueError("THETA must be between 0 and 1")

    alleles = list(freq)
    result = {}

    for i, a in enumerate(alleles):
        for b in alleles[i:]:
            if a == b:
                p = freq[a] * (theta + (1 - theta) * freq[a])
            else:
                p = 2 * freq[a] * freq[b] * (1 - theta)

            result[(a, b)] = p

    total = sum(result.values())
    if total <= 0:
        raise ValueError("Genotype distribution has zero probability")

    return {g: p / total for g, p in result.items()}


def make_genotype(prior_dist, rng):
    """Sample a genotype from the same prior used in the LR denominator."""
    genotypes = list(prior_dist)
    probabilities = np.array(
        [prior_dist[g] for g in genotypes], dtype=float
    )
    return genotypes[int(rng.choice(len(genotypes), p=probabilities))]


def build_mutation_kernel(freq, rate):
    """
    Build a row-stochastic transmission model.

    A mutation moves to one of the supported neighboring alleles.
    If no supported neighbor exists, the allele remains unchanged.
    """
    alleles = list(freq)
    kernel = {}

    for source in alleles:
        neighbors = [
            target for target in alleles
            if (
                np.isclose(abs(target - source), 1.0, atol=0.1)
                or np.isclose(abs(target - source), 0.2, atol=0.01)
            )
        ]

        row = {}

        if neighbors:
            row[source] = 1.0 - rate
            for target in neighbors:
                row[target] = rate / len(neighbors)
        else:
            # No modeled destination: mutation cannot be represented here.
            row[source] = 1.0

        if not np.isclose(sum(row.values()), 1.0):
            raise ValueError(f"Invalid mutation kernel for allele {source}")

        kernel[source] = row

    return kernel


def transmit(genotype, kernel, rng):
    allele = float(rng.choice(genotype))
    destinations = list(kernel[allele])
    probabilities = [kernel[allele][a] for a in destinations]

    return float(rng.choice(destinations, p=probabilities))


def make_child(mother, father, kernel, rng):
    maternal = transmit(mother, kernel, rng)
    paternal = transmit(father, kernel, rng)
    return tuple(sorted((maternal, paternal)))


def child_likelihood(child, mother, father, kernel):
    """
    P(unordered child genotype | mother, father).

    Enumerate both maternal allele copies and both paternal allele
    copies. Compare the resulting unordered genotype to the child.
    """
    child = tuple(sorted(map(float, child)))
    total = 0.0

    for maternal_allele in mother:
        for paternal_allele in father:
            p_m = 0.5 * kernel[float(maternal_allele)].get(
                child[0], 0.0
            )
            p_f = 0.5 * kernel[float(paternal_allele)].get(
                child[1], 0.0
            )

            if tuple(sorted((float(child[0]), float(child[1])))) == child:
                # The first assignment is included below using the
                # explicit unordered-genotype probability calculation.
                pass

            # Sum over ordered transmissions that produce this
            # unordered genotype. Handle homozygotes only once.
            total += 0.25 * (
                kernel[float(maternal_allele)].get(child[0], 0.0)
                * kernel[float(paternal_allele)].get(child[1], 0.0)
            )

            if child[0] != child[1]:
                total += 0.25 * (
                    kernel[float(maternal_allele)].get(child[1], 0.0)
                    * kernel[float(paternal_allele)].get(child[0], 0.0)
                )

    return total


def locus_lr(child, mother, alleged_father, kernel, prior_dist):
    numerator = child_likelihood(
        child, mother, alleged_father, kernel
    )

    denominator = math.fsum(
        p_genotype * child_likelihood(child, mother, genotype, kernel)
        for genotype, p_genotype in prior_dist.items()
    )
    if denominator <= 0:
            raise ValueError(
                f"Zero denominator: child={child}, mother={mother}, "
                f"alleged_father={alleged_father}"
            )

    if numerator <= 0:
        # Diagnostic: do not silently invent a positive LR.
        return 0.0

    return numerator / denominator

def encode_genotype(genotype):
    return f"{genotype[0]:.1f}/{genotype[1]:.1f}"


def posterior_from_log_lr(log_lr, prior_probability):
    if not 0 < prior_probability < 1:
        raise ValueError("Prior probability must be strictly between 0 and 1")

    prior_log_odds = math.log(
        prior_probability / (1.0 - prior_probability)
    )
    z = log_lr + prior_log_odds

    # Numerically stable logistic transformation.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def generate():
    if not 0 <= N_POSITIVE <= N_CASES:
        raise ValueError("N_POSITIVE must be between 0 and N_CASES")

    if any(not 0 <= rate <= 1 for rate in MUTATION_RATE.values()):
        raise ValueError("Mutation rates must be between 0 and 1")

    freq = load_frequencies(FREQUENCY_FILE, POPULATION)

    prior_distributions = {
        locus: genotype_distribution(freq[locus], THETA)
        for locus in CODIS_CORE_20
    }

    kernels = {
        locus: build_mutation_kernel(freq[locus], MUTATION_RATE[locus])
        for locus in CODIS_CORE_20
    }

    rng = np.random.default_rng(SEED)
    labels = np.array(
        [1] * N_POSITIVE + [0] * (N_CASES - N_POSITIVE),
        dtype=np.int8,
    )
    rng.shuffle(labels)

    rows = []

    for i, label in enumerate(labels):
        row = {
            "case_id": f"CASE{i + 1:05d}",
            "paternity_label": int(label),
        }

        log_lrs = []

        for locus in CODIS_CORE_20:
            prior_dist = prior_distributions[locus]
            kernel = kernels[locus]

            mother = make_genotype(prior_dist, rng)

            # Always generate a biological father.
            biological_father = make_genotype(prior_dist, rng)

            if label == 1:
                # Positive: the alleged father is the biological father.
                alleged_father = biological_father
            else:
                # Negative: the alleged father is unrelated to the
                # independently generated biological father.
                alleged_father = make_genotype(prior_dist, rng)

            child = make_child(
                mother, biological_father, kernel, rng
            )

            lr = locus_lr(
                child, mother, alleged_father, kernel, prior_dist
            )

            if not math.isfinite(lr) or lr < 0:
                raise ValueError(
                    f"Invalid LR {lr} in case {i + 1}, locus {locus}"
                )
            if lr == 0:
                print(
                    f"Zero LR diagnostic: case={i + 1}, locus={locus}, "
                    f"mother={mother}, alleged_father={alleged_father}, "
                    f"child={child}"
                )
                # Keep the true zero for now; do not replace it with a pseudocount.
                log_lrs.append(float("-inf"))
            else:
                log_lrs.append(math.log(lr))

            row[f"mother_{locus}"] = encode_genotype(mother)
            row[f"father_{locus}"] = encode_genotype(alleged_father)
            row[f"child_{locus}"] = encode_genotype(child)

        log_clr = math.fsum(log_lrs)

        # Store the LR in ordinary space for compatibility with the
        # original CSV. It may be 0 or infinity if outside float range.
        try:
            clr = math.exp(log_clr)
        except OverflowError:
            clr = math.inf

        row["combined_lr"] = clr
        row["log_combined_lr"] = log_clr
        row["posterior_probability"] = posterior_from_log_lr(
            log_clr, PRIOR_PROBABILITY
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.12g")

    print(f"Generated {len(df):,} synthetic cases")
    print(f"Positive labels: {(df['paternity_label'] == 1).sum():,}")
    print(f"Negative labels: {(df['paternity_label'] == 0).sum():,}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate()
