"""Full CDS evaluators with explicitly named computational, not measured, endpoints."""

import ast
import math
import os
import sys
from pathlib import Path


def dna(value):
    if not isinstance(value, str) or not value or set(value) - set("ACGT"):
        raise ValueError("Expected nonempty canonical uppercase DNA")
    return value


def check_cds(task, design):
    from Bio.Seq import Seq

    cds = dna(design["cds"])
    protein = str(Seq(cds).translate(table=task["inputs"].get("genetic_code", 1), cds=True))
    if protein != task["inputs"]["protein_sequence"]:
        raise ValueError("The complete CDS does not encode the required protein")
    stop = task["inputs"].get("required_terminal_stop")
    if stop and cds[-3:] != stop:
        raise ValueError("Required terminal stop changed")
    return cds


def structure_metrics(rna, start):
    import RNA

    sequence = rna.replace("T", "U")
    fold = RNA.fold_compound(sequence)
    _, mfe = fold.mfe()
    # A partition-function calculation on a fixed local window around the annotated AUG.
    left, right = max(0, start - 30), min(len(sequence), start + 60)
    local = RNA.fold_compound(sequence[left:right])
    local.pf()
    bpp = local.bpp()
    positions = range(max(left, start - 4), min(right, start + 15))
    unpaired = []
    for pos in positions:
        i = pos - left + 1
        paired = sum(bpp[j][i] for j in range(1, i)) + sum(bpp[i][j] for j in range(i + 1, right - left + 1))
        unpaired.append(max(0.0, min(1.0, 1 - paired)))
    return {
        "rnafold_mfe_kcal_mol": float(mfe),
        "rnafold_mfe_per_nt": float(mfe) / len(sequence),
        "initiation_mean_unpaired_probability": sum(unpaired) / len(unpaired),
        "initiation_window": {
            "start_0based": left,
            "end_exclusive": right,
            "scored_start_0based": max(left, start - 4),
            "scored_end_exclusive": min(right, start + 15),
        },
    }


class RiboNNPredictor:
    def __init__(self, resource_root):
        import pandas as pd
        import torch

        self.root = Path(resource_root)
        sys.path.insert(0, str(self.root))
        from src.model import RiboNN
        from src.data import DataFrameDataset
        from src.utils.helpers import extract_config

        self.dataset_class = DataFrameDataset
        torch.set_num_threads(2)
        frame = pd.read_csv(self.root / "human/runs.csv")
        self.config = extract_config(frame, frame.iloc[0].run_id)
        # Same top-five-per-fold selection as the authors, using THEIR validation R2.
        selected = frame.groupby("params.test_fold", group_keys=False).apply(
            lambda rows: rows.sort_values("metrics.val_r2", ascending=False).head(5), include_groups=True
        )
        self.models, self.ids = [], selected.run_id.tolist()
        for rid in self.ids:
            model = RiboNN(**self.config)
            model.load_state_dict(
                torch.load(
                    self.root / "human" / rid / "state_dict.pth", map_location="cpu", weights_only=True
                )
            )
            model.eval()
            self.models.append(model)
        # Extract the canonical head ordering from the original inference function.
        tree = ast.parse((self.root / "src/predict.py").read_text())
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("TE_108T,")
        ]
        if len(strings) != 1:
            raise ValueError("Cannot identify original human target ordering")
        self.columns = strings[0].split(",")
        self.index = self.columns.index("TE_HEK293T")

    def encode(self, prefix, cds, suffix):
        import pandas as pd

        if len(prefix) > 1381 or len(cds + suffix) > 11937 or len(prefix + cds + suffix) > 12288:
            raise ValueError("Transcript outside original RiboNN input limits; no truncation permitted")
        cfg = self.config
        df = pd.DataFrame(
            [
                {
                    "tx_sequence": prefix + cds + suffix,
                    "utr5_size": len(prefix),
                    "cds_size": len(cds),
                    "tx_size": len(prefix + cds + suffix),
                }
            ]
        )
        ds = self.dataset_class(
            df,
            None,
            1381,
            11937,
            len(prefix + cds + suffix),
            *[
                cfg[k]
                for k in [
                    "pad_5_prime",
                    "split_utr5_cds_utr3_channels",
                    "label_codons",
                    "label_3rd_nt_of_codons",
                    "label_utr5",
                    "label_utr3",
                    "label_splice_sites",
                    "label_up_probs",
                ]
            ],
        )
        return ds[0].unsqueeze(0)

    def predict(self, prefix, cds, suffix):
        import torch

        x = self.encode(prefix, cds, suffix)
        with torch.inference_mode():
            values = [float(model(x)[0, self.index]) for model in self.models]
        return {
            "ribonn_HEK293T_score": sum(values) / len(values),
            "ribonn_model_count": len(values),
            "ribonn_model_sd": (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values))
            ** 0.5,
        }


class SalukiPredictor:
    def __init__(self, root):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tensorflow as tf
        import tf_keras as keras

        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        class StochasticShift(keras.layers.Layer):
            """Original augmentation is identity during inference; training is disallowed."""

            def __init__(self, shift_max=3, symmetric=False, pad="uniform", **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs, training=False):
                if training:
                    raise ValueError("This adapter supports inference only")
                return inputs

        self.model = keras.models.load_model(
            Path(root) / "model_best.h5", compile=False, custom_objects={"StochasticShift": StochasticShift}
        )

    def encode(self, prefix, cds, suffix):
        import numpy as np

        full = prefix + cds + suffix
        if len(full) > 12288:
            raise ValueError("Transcript exceeds Saluki limit; no truncation permitted")
        x = np.zeros((1, 12288, 6), dtype=np.float32)
        for i, base in enumerate(full):
            x[0, i, "ACGT".index(base)] = 1
        x[0, len(prefix) : len(prefix) + len(cds) : 3, 4] = 1
        # Synthetic in-vitro-transcribed RNA has no splice-junction annotations.
        return x

    def predict(self, prefix, cds, suffix):
        return float(self.model(self.encode(prefix, cds, suffix), training=False).numpy()[0, 0])


class SequenceEvaluator:
    def __init__(self, resources):
        self.resources = Path(resources)
        self.ribonn = self.saluki = None

    def evaluate(self, task, design):
        cds = check_cds(task, design)
        inputs = task["inputs"]
        metrics = {"protein_preserved": True, "cds_gc_fraction": (cds.count("G") + cds.count("C")) / len(cds)}
        if task["track"] == "bacterial_cds":
            from ostir import run_ostir

            prefix = inputs["fixed_5prime_context"]
            suffix = inputs["fixed_3prime_context_after_stop"]
            full = prefix + cds + suffix
            rows = run_ostir(
                full, start=len(prefix) + 1, end=len(prefix) + 1, threads=1, aSD="ACCUCCUUA", decimal_places=8
            )
            if len(rows) != 1 or rows[0]["start_position"] != len(prefix) + 1 or rows[0]["expression"] <= 0:
                raise RuntimeError("OSTIR did not evaluate the annotated start codon")
            metrics.update(
                predicted_tir=float(rows[0]["expression"]),
                log10_predicted_tir=math.log10(float(rows[0]["expression"])),
            )
            scope = "OSTIR translation initiation model, not full-length protein yield."
            raw = {"ostir": rows[0]}
        elif task["track"] == "mammalian_mrna":
            prefix = inputs["fixed_rna_prefix"].replace("U", "T")
            barcode = dna(design["assay_barcode"].replace("U", "T"))
            if len(barcode) != inputs["assay_barcode_length"]:
                raise ValueError("Incorrect assay barcode length")
            if inputs.get("fixed_assay_barcode") and barcode != inputs["fixed_assay_barcode"].replace(
                "U", "T"
            ):
                raise ValueError("The task fixes the assay barcode")
            suffix = (
                inputs["fixed_suffix_before_barcode"] + barcode + inputs["fixed_suffix_after_barcode"]
            ).replace("U", "T")
            full = prefix + cds + suffix
            if design["rna_body"] != full.replace("T", "U"):
                raise ValueError("RNA body differs from the annotated scaffold/CDS/barcode assembly")
            if self.ribonn is None:
                self.ribonn = RiboNNPredictor(self.resources / "compute_models/ribonn")
            if self.saluki is None:
                self.saluki = SalukiPredictor(self.resources / "compute_models/saluki")
            metrics.update(self.ribonn.predict(prefix, cds, suffix))
            metrics["saluki_human_score"] = self.saluki.predict(prefix, cds, suffix)
            scope = (
                "RiboNN HEK293T translation score and Saluki human consensus stability score. "
                "Saluki is not calibrated to this synthetic HEK293T assay; neither score is a measured "
                "ribosome load or half-life in hours. Cap, poly(A) processing and modifications are not modelled."
            )
            raw = {
                "ribonn_model_ids": self.ribonn.ids,
                "saluki_checkpoint": "f0_c0/model0_best.h5",
                "saluki_splice_channel": "all zero: synthetic RNA; no splicing",
            }
        else:
            raise ValueError("Unsupported sequence track")
        metrics.update(structure_metrics(full, len(prefix)))
        return {
            "valid": True,
            "metrics": metrics,
            "raw": raw,
            "scope": scope,
            "experimental_outcome": "not_measured",
        }
