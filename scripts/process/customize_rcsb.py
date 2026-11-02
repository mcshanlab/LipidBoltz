import argparse
import csv
import hashlib
import json
import multiprocessing
import pickle
import traceback
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Optional

import gemmi
import numpy as np
import rdkit
from mmcif import parse_mmcif
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm

from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)
from boltz.data.types import ChainInfo, InterfaceInfo, Record, Target

BIODOLPHIN_CSV_PATH = Path("PATH_TO_BIODOLPHIN_CSV") # Edit this to provide a biodolphin file
_BIODOLPHIN_SEQUENCES: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PDB:
    """A raw MMCIF PDB file."""

    id: str
    path: str


class Resource:
    """A shared resource for processing."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        value = self._redis.get(key)
        if value is not None:
            value = pickle.loads(value)  # noqa: S301
        return value

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def load_selected_cif_names(path: Path) -> set[str]:
    """Load the allowed CIF file names from a text file."""
    with path.open("r") as f:
        names = {
            line.strip().lower()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }

    normalized_names = set()
    for name in names:
        normalized_names.add(name)
        if name.endswith(".gz"):
            normalized_names.add(name[:-3])
        else:
            normalized_names.add(f"{name}.gz")

    return normalized_names


def fetch(
    datadir: Path,
    max_file_size: Optional[int] = None,
    selected_cif_names: Optional[set[str]] = None,
) -> list[PDB]:
    """Fetch the PDB files."""
    data = []
    excluded = 0
    skipped = 0
    for file in datadir.rglob("*.cif*"):
        if selected_cif_names is not None and file.name.lower() not in selected_cif_names:
            skipped += 1
            continue

        # The clustering file is annotated by pdb_entity id
        #pdb_id = str(file.stem).lower()
        #pdb_id = str[2:6].lower()
        BioDolphin_id = str(file.stem)
        #print(f'fetched: pdb_id: {pdb_id}')

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        # Create the target
        target = PDB(id=BioDolphin_id, path=str(file))
        data.append(target)

    print(f"Excluded {excluded} files due to size.")  # noqa: T201
    if selected_cif_names is not None:
        print(f"Skipped {skipped} files not listed in the CIF name file.")  # noqa: T201
    return data


def finalize(outdir: Path) -> None:
    """Run post-processing in main thread.

    Parameters
    ----------
    outdir : Path
        The output directory.

    """
    # Group records into a manifest
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in records_dir.iterdir():
        path = record
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except:  # noqa: E722
            failed_count += 1
            print(f"Failed to parse {record}")  # noqa: T201
    if failed_count > 0:
        print(f"Failed to parse {failed_count} entries.")  # noqa: T201
    else:
        print("All entries parsed successfully.")

    # Save manifest
    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)

def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


def get_biodolphin_sequences() -> dict[str, str]:
    """Load BioDolphinID -> protein sequence mapping once per process."""
    global _BIODOLPHIN_SEQUENCES
    if _BIODOLPHIN_SEQUENCES is None:
        sequences = {}
        with BIODOLPHIN_CSV_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                biodolphin_id = row["BioDolphinID"].strip()
                if biodolphin_id not in sequences:
                    sequences[biodolphin_id] = (
                        row["protein_Sequence"].strip().upper()
                    )
        _BIODOLPHIN_SEQUENCES = sequences
    return _BIODOLPHIN_SEQUENCES


def validate_mmcif_protein_sequence(
    path: str,
    expected_sequence: str,
    target_id: str,
) -> None:
    """Require the mmCIF protein entity to use the MSA query sequence."""
    structure = gemmi.read_structure(path)
    protein_entities = [
        entity
        for entity in structure.entities
        if entity.entity_type.name == "Polymer"
        and entity.polymer_type.name == "PeptideL"
    ]
    if len(protein_entities) != 1:
        msg = f"{target_id}: expected one protein entity, found {len(protein_entities)}"
        raise ValueError(msg)

    monomers = [
        gemmi.Entity.first_mon(monomer)
        for monomer in protein_entities[0].full_sequence
    ]
    structure_sequence = gemmi.one_letter_code(monomers)
    if structure_sequence != expected_sequence:
        msg = (
            f"{target_id}: mmCIF protein sequence does not match its MSA query: "
            f"mmCIF={len(structure_sequence)}, MSA={len(expected_sequence)}"
        )
        raise ValueError(msg)


def existing_structure_matches_sequence(
    struct_path: Path,
    expected_sequence: str,
) -> bool:
    """Check whether an existing encoded structure has the expected sequence."""
    with np.load(struct_path) as structure:
        protein_chains = [
            chain
            for chain in structure["chains"]
            if int(chain["mol_type"]) == 0
        ]
        if len(protein_chains) != 1:
            return False

        protein_chain = protein_chains[0]
        res_start = int(protein_chain["res_idx"])
        res_end = res_start + int(protein_chain["res_num"])
        monomers = structure["residues"]["name"][res_start:res_end].tolist()
        encoded_sequence = gemmi.one_letter_code(
            [gemmi.Entity.first_mon(monomer) for monomer in monomers]
        )
        return encoded_sequence == expected_sequence


def parse(data: PDB, resource: Resource, clusters: dict) -> Target:
    """Process a structure.

    Parameters
    ----------
    data : PDB
        The raw input data.
    resource: Resource
        The shared resource.

    Returns
    -------
    Target
        The processed data.

    """
    # Get the PDB id
    #pdb_id = data.id.lower()
    BioDolphinID = data.id
    sequence = get_biodolphin_sequences()[BioDolphinID]

    validate_mmcif_protein_sequence(
        path=data.path,
        expected_sequence=sequence,
        target_id=BioDolphinID,
    )

    # Parse structure
    parsed = parse_mmcif(data.path, resource)
    structure = parsed.data
    structure_info = parsed.info

    protein_chains = [
        chain for chain in structure.chains if int(chain["mol_type"]) == 0
    ]
    if len(protein_chains) != 1:
        msg = f"{BioDolphinID}: expected one processed protein chain."
        raise ValueError(msg)
    if int(protein_chains[0]["res_num"]) != len(sequence):
        msg = (
            f"{BioDolphinID}: processed protein length does not match its MSA query: "
            f"structure={int(protein_chains[0]['res_num'])}, MSA={len(sequence)}"
        )
        raise ValueError(msg)

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        #key = f"{pdb_id}_{chain['entity_id']}"
        key = None
        msa_id = ""
        if int(chain["mol_type"]) == 0:
            key = hash_sequence(sequence)
            msa_id = key
        elif int(chain["mol_type"]) == 3:
            key = BioDolphinID.split("-")[-1][:3]

        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id=msa_id,
                mol_type=int(chain["mol_type"]),
                cluster_id=clusters.get(key, -1) if key is not None else -1,
                num_residues=int(chain["res_num"]),
            )
        )

    # Get interface metadata
    interface_info = []
    for interface in structure.interfaces:
        chain_1 = int(interface["chain_1"])
        chain_2 = int(interface["chain_2"])
        interface_info.append(
            InterfaceInfo(
                chain_1=chain_1,
                chain_2=chain_2,
            )
        )

    # Create record
    record = Record(
        id=data.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)


def process_structure(
    data: PDB,
    outdir: Path,
    filters: list[StaticFilter],
    clusters: dict,
    redis_host: str = "localhost",
    redis_port: int = 7777,
) -> None:
    """Process a target.

    Parameters
    ----------
    data : PDB
        The raw input data.
    outdir : Path
        The output directory.
    filters : list[StaticFilter]
        The filters to apply.
    clusters : dict
        The cluster assignments.
    redis_host : str
        The Redis host.
    redis_port : int
        The Redis port.

    """
    # Create a new resource connection for this worker process
    resource = Resource(host=redis_host, port=redis_port)

    # Check if we need to process
    struct_path = outdir / "structures" / f"{data.id}.npz"
    record_path = outdir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        sequence = get_biodolphin_sequences()[data.id]
        if existing_structure_matches_sequence(struct_path, sequence):
            return
        print(f"Regenerating stale encoded structure for {data.id}")  # noqa: T201

    try:
        # Parse the target
        target: Target = parse(data, resource, clusters)
        structure = target.structure
        print(f"Parsed {data.id} has sturcture: {structure}")

        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return

    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)


def process(args) -> None:
    """Run the data processing task."""
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    records_dir = args.outdir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = args.outdir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    # Load clusters #TODO: Change this to load from customized cluster file
    with Path(args.clusters).open("r") as f:
        clusters: dict[str, str] = json.load(f)
        #clusters = {k.lower(): v.lower() for k, v in clusters.items()}

    # Load filters
    filters = [
        #ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        #UnknownFilter(),
        #ConsecutiveCA(max_dist=10.0),
        #ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    selected_cif_names = None
    if args.cif_name_list is not None:
        selected_cif_names = load_selected_cif_names(args.cif_name_list)

    # Get data points
    print("Fetching data...")
    data = fetch(
        args.datadir,
        max_file_size=args.max_file_size,
        selected_cif_names=selected_cif_names,
    )

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(data)))
    parallel = num_processes > 1

    # Run processing
    print("Processing data...")
    if parallel:
        # Create processing function
        fn = partial(
            process_structure,
            outdir=args.outdir,
            clusters=clusters,
            filters=filters,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
        )
        # Run processing in parallel
        p_umap(fn, data, num_cpus=num_processes)
    else:
        for item in tqdm(data):
            process_structure(
                item,
                outdir=args.outdir,
                clusters=clusters,
                filters=filters,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
            )
            break #TODO: Remove this line after testing

    # Finalize
    finalize(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MSA data.")
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="The data containing the MMCIF files.",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to the cluster file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--cif-name-list",
        type=Path,
        default=None,
        help="Optional text file listing CIF file names to process, one per line.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="The Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=7777,
        help="The Redis port.",
    )
    parser.add_argument(
        "--use-assembly",
        action="store_true",
        help="Whether to use assembly 1.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    process(args)
