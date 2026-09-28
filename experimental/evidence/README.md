# Original experimental evidence

`records.tar.gz` contains the 249 original JSON and text records referenced by
the GPU manifests and selected-source validation, plus the H20 measurement run
used in the performance figure. Records preserve their original bytes.
`index.json` maps each original SHA-256 to an archive member and its original
workspace locations. Those locations are provenance labels; the files are now
available in this archive.

Run from the repository root:

```sh
python tools/verify_archive.py
python tools/verify_archive.py --evidence <record_sha256>
```

The first command verifies source hashes, the evidence archive and every record,
the selected-source timing grids, and all six figure aggregates. The second
prints a single original record for inspection. No GPU is needed.

The original records retain their recorded environment and baseline identities.
They establish the scope of the historical checks; they are not fresh release
audits and do not imply that every snapshot passed today's expanded suite.
