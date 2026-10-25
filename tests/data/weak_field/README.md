# weak_field reference fixture

This small-lattice fixture was copied verbatim from the earlier
`EasyDistillation/test` reference set on 2026-09-26. `manifest.json` records
the hashes and array shapes. The automated tests read these stored arrays only;
they do not import or execute the earlier application.

`test_weak_field.py` checks fixture integrity, basis orthogonality, and the
current CPU elemental implementation against the archived all-time reference.
The CUDA regression uses the same bundled files when a CUDA device is
available. The QIO, raw gauge, perambulator, and displacement arrays are kept
with the fixture for future adapter checks; the current CPU regression does
not interpret every file in the set.

The explicitly requested slow audit now rebuilds a separately specified
unimproved Wilson problem on the complete 4³ × 8 gauge field, including actual
Laplace/NV/spatial-point production, six-saved/nine-read propagators, elemental
and correlator checks. It does not reuse the archived propagator as the truth
for this new action. See the [slow-test list](../../slow/README.md) and
[detailed scientific audit](../../reference/weak_field_audit.md) for parameters,
independent references, results, and remaining acceptance gaps.
