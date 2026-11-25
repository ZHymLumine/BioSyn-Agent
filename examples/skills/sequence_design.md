# Sequence design

Read the task's requested sequence, objectives and constraints. Use the configured
DNA and RNA tools to inspect a candidate, then submit it with `biosyn_submit`.

For a complete CDS, provide the coding-strand DNA, its RNA transcript and the
translated protein. The transcript replaces T with U; the protein omits the
terminal stop codon. Preserve the requested protein when changing synonymous codons.

Use the evaluation feedback to choose the next edit. Give each revision a new
candidate ID and identify its predecessor with `parent_id`. Consult
`biosyn_get_state` for the current design history and remaining budget.

Describe each score using the quantity returned by its evaluator, such as GC
fraction, RNA folding energy or model likelihood. Finish with the selected
candidate and its evaluated properties.
