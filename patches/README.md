# Biomni execution support

`biomni-execution.patch` applies to Biomni revision
`400c1f366b96a35ca253e13c9b06c5076af41d65`.

It trims whitespace around executable blocks before language dispatch and returns
token-limited responses to the planner for a shorter continuation. The setup
script applies the patch, and execution regression tests cover both behaviors.

Biomni is distributed under Apache-2.0. Its license is included as
[`Biomni-LICENSE`](Biomni-LICENSE).
