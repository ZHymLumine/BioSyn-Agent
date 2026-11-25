# Contributing

Install the development environment using the README, then run `make check` and
`make test` before proposing a change.

For a new scientific tool, provide an MCP interface, a configuration example and
tests for its inputs and returned fields. Name prediction quantities and units
explicitly so they can be mapped into task objectives.

For a new application, define its decision schema, required resources, design
assembly and evaluation contract. Keep resource preparation separate from model
execution, and use a new output directory for each run.

Describe the behavior your change introduces, how to use it and the checks you ran.
