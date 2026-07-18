"""Phase-one sandbox for the conserved-ψ core.

A dependency-light numerical demonstration that a *conserved charge* read from the
hidden part of a system's state survives the **replica test** (survivor vs.
identical-endpoint copy), where any reader of the observable endpoint alone cannot.

This is the smallest honest version of the thesis in ``docs/CORE-SPEC.md``:
identity is an invariant of the dynamics, native to the substrate — not a check
bolted onto a generic state. The toy here is hand-built (a 1-DOF Hamiltonian);
the next phase replaces the hand-built ``Q`` with one learned on the identity
manifold.
"""
