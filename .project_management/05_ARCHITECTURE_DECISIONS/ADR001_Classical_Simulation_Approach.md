# ADR 001: Classical Simulation Approach

## Status

Accepted

## Date

2025-05-24

## Context

This document outlines our approach to Classical Simulation in `quactuary`. The intent is to provide a state-of-the-art simulation system for testing model portfolios before loading production runs into quantum circuits. The classical simulations should be robust and provide top notch performance even in production runs while enterprise-level quantum computing is still being developed. The simulations are expected to run in enterprise insurance environments on large books of diverse policies, so the classical algorithms must be be highly optimized.

The interface must be familiar to Pandas users, so we support outputs in Pandas Series and DataFrame formats.

## Decision

We will provide optional parallel implementation that can be disabled in case upstream parallel processing is used.

We will not use GPU optimization for now because I don't have a good way to test on GPUs, but we should use NumPy arrays dynamically adjusted to available memory, which will set us up to implement GPU architecture later.

## Rationale

We need to make parallelization optional because nesting parallel runs can cause problems, and we cannot guarantee that enterprise applications won't be parallelized.
