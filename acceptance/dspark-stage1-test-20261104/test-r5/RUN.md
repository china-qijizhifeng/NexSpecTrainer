# Test r5: one eight-H100 scheduling migration

User prioritized starting one8GPU group and explicitly relaxed the850GB GPFS reserve for this run. Use node090 with init-only fixed image import and actual8H100 check. Worker generation, four variants, frozen inputs and ten EOS A/A diagnostics are unchanged from r4. Source r4 must be cancelled exactly while still Pending with no worker output, and its controller identity stopped before this attempt starts. No training restart. Temporary50GB GPFS write reserve; submit/cache-copy need60GB. Current serving replicas are preserved.
