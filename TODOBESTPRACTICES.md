1. Add a Warmup Step (Best Practice)
If you are benchmarking or want to ensure the autotuner gets accurate reads, run a single "dummy" batch of data through your model before starting your actual training loop or timer. This forces the GPU to compile everything and get the cold-start overhead out of the way.

