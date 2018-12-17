# TODOs

## Algorithm

* [ ] batch_size calculation according to total_size and memory_limit
* [ ] device selection
* [ ] taking into account distinct radiuses for x and y
* [ ] element_size selection (float64, 32, 16)
* [ ] windows generation strategy: the right one or the legacy one
* [ ] mark internal functions with "_"
* [ ] add numpy version of algorithm and make an option for this
* [ ] make an option for auto or manual setting of memory_limit

## MVP

* [ ] complete "algorithm" todos
* [ ] add ProjMapper
* [ ] function for reading projections files
* [ ] read parameters from command line or file
* [ ] calculate dt
* [ ] calculate pix_size_km
* [ ] write vectors to files (right and filtered ones)

## Benchmark

* [ ] make notebook for benchmark
* [ ] generate set of params for benchmark
* [ ] define functions for calling legacy and pyvecplotter with params
* [ ] collect times of execution and plot them
* [ ] define function for comparing vectors (using legacy strategy of windows)

## Presentation

* [ ] slides for problem definition
* [ ] slides for implementation details (from loops to tensors)
* [ ] slides for benchmark
* [ ] slides for future improvements suggestions
