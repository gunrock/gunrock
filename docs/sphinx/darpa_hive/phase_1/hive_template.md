# Name Of Application

!> JDO notes, delete these when you copy this to `hive_yourworkflowname`: The goal of this report is to be useful to DARPA and to your colleagues. This is not a research paper. Be very honest. If there are limitations, spell them out. If something is broken or works poorly, say so. Above all else, make sure that the instructions to replicate the results you report are good instructions, and the process to replicate are as simple as possible; we want anyone to be able to replicate these results in a straightforward way.

One-paragraph summary of application, written at a high level.

## Summary of Results

One or two sentences that summarize "if you had one or two sentences to sum up your whole effort, what would you say". I will copy this directly to the high-level executive summary in the first page of the report. Talk to JDO about this. Write it last, probably.

## Summary of Gunrock Implementation

As long as you need. Provide links (say, to papers) where appropriate. What was the approach you took to implementing this on a GPU / in Gunrock? Pseudocode is fine but not necessary. Whatever is clear.

Be specific about how to map the algorithm to Gunrock operators. That is helpful for everyone.

Be specific about what you actually implemented with respect to the entire workflow (most workflows have non-graph components; as a reminder, our goal is implementing single-GPU code on only the graph components where the graph is static).

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

(e.g., "build Gunrock's `dev-refactor` branch", "this particular dataset needs to be in this particular directory")

### Running the application

<code>
include a transcript
</code>

Note: This run / these runs need to be on DARPA's DGX-1.

### Output

What is output when you run? Output file? JSON? Anything else? How do you extract relevant statistics from the output?

How do you make sure your output is correct/meaningful? (What are you comparing against?)

## Performance and Analysis

How do you measure performance? What are the relevant metrics? Runtime? Throughput? Some sort of accuracy/quality metric?

### Implementation limitations

e.g.:

- Size of dataset that fits into GPU memory (what is the specific limitation?)
- Restrictions on the type/nature of the dataset

### Comparison against existing implementations

- Reference implementation (python? Matlab?)
- OpenMP reference

Comparison is both performance and accuracy/quality.



### Performance limitations

e.g., random memory access?

## Next Steps

### Alternate approaches

If you had an infinite amount of time, is there another way (algorithm/approach) we should consider to implement this?

### Gunrock implications

What did we learn about Gunrock? What is hard to use, or slow? What potential Gunrock features would have been helpful in implementing this workflow?

### Notes on multi-GPU parallelization

What will be the challenges in parallelizing this to multiple GPUs on the same node?

Can the dataset be effectively divided across multiple GPUs, or must it be replicated?

### Notes on dynamic graphs

(Only if appropriate)

Does this workload have a dynamic-graph component? If so, what are the implications of that? How would your implementation change? What support would Gunrock need to add?

### Notes on larger datasets

What if the dataset was larger than can fit into GPU memory or the aggregate GPU memory of multiple GPUs on a node? What implications would that have on performance? What support would Gunrock need to add?

### Notes on other pieces of this workload

Briefly: What are the important other (non-graph) pieces of this workload? Any thoughts on how we might implement them / what existing approaches/libraries might implement them?
