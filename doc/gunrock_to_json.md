From Gunrock to JSON {#gunrock_to_json}
====================

How do we export information from Gunrock?
------------------------------------------

Typical programs use "printf" to emit a bunch of unstructured information. As the program gets more sophisticated, "printf" is augmented with command-line switches, perhaps a configuration file, but it's hard to easily parse random printf output.

More structured is JSON format. JSON is a nested dict (hash) data structure with arbitrary keys (and arbitrary nesting). It can be used to hold scalar, vector, and key-value data. Many tools can input and output JSON. It is a good choice for exporting information from a Gunrock program.

Ideally, we would declare a C++ struct or class and simply print it to stdout. The particular issue with C++, however, is that it poorly supports introspection: a running C++ executable does not know anything about the internals of the program that created it. Specifically, it doesn't know its own variable names, at least not without an incredible amount of pain. Maintaining a set of strings that map to variable names is undesirable since that can get out of sync.

Instead, we've elected to use a dict data structure that stores the JSON data, and we will write directly into it. We are using a [header-only JSON generator](http://www.codeproject.com/Articles/20027/JSON-Spirit-A-C-JSON-Parser-Generator-Implemented) based on Boost Spirit. It's used like this:

    json_spirit::mObject info;
    info["engine"] = "Gunrock";

Currently we can output JSON data in one of three ways, controlled from the command line:

- `--json` writes the JSON structure to stdout.
- `--jsonfile=filename` writes the JSON structure to *filename*.
- `--jsondir=dir` writes the JSON structure to an automatically-uniquely-named file in the *dir* directory. This is the preferred option, since presumably there is a single directory where all JSON output is stored, and all Gunrock runs can use the same `--jsondir=dir` command-line option.

The current "automatically-uniquely-named file" producer creates `name_dataset_time.json`. By design, the file name should not matter, so long as it is unique (and thus doesn't stomp on other files in the same directory when it's written). *No program or person should rely on the contents of file names.*

The current JSON structure (`info`) is passed by reference between various routines. Yuechao suggests that putting `info` into the global `Test_Parameter` is a better idea, where it can be passed into the enactor's and problem's `Init()` routines.

We don't have a fixed schema (yet), so what's below reflects what we put into the test_bfs code. Some of these are likely not useful for any analysis, but it's preferable to include too much info in the JSON output rather than not enough.

Fields that should be in any Gunrock run
----------------------------------------

- *avg_duty*. Average kernel running duty, calculated by kernel run time / kernel lifetime.
- *command_line*. Reflects how the program was run. Use `args.GetEntireCommandLine()`.
- *dataset*. Short name of dataset, likely the stem of `foo.mtx` (in this case, `foo`). Examples: `ak2010` or `soc_liveJournal1`. Important to standardize dataset names to allow joins when compiling stats.
- *elapsed*. Elapsed runtime, e.g., `0.25508800148963928`. Measured in ms.
- *engine*. `Gunrock` for Gunrock runs.
- *git_commit_sha1*. The git commit identifier, e.g., `6f775b82359c3c8529b8878cdae80e9dfbaf5330`. (Strategy from [StackOverflow](http://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake), of course.) Settable by

        #include <gunrock/util/gitsha1.h>
        info["git_commit_sha1"] = g_GIT_SHA1;
- *gpuinfo*. What GPU did we use? Use:

        #include <gunrock/util/sysinfo.h>
        Gpuinfo gpuinfo;
        info["gpuinfo"] = gpuinfo.getGpuinfo();
- *gunrock_version*. Reflects what's set in the top-level CMakeFiles; set as a constant "#define" using a compiler flag `-D` during compilation. (Since *gunrock_version* should not change often, this "#define" is acceptable; we only need rebuild when we update the version.) Example: `0.1.0`.

        #define STR(x) #x
        #define XSTR(x) STR(x)
        info["gunrock_version"] = XSTR(GUNROCKVERSION);
- *iterations*. The number of times we run the test, with runtime averaged over all runs. Yuduo suggests considering better names, e.g.:
  - *search_depth*. number of BSP super-steps
  - *max_iter*. Maximum allowed BSP super-steps, breaking after reaching this value
  - *iterations*. Number of runs of a primitive
- *mark_predecessors*. Set by command-line flag; true or false. Can be set for any primitive, as long as when a primitive does advance, it keeps track of predecessors.
- *name*. Name of primitive; I reused the `Stats` name `GPU BFS`. `BFS` is probably more appropriate. We definitely need canonical names here.
- *num_gpus*. Integer. I think this only works with 1 now. We'll extend to multi-GPU later.
- *quick*: Set by command-line flag; true or false. I don't know what this is.
- *sysinfo*. What GPU/system did we use? Use:

        #include <gunrock/util/sysinfo.h>
        Sysinfo sysinfo;
        info["sysinfo"] = sysinfo.getSysinfo();
- *time*. When the test was run. This is a standard format from `ctime()`, e.g. ``Wed Jul 22 09:04:05 2015\n``. (Oddly, the `\n` is part of the format.)

        time_t now = time(NULL);
        info["time"] = ctime(&now);

- *userinfo*. Who ran this test? Use:

        #include <gunrock/util/sysinfo.h>
        Userinfo userinfo;
        info["userinfo"] = userinfo.getUserinfo();
- *verbose*: Set by command-line flag; true or false. Presumably relates to logging output.

Fields for any traversal-based primitive
----------------------------------------

- *edges_visited*. Self-explanatory.
- *m_temps*. Millions of edge-traversals per second, e.g., `0.40378222181564849`.
- *nodes_visited*. Self-explanatory.
- *redundant_work*. Percentage indicating concurrent discovery calculated by: `[(total edges we put into frontier - edge visited) / edge visited] * 100`.
  Actual code:

        // measure duplicate edges put through queue
        redundant_work = ((double)total_queued - edges_visited) / edges_visited;

- *total_queued*. Total elements put into the frontier.

BFS-specific fields
-------------------

- *impotence*: Set by command-line flag; true or false.
- *instrumented*: Set by command-line flag; true or false.
- *iterations*. Not entirely sure what this indicates. Example: `10`.
- *max_grid_size*. Number of grids for GPU kernels, but sometimes the enactor itself will calculate a optimal number and ignore this setting.
- *max_queue_sizing*. Used to calculate the initial size of frontiers (\#vertices * queue-sizing, or \#edges * queue-sizing). --*in-sizing* is for similar purpose, but for those buffers used by GPU-GPU communication.
- *search_depth*. Presumably maximum distance from the source found in the graph. Integer.
- *traversal_mode*. Switch for *advance* kernels. 0: load-balanced partitioning (Davidson); 1: Merrill's load-balance strategy. Default is currently dynamically choosing between the two.
- *undirected*. Is the graph is undirected (true) or directed (false)? Command-line option.
- *vertex_id*. Starting vertex ID. Integer.

Thread safety: "Using JSON Spirit with Multiple Threads"
========================================================

 "If you intend to use JSON Spirit in more than one thread, you will need to uncomment the following line near the top of json_spirit_reader.cpp.

"//#define BOOST_SPIRIT_THREADSAFE"

"In this case, Boost Spirit will require you to link against Boost Threads."

[link](http://www.codeproject.com/KB/recipes/JSON_Spirit.aspx#threads)

If compilation is too slow
==========================

Currently we're using the header-only version of JSON Spirit, which is easier to integrate but requires more compilation. The (docs)[http://www.codeproject.com/KB/recipes/JSON_Spirit.aspx#reduc] have ways to increase compilation speed.
