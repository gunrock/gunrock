# Gunrock standard benchmarking scripts

Make sure you have downloaded the required datasets by:
```cd <gunrockRoot>/dataset/large && make STANDARD```

In case you see errors from wget complaining about certificates, use the following command instead:
```cd <gunrockRoot>/dataset/large && make WGET="wget --no-check-certificate" STANDARD```

Running the benchmarks after that is as simple as:
```cd <gunrockRoot>/scripts/standard && ./run-tests.sh <EXEDIR> <DATADIR> <DEVICE> <TAG>```

If you have followed the default build commands, the above commandwill be:
```cd <gunrockRoot>/scripts/standard && ./run-tests.sh ../../../build/bin/ ../../large/```

# Notes on test scripts

These are the scripts we have used repeatedly to test/benchmark Gunrock. We should expand on these as the project evolves, and only create new specialized scripts when needed for specific papers and merge them back into this standard folder.
