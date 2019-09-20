TOPC benchmarking scripts
=========================
Make sure you have downloaded the required datasets by:
```cd <gunrockRoot>/dataset/large && make TOPC```
In case you see errors from wget complaining about certificates, use the following command instead:
```cd <gunrockRoot>/dataset/large && make WGET="wget --no-check-certificate" TOPC```

Running the benchmarks after that is as simple as:
```cd <gunrockRoot>/dataset/test-scripts/topc && ./topc-test.sh <EXEDIR> <DATADIR>```
If you have followed the default build commands, the above commandwill be:
```cd <gunrockRoot>/dataset/test-scripts/topc && ./topc-test.sh ../../../build/bin/ ../../large/```
