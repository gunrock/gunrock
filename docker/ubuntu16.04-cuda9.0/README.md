#### Instructions

```
user@home$ docker build . -t gunrock                       # Build container
user@home$ nvidia-docker run -it gunrock /bin/bash         # Start bash session in container

root@docker$ cd gunrock/build
root@docker$ make test                                     # Run tests
root@docker$ ./bin/cc market ../dataset/small/data_sm.mtx  # Run app
```