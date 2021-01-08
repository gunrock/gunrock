# begin /* SET GENCODE_SM */
set(GENCODE_SM10
  -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13
  -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20
  -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30
  -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35
  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37
  -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50
  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM52
  -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52)
set(GENCODE_SM60
  -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM72
  -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_72,code=compute_72)
set(GENCODE_SM75
  -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)
set(GENCODE_SM80
  -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80)
set(GENCODE_SM86
  -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86)

#set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX
# end /* SET GENCODE_SM */

option(ESSENTIALS_GENCODE_SM10
    "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
    OFF)

option(ESSENTIALS_GENCODE_SM13
    "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
    OFF)

option(ESSENTIALS_GENCODE_SM20
    "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
    OFF)

option(ESSENTIALS_GENCODE_SM30
    "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
    OFF)

option(ESSENTIALS_GENCODE_SM35
    "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K40)"
    OFF)

option(ESSENTIALS_GENCODE_SM37
    "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
    OFF)

option(ESSENTIALS_GENCODE_SM50
    "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
    OFF)

option(ESSENTIALS_GENCODE_SM52
    "ON to generate code for Compute Capability 5.2 devices (e.g. GeForce Titan X)"
    OFF)

#Pascal Architecture: CUDA 8
option(ESSENTIALS_GENCODE_SM60
    "ON to generate code for Compute Capability 6.0 devices (e.g. Tesla P100)"
    OFF)

option(ESSENTIALS_GENCODE_SM61
    "ON to generate code for Compute Capability 6.1 devices (e.g. GeForce GTX 1080)"
    ON)

#Volta Architecture: CUDA 9
option(ESSENTIALS_GENCODE_SM70
    "ON to generate code for Compute Capability 7.0 devices (e.g. Volta V100)"
    OFF)

option(ESSENTIALS_GENCODE_SM72
    "ON to generate code for Compute Capability 7.2 devices (e.g. Jetson AGX Xavier)"
    OFF)

#Turing Architecture: CUDA 10
option(ESSENTIALS_GENCODE_SM75
    "ON to generate code for Compute Capability 7.5 devices (e.g. GTX 1160 or RTX 2080)"
    OFF)

#Ampere Architecture: CUDA 11
option(ESSENTIALS_GENCODE_SM80
    "ON to generate code for Compute Capability 8.0 devices"
    OFF)

option(ESSENTIALS_GENCODE_SM86
    "ON to generate code for Compute Capability 8.6 devices"
    OFF)

if (ESSENTIALS_GENCODE_SM10)
    set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(ESSENTIALS_GENCODE_SM10)

if (ESSENTIALS_GENCODE_SM13)
    set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(ESSENTIALS_GENCODE_SM13)

if (ESSENTIALS_GENCODE_SM20)
    set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(ESSENTIALS_GENCODE_SM20)

if (ESSENTIALS_GENCODE_SM30)
    set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(ESSENTIALS_GENCODE_SM30)

if (ESSENTIALS_GENCODE_SM35)
    set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(ESSENTIALS_GENCODE_SM35)

if (ESSENTIALS_GENCODE_SM37)
    set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(ESSENTIALS_GENCODE_SM37)

if (ESSENTIALS_GENCODE_SM50)
    set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(ESSENTIALS_GENCODE_SM50)

if (ESSENTIALS_GENCODE_SM52)
    set(GENCODE ${GENCODE} ${GENCODE_SM52})
endif(ESSENTIALS_GENCODE_SM52)

if (ESSENTIALS_GENCODE_SM60)
    set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(ESSENTIALS_GENCODE_SM60)

if (ESSENTIALS_GENCODE_SM61)
    set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(ESSENTIALS_GENCODE_SM61)

if (ESSENTIALS_GENCODE_SM70)
    set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif(ESSENTIALS_GENCODE_SM70)

if (ESSENTIALS_GENCODE_SM72)
    set(GENCODE ${GENCODE} ${GENCODE_SM72})
endif(ESSENTIALS_GENCODE_SM72)

if (ESSENTIALS_GENCODE_SM75)
    set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif(ESSENTIALS_GENCODE_SM75)

if (ESSENTIALS_GENCODE_SM80)
    set(GENCODE ${GENCODE} ${GENCODE_SM80})
endif(ESSENTIALS_GENCODE_SM80)

if (ESSENTIALS_GENCODE_SM86)
    set(GENCODE ${GENCODE} ${GENCODE_SM86})
endif(ESSENTIALS_GENCODE_SM86)

message(STATUS "Listing chosen GENCODE commands")
foreach(code IN LISTS GENCODE)
    message(STATUS "${code}")
endforeach()
# end /* Configure ESSENTIALS build options */