// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * array_utils.cuh
 *
 * @brief array utilities for Array1D
 */

#pragma once

#include <string>
#include <fstream>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace util {

/*static const unsigned int FLAGBASE = 0x00;
static const unsigned int PINNED   = 0x01;
static const unsigned int UNIFIED  = 0x02;
static const unsigned int STREAM   = 0x04;
static const unsigned int MAPPED   = 0x04;*/
static const unsigned int NONE       = 0x00;
static const unsigned int TARGETBASE = 0x10;
static const unsigned int HOST       = 0x11;
static const unsigned int CPU        = 0x11;
static const unsigned int DEVICE     = 0x12;
static const unsigned int GPU        = 0x12;
static const unsigned int DISK       = 0x14;
static const unsigned int TARGETALL  = 0x1F;

#define ARRAY_DEBUG false

template <
    typename _SizeT,
    typename _Value>
struct Array1D
{
    typedef _SizeT SizeT;
    typedef _Value Value;

private:
    std::string  name;
    std::string  file_name;
    SizeT        size;
    unsigned int flag;
    bool         use_cuda_alloc;
    unsigned int setted, allocated;
    Value        *h_pointer;
    Value        *d_pointer;

public:
    Array1D()
    {
        //name.reserve(40);
        //file_name.reserve(512);
        name      = "";
        file_name = "";
        h_pointer = NULL;
        d_pointer = NULL;
        flag      = cudaHostAllocDefault;
        setted    = NONE;
        allocated = NONE;
        use_cuda_alloc = false;
        Init(0,NONE,false,flag);
    } // Array1D()

   Array1D(const char* const name)
    {
        //this->name.reserve(40);
        //file_name.reserve(512);
        this->name= std::string(name);
        file_name = "";
        h_pointer = NULL;
        d_pointer = NULL;
        setted    = NONE;
        allocated = NONE;
        flag      = cudaHostAllocDefault;
        use_cuda_alloc = false;
        Init(0,NONE,false,NONE);
    }

    /*Array1D(SizeT size, std::string name = "", unsigned int target = HOST, bool use_cuda_alloc = false, unsigned int flag = cudaHostAllocDefault)
    {
        this->name= name;
        file_name = "";
        h_pointer = NULL;
        d_pointer = NULL;
        setted    = NONE;
        allocated = NONE;
        Init(size,target,use_cuda_alloc,flag);
    } // Array1D(...)*/

    virtual ~Array1D()
    {
    #ifdef __CUDA_ARCH__
    #else
        //Release();
    #endif
    } // ~Array1D()

    cudaError_t Init(
        SizeT        size,
        unsigned int target = HOST,
        bool use_cuda_alloc = false,
        unsigned int flag   = cudaHostAllocDefault)
    {
        cudaError_t retval = cudaSuccess;

        //if (ARRAY_DEBUG) {printf("%s Init size = %d, target = %d\n", name.c_str(), size, target);fflush(stdout);}
        if (retval = Release()) return retval;
        setted     = NONE;
        allocated  = NONE;
        this->size = size;
        this->flag = flag;
        this->use_cuda_alloc = use_cuda_alloc;

        if (size == 0) return retval;
        retval = Allocate(size,target);
        return retval;
   } // Init(...)

    void SetFilename(std::string file_name)
    {
        this->file_name=file_name;
        setted = setted | DISK;
    }

    void SetName(std::string name)
    {
        this->name=name;
    }

    cudaError_t Allocate(SizeT size, unsigned int target = HOST)
    {
        cudaError_t retval = cudaSuccess;

        /*if (((target & HOST) == HOST) && //((target & DEVICE) == DEVICE) &&
            (use_cuda_alloc ))
        {
            if (retval = Release(HOST  )) return retval;
            //if (retval = Release(DEVICE)) return retval;
            UnSetPointer(HOST);//UnSetPointer(DEVICE);
            if ((setted    & (~(target    | DISK)) == NONE) &&
                (allocated & (~(allocated | DISK)) == NONE)) this->size=size;

            if (retval = GRError(cudaHostAlloc((void **)&h_pointer, sizeof(Value) * size, flag),
                         name + "cudaHostAlloc failed", __FILE__, __LINE__)) return retval;
            //if (retval = GRError(cudaHostGetDevicePointer((void **)&d_pointer, (void *)h_pointer,0),
                         //name + "cudaHostGetDevicePointer failed", __FILE__, __LINE__)) return retval;
            allocated = allocated | HOST  ;
            //allocated = allocated | DEVICE;
            if (ARRAY_DEBUG) {printf("%s allocated on HOST, size = %d, flag = %d\n",name.c_str(),size, flag);fflush(stdout);}
        } else {*/
        if ((target & HOST) == HOST)
        {
            if (retval = Release(HOST)) return retval;
            UnSetPointer(HOST);
            if ((setted    & (~(target    | DISK)) == NONE) &&
                (allocated & (~(allocated | DISK)) == NONE)) this->size=size;
            h_pointer = new Value[size];
            if (h_pointer == NULL)
                return GRError(name+" allocation on host failed", __FILE__, __LINE__);
            if (use_cuda_alloc)
            {
                retval = util::GRError(
                    cudaHostRegister((void*)h_pointer, sizeof(Value)*size, flag),
                    name+" cudaHostRegister failed.", __FILE__, __LINE__);
                if (retval) return retval;
            }
            allocated = allocated | HOST;
            if (ARRAY_DEBUG)
            {
                printf("%s\t allocated on HOST, length =\t %lld, "
                    "size =\t %lld bytes, pointer =\t %p\n",
                    name.c_str(), (long long) size,
                    (long long) size*sizeof(Value), h_pointer);
                fflush(stdout);
            }
        }
        //}

        if ((target & DEVICE) == DEVICE)
        {
            if (retval = Release(DEVICE)) return retval;
            UnSetPointer(DEVICE);
            if ((setted    & (~(target    | DISK)) == NONE) &&
                (allocated & (~(allocated | DISK)) == NONE)) this->size=size;

            /*if (ARRAY_DEBUG)
            {
                printf("%s\t allocating on DEVICE, length =\t %lld,"
                       "size =\t %lld bytes, pointer =\t %p\n",
                       name.c_str(), (long long) size,
                       (long long) size*sizeof(Value), d_pointer);
                fflush(stdout);
            }*/
            if (size!=0) {
                retval = GRError(
                    cudaMalloc((void**)&(d_pointer), sizeof(Value) * size),
                    name+" cudaMalloc failed", __FILE__, __LINE__);
                if (retval) return retval;
            }
            allocated = allocated | DEVICE;
            if (ARRAY_DEBUG)
            {
                printf("%s\t allocated on DEVICE, length =\t %lld, "
                    "size =\t %lld bytes, pointer =\t %p\n",
                    name.c_str(), (long long) size,
                    (long long) size*sizeof(Value), d_pointer);
                fflush(stdout);
            }
        }
        //}
        this->size=size;
        return retval;
    } // Allocate(...)

    cudaError_t Release(unsigned int target = TARGETALL)
    {
        cudaError_t retval = cudaSuccess;

        /*if (((allocated & HOST) == HOST)  && ((target    & DEVICE) == HOST) &&
        {
            if (retval = GRError(cudaFreeHost(h_pointer),name+" cudaFreeHost failed",__FILE__, __LINE__)) return retval;
            h_pointer = NULL;
            //d_pointer = NULL;
            allocated = allocated - HOST   + TARGETBASE;
            //allocated = allocated - DEVICE + TARGETBASE;
            if (ARRAY_DEBUG) {printf("%s released on HOST & DEVICE\n", name.c_str());fflush(stdout);}
        } else {*/
            if (((target & HOST) == HOST)&&((allocated & HOST) == HOST))
            {
                if (use_cuda_alloc)
                {
                    retval = GRError(
                        cudaHostUnregister((void*)h_pointer),
                        name+" cudaHostUnregister failed",__FILE__,__LINE__);
                    if (retval) return retval;
                }
                if (ARRAY_DEBUG)
                {
                    printf("%s\t released on HOST, length =\t %lld, pointer = %p\n",
                        name.c_str(), (long long) size, h_pointer);
                    fflush(stdout);
                }
                delete[] h_pointer;
                h_pointer = NULL;
                allocated = allocated - HOST + TARGETBASE;
            } else if ((target & HOST)==HOST && (setted & HOST) == HOST) {
                UnSetPointer(HOST);
            }

        if (((target & DEVICE) == DEVICE)&&((allocated & DEVICE) ==DEVICE))
        {
            if (ARRAY_DEBUG)
            {
                printf("%s\t releasing on DEVICE, length =\t %lld, pointer = %p\n",
                       name.c_str(), (long long) size, d_pointer);
                fflush(stdout);
            }
            retval = GRError(cudaFree((void*)d_pointer),
                name + " cudaFree failed", __FILE__, __LINE__);
            if (retval) return retval;
            d_pointer = NULL;
            allocated = allocated - DEVICE + TARGETBASE;
        }
        else if ((target & DEVICE) == DEVICE && (setted & DEVICE) == DEVICE)
        {
            UnSetPointer(DEVICE);
        }

        if (target == TARGETALL) size = 0;
        return retval;
    } // Release(...)

    __host__ __device__ SizeT GetSize() const
    {
        return this->size;
    }

    cudaError_t EnsureSize(SizeT size, bool keep = false, cudaStream_t stream = 0)
    {
        if (ARRAY_DEBUG)
        {
            printf("%s EnsureSize : %lld -> %lld\n",
                name.c_str(), (long long) this->size, (long long) size);
            fflush(stdout);
        }
        if (this->size >= size) return cudaSuccess;
        else {
            //printf("Expanding %s : %d -> %d\n",name.c_str(),this->size,size);fflush(stdout);
            if (!keep) return Allocate(size, allocated);
            else {
                Array1D<SizeT, Value> temp_array;
                cudaError_t retval = cudaSuccess;
                unsigned int org_allocated = allocated;

                temp_array.SetName("t_array");
                if (retval = temp_array.Allocate(size, allocated)) return retval;
                if ((allocated & HOST) == HOST)
                    memcpy(temp_array.GetPointer(HOST), h_pointer, sizeof(Value) * this->size);
                if ((allocated & DEVICE) == DEVICE)
                    MemsetCopyVectorKernel<<<256,256,0,stream>>>(
                        temp_array.GetPointer(DEVICE), d_pointer, this->size);
                if (retval = Release(HOST  )) return retval;
                if (retval = Release(DEVICE)) return retval;
                if ((org_allocated & HOST  ) == HOST  ) h_pointer = temp_array.GetPointer(HOST  );
                if ((org_allocated & DEVICE) == DEVICE) d_pointer = temp_array.GetPointer(DEVICE);
                allocated=org_allocated; this->size= size;
                if ((allocated & DEVICE) == DEVICE) temp_array.ForceUnSetPointer(DEVICE);
                if ((allocated & HOST  ) == HOST  ) temp_array.ForceUnSetPointer(HOST  );
                return retval;
            }
        }
    } // EnsureSize(...)

    cudaError_t ShrinkSize(SizeT size, bool keep = false, cudaStream_t stream = 0)
    {
        if (ARRAY_DEBUG)
        {
            printf("%s ShrinkSize : %lld -> %lld\n",
                name.c_str(), (long long) this->size, (long long) size);
            fflush(stdout);
        }
        if (this->size <= size) return cudaSuccess;
        else {
            //printf("Expanding %s : %d -> %d\n",name.c_str(),this->size,size);fflush(stdout);
            if (!keep) return Allocate(size, allocated);
            else {
                Array1D<SizeT, Value> temp_array;
                cudaError_t retval = cudaSuccess;
                unsigned int org_allocated = allocated;

                temp_array.SetName("t_array");
                if (retval = temp_array.Allocate(size, allocated)) return retval;
                if ((allocated & HOST) == HOST)
                    memcpy(temp_array.GetPointer(HOST), h_pointer, sizeof(Value) * this->size);
                if ((allocated & DEVICE) == DEVICE)
                    MemsetCopyVectorKernel<<<256,256,0,stream>>>(
                        temp_array.GetPointer(DEVICE), d_pointer, this->size);
                if (retval = Release(HOST  )) return retval;
                if (retval = Release(DEVICE)) return retval;
                if ((org_allocated & HOST  ) == HOST  ) h_pointer = temp_array.GetPointer(HOST  );
                if ((org_allocated & DEVICE) == DEVICE) d_pointer = temp_array.GetPointer(DEVICE);
                allocated=org_allocated; this->size= size;
                if ((allocated & DEVICE) == DEVICE) temp_array.ForceUnSetPointer(DEVICE);
                if ((allocated & HOST  ) == HOST  ) temp_array.ForceUnSetPointer(HOST  );
                return retval;
            }
        }
    } // ShrinkSize(...)

    __host__ __device__ __forceinline__ Value* GetPointer(unsigned int target = HOST)
    {
        if (target == HOST  )
        {
            //if (ARRAY_DEBUG) {printf("%s \tpointer on HOST   get = %p\n", name.c_str(), h_pointer);fflush(stdout);}
            return h_pointer;
        }
        if (target == DEVICE)
        {
            //if (ARRAY_DEBUG) {printf("%s \tpointer on DEVICE get = %p\n",name.c_str(), d_pointer);fflush(stdout);}
            return d_pointer;
        }
        return NULL;
    } // GetPointer(...)

    cudaError_t SetPointer(Value* pointer, SizeT size = -1, unsigned int target = HOST)
    {
        cudaError_t retval = cudaSuccess;
        if (size == -1) size=this->size;
        if (size < this->size)
        {
            if (ARRAY_DEBUG) {
                printf("%s\t setting pointer, size too small, size = %lld, this->size = %lld\n",
                    name.c_str(), (long long)size, (long long)this->size);
                fflush(stdout);
            }
            return GRError(name+" SetPointer size is too small",__FILE__,__LINE__);
        }

        if (target == HOST)
        {
            if (retval = Release(HOST)) return retval;
            if (use_cuda_alloc)
            {
                retval = util::GRError(
                    cudaHostRegister(pointer, sizeof(Value)*size, flag),
                    name+" cudaHostRegister failed.", __FILE__, __LINE__);
                if (retval) return retval;
            }
            h_pointer = pointer;
            if (setted == NONE && allocated == NONE) this->size=size;
            setted    = setted | HOST;
            if (ARRAY_DEBUG) {
                printf("%s\t setted on HOST, size =\t %lld, "
                    "pointer =\t %p setted = %d\n", name.c_str(),
                    (long long) this->size, h_pointer, setted);
                fflush(stdout);
            }
        }

        if (target == DEVICE)
        {
            if (retval = Release(DEVICE)) return retval;
            d_pointer = pointer;
            if (setted == NONE && allocated == NONE) this->size=size;
            setted    = setted | DEVICE;
            if (ARRAY_DEBUG) {
                printf("%s\t setted on DEVICE, size =\t %lld, "
                    "pointer =\t %p\n", name.c_str(),
                    (long long)this->size, d_pointer);
                fflush(stdout);
            }
        }
        return retval;
    } // SetPointer(...)

    cudaError_t ForceSetPointer(Value* pointer, unsigned int target = HOST)
    {
        if (target == HOST)
            h_pointer = pointer;
        if (target == DEVICE)
            d_pointer = pointer;
        return cudaSuccess;
    }

    void ForceUnSetPointer(unsigned int target = HOST)
    {
        if ((setted & target) == target)
            setted = setted - target + TARGETBASE;
        if ((allocated & target) == target)
            allocated = allocated - target + TARGETBASE;

        if (target == HOST && h_pointer!=NULL )
        {
            if (use_cuda_alloc) util::GRError(cudaHostUnregister((void*)h_pointer),
                name + " cudaHostUnregister failed.", __FILE__, __LINE__);
            h_pointer = NULL;
            if (ARRAY_DEBUG) {
                printf("%s\t unsetted on HOST\n",name.c_str());
                fflush(stdout);
            }
        }
        if (target == DEVICE && d_pointer!=NULL)
        {
            d_pointer = NULL;
            if (ARRAY_DEBUG) {
                printf("%s\t unsetted on DEVICE\n",name.c_str());
                fflush(stdout);
            }
        }
        if (target == DISK  ) file_name = "";
    } // UnSetPointer(...)

   void UnSetPointer(unsigned int target = HOST)
    {
        if ((setted & target) == target)
        {
            ForceUnSetPointer(target);
        }
    } // UnSetPointer(...)

    void SetMarker(int t, unsigned int target = HOST, bool s = true)
    {
        if (t==0)
        {
            if ((setted & target)!=target && s)
                setted = setted | target;
            else if ((setted & target)==target && (!s))
                setted = setted - target + TARGETBASE;
        } else if (t==1)
        {
             if ((allocated & target)!=target && s)
                allocated = allocated | target;
            else if ((setted & target)==target && (!s))
                allocated = allocated - target + TARGETBASE;
        }
    }

    cudaError_t Move(
        unsigned int source,
        unsigned int target,
        SizeT size  =-1,
        SizeT offset=0,
        cudaStream_t stream=0)
    {
        cudaError_t retval = cudaSuccess;
        if ((source == HOST || source == DEVICE) &&
            ((source & setted) != source) && ((source & allocated) != source))
            return GRError(name+" movment source is not valid", __FILE__, __LINE__);
        if ((target == HOST || target == DEVICE) &&
            ((target & setted) != target) && ((target & allocated) != target))
            if (retval = Allocate(this->size,target)) return retval;
        if ((target == DISK || source == DISK) && ((setted & DISK) != DISK))
            return GRError(name+" filename not set", __FILE__, __LINE__);
        if (size == -1) size=this->size;
        if (size > this->size)
            return GRError(name+" size is invalid",__FILE__, __LINE__);
        if (size+offset > this->size)
            return GRError(name+" size+offset is invalid", __FILE__, __LINE__);
        if (size == 0) return retval;
        if (ARRAY_DEBUG) {
            printf("%s Moving from %d to %d, size = %lld, offset = %lld, "
                "stream = %p, d_pointer = %p, h_pointer = %p\n",
                name.c_str(), source, target, (long long) size,
                (long long) offset, stream, d_pointer, h_pointer);
            fflush(stdout);
        }

        if      (source == HOST   && target == DEVICE) {
            if (use_cuda_alloc && stream != 0)
            {
                retval = GRError(
                    cudaMemcpyAsync( d_pointer + offset, h_pointer + offset,
                    sizeof(Value) * size, cudaMemcpyHostToDevice, stream),
                    name+" cudaMemcpyAsync H2D failed", __FILE__, __LINE__);
                if (retval) return retval;
            } else {
                retval = GRError(
                    cudaMemcpy( d_pointer + offset, h_pointer + offset,
                    sizeof(Value) * size, cudaMemcpyHostToDevice),
                    name+" cudaMemcpy H2D failed", __FILE__, __LINE__);
                if (retval) return retval;
            }
        }

        else if (source == DEVICE && target == HOST  ) {
            if (use_cuda_alloc && stream != 0)
            {
                //printf("%s MemcpyAsync\n");
                retval = GRError(
                    cudaMemcpyAsync( h_pointer + offset, d_pointer + offset,
                    sizeof(Value) * size, cudaMemcpyDeviceToHost, stream),
                    name+" cudaMemcpyAsync D2H failed", __FILE__, __LINE__);
                if (retval) return retval;
            } else {
                retval = GRError(
                    cudaMemcpy( h_pointer + offset, d_pointer + offset,
                    sizeof(Value) * size, cudaMemcpyDeviceToHost),
                    name+" cudaMemcpy D2H failed", __FILE__, __LINE__);
                if (retval) return retval;
            }
        }

        else if (source == HOST   && target == DISK  ) {
           std::ofstream fout;
           fout.open(file_name.c_str(), std::ios::binary);
           fout.write((const char*)(h_pointer+offset),sizeof(Value)*size);
           fout.close();
        }

        else if (source == DISK   && target == HOST  ) {
           std::ifstream fin;
           fin.open(file_name.c_str(), std::ios::binary);
           fin.read((char*)(h_pointer+offset),sizeof(Value)*size);
           fin.close();
        }

        else if (source == DEVICE && target == DISK  ) {
           bool t_allocated=false;
           if (((setted & HOST) != HOST) && ((allocated & HOST) !=HOST))
           {
               if (retval = Allocate(this->size,HOST)) return retval;
               t_allocated=true;
           }
           if (retval = Move(DEVICE,HOST,size,offset,stream)) return retval;
           if (retval = Move(HOST,DISK,size,offset)) return retval;
           if (t_allocated)
           {
               if (retval = Release(HOST)) return retval;
           }
        }

        else if (source == DISK   && target == DEVICE) {
           bool t_allocated=false;
           if (((setted & HOST) != HOST) && ((allocated & HOST) != HOST))
           {
               if (retval = Allocate(this->size,HOST)) return retval;
               t_allocated=true;
           }
           if (retval = Move(DISK,HOST,size,offset)) return retval;
           if (retval = Move(HOST,DEVICE,size,offset,stream)) return retval;
           if (t_allocated)
           {
               if (retval = Release(HOST)) return retval;
           }
        }
        return retval;
    } // Move(...)

    Array1D& operator=(const Array1D& other)
    {
       if (ARRAY_DEBUG)
       {
           printf("%s Assigment\n", other.name.c_str());fflush(stdout);
       }
       //name      = other.name     ;
       //file_name = other.file_name;
       size      = other.size     ;
       flag      = other.flag     ;
       use_cuda_alloc = other.use_cuda_alloc;
       setted    = other.setted   ;
       allocated = other.allocated;
       h_pointer = other.h_pointer;
       d_pointer = other.d_pointer;
       return *this;
    }

    __host__ __device__ __forceinline__ Value& operator[](std::size_t idx)
    {
    #ifdef __CUDA_ARCH__
        return d_pointer[idx];
    #else
        if (ARRAY_DEBUG)
        {
            if (h_pointer==NULL) GRError(name+" not defined on HOST",__FILE__, __LINE__);
            if (idx >= size) GRError(name+" access out of bound", __FILE__, __LINE__);
            //printf("%s @ %p [%ld]ed1\n", name.c_str(), h_pointer,idx);fflush(stdout);
        }
        return h_pointer[idx];
    #endif
    }

    __host__ __device__ const __forceinline__ Value& operator[](std::size_t idx) const
    {
    #ifdef __CUDA_ARCH__
        return const_cast<Value&>(d_pointer[idx]);
    #else
        if (ARRAY_DEBUG)
        {
            if (h_pointer==NULL) GRError(name+" not defined on HOST", __FILE__, __LINE__);
            if (idx >= size) GRError(name+" access out of bound", __FILE__, __LINE__);
            //printf("%s [%ld]ed2\n", name.c_str(), idx);fflush(stdout);
        }
        return const_cast<Value&>(h_pointer[idx]);
    #endif
    }

    __host__ __device__ __forceinline__ Value* operator->() const
    {
    #ifdef __CUDA_ARCH__
        return d_pointer;
    #else
        if (ARRAY_DEBUG)
        {
            if (h_pointer==NULL) GRError(name+" not deined on HOST", __FILE__, __LINE__);
            //printf("%s ->ed\n",name.c_str());fflush(stdout);
        }
        return h_pointer;
    #endif
    }

    __host__ __device__ __forceinline__ Value* operator+(const _SizeT& offset)
    {
    #ifdef __CUDA_ARCH__
        return d_pointer + offset;
    #else
        if (ARRAY_DEBUG)
        {
            if (h_pointer==NULL) GRError(name+" not deined on HOST", __FILE__, __LINE__);
            //printf("%s ->ed\n",name.c_str());fflush(stdout);
        }
        return h_pointer + offset;
    #endif
    }
}; // struct Array1D

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
