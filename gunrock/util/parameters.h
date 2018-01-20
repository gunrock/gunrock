// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * parameters.h
 *
 * @brief Parameter class to hold running parameters
 */

#pragma once

#include <string>
#include <map>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <typeinfo>
#include <typeindex>
#include <gunrock/util/error_utils.cuh>
//#include <gunrock/util/types.cuh>
#include <gunrock/util/str_to_T.cuh>

namespace gunrock {
namespace util {

//#define ENABLE_PARAMETER_DEBUG

using Parameter_Flag = unsigned int;

enum {
    NO_ARGUMENT        = 0x1,
    REQUIRED_ARGUMENT  = 0x2,
    OPTIONAL_ARGUMENT  = 0x4,

    //ZERO_VALUE         = 0x10,
    SINGLE_VALUE       = 0x20,
    MULTI_VALUE        = 0x40,

    REQUIRED_PARAMETER = 0x100,
    OPTIONAL_PARAMETER = 0x200,
    INTERNAL_PARAMETER = 0x400,
};

class Parameter_Item
{
public:
    std::string     name;
    Parameter_Flag  flag;
    std::string     default_value;
    std::string     description;
    std::string     detailed_description;
    std::string     value;
    bool            use_default;
    const std::type_info* value_type_info;
    std::string     file_name;
    int             line_num;

    Parameter_Item()
    :   name            (""),
        flag            (OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER),
        default_value   (""),
        description     (""),
        detailed_description(""),
        value           (""),
        use_default     (true),
        value_type_info (NULL),
        file_name       (""),
        line_num        (0)
    {
    }

    Parameter_Item(const std::type_info* value_tinfo)
    :   name            (""),
        flag            (OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER),
        default_value   (""),
        description     (""),
        detailed_description(""),
        value           (""),
        use_default     (true),
        value_type_info (value_tinfo),
        file_name       (""),
        line_num        (0)
    {
    }

    Parameter_Item(const Parameter_Item &item)
    :   name            (item.name),
        flag            (item.flag),
        default_value   (item.default_value),
        description     (item.description),
        detailed_description(item.detailed_description),
        value           (item.value),
        use_default     (item.use_default),
        value_type_info (item.value_type_info),
        file_name       (item.file_name),
        line_num        (item.line_num)
    {
    }
}; // Parameter_Item

class Parameters
{
private:
    std::map<std::string, Parameter_Item> p_map;
    std::string summary;
    std::string command_line;

public:
    Parameters(
        std::string summary = "test <graph-type> [optional arguments]")
    :   summary(summary)
    {
        p_map.clear();
        Use("quiet",
            OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER,
            false,
            "No output (unless --json is specified).",
            __FILE__, __LINE__);

        Use("v",
            OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER,
            false,
            "Print verbose per iteration debug info.",
            __FILE__, __LINE__);

        Use("help",
            OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER,
            false,
            "Print this usage.",
            __FILE__, __LINE__);

        Use("quick",
            OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER,
            false,
            "Whether to skip the CPU reference validation process.",
            __FILE__, __LINE__);
    }

    ~Parameters()
    {
        p_map.clear();
    }

    cudaError_t Use(
        std::string     name,
        Parameter_Flag  flag,
        std::string     default_value,
        std::string     description,
        const std::type_info* value_type_info,
        const char*     file_name,
        int             line_num,
        std::string     detailed_description = "")
    {
        // special case for no argument parameters
        if ((flag & NO_ARGUMENT) == NO_ARGUMENT)
        {
            if (std::type_index(*value_type_info) != std::type_index(typeid(bool)))
            {
                return GRError(cudaErrorInvalidValue,
                    "NO_ARGUMENT can only be applied to bool parameter, but "
                    + name + " is " + TypeName(value_type_info),
                    file_name, line_num);
            }

            if (default_value == "true")
            {
                std::cerr << "Warning: Bool parameter " << name
                    << "(" << file_name << ":" << line_num << ")"
                    << " with NO_ARGUMENT and true default value, has no effect"
                    << std::endl;
            }
        }

        Parameter_Item p_item(
            ((flag & MULTI_VALUE) == MULTI_VALUE && !isVector(value_type_info)) ?
            toVector(value_type_info) : value_type_info);
        if (isVector(value_type_info))
            flag = (flag & (~SINGLE_VALUE)) | MULTI_VALUE;
        p_item.name           = name;
        p_item.flag           = flag;
        p_item.default_value  = default_value;
        p_item.description    = description;
        p_item.detailed_description = detailed_description;
        p_item.value          = default_value;
        p_item.use_default    = true;
        p_item.file_name      = std::string(file_name);
        p_item.line_num       = line_num;

        // test for duplication
        auto it = p_map.find(name);
        if (it != p_map.end()
            && (it -> second.file_name != std::string(file_name)
                || it -> second.line_num != line_num))
        {
            return GRError(cudaErrorInvalidSymbol,
                "Parameter " + name + " has been defined before, "
                + it -> second.file_name + ":"
                + std::to_string(it -> second.line_num),
                file_name, line_num);
        }

        //std::cout << name << " flag = " << flag << " ";
        //std::cout << std::ios::hex << flag << std::endl;
        //std::cout << flag / 16 / 16 << (flag / 16) % 16 << flag % 16 << std::endl;
        p_map[name] = p_item;
        return cudaSuccess;
    }

    template <typename T>
    cudaError_t Use(
        std::string     name,
        Parameter_Flag  flag,
        T               default_value,
        std::string     description,
        const char*     file_name,
        int             line_num,
        std::string     detailed_description = "")
    {
        std::ostringstream ostr;
        ostr << default_value;
        return Use(name, flag,
            ostr.str(), description,
            (((flag & MULTI_VALUE) == MULTI_VALUE
               && !IS_VECTOR<T>::value ) ?
              &typeid(std::vector<T>) : &typeid(T)),
            file_name, line_num, detailed_description);
    } // Use()

    cudaError_t Set(
        std::string name,
        std::string value)
    {
        // find the record
        auto it = p_map.find(name);
        if (it == p_map.end())
        {
            return GRError(cudaErrorInvalidValue,
                "Parameter " + name + " has not been defined", __FILE__, __LINE__);
        }

        if (!isValidString(value, it -> second.value_type_info))
        {
            Parameter_Item &p_item = it->second;
            return GRError(cudaErrorInvalidValue,
                 "Parameter " + name + "(" + p_item.file_name + ":"
                 + std::to_string(p_item.line_num) + ") only takes in "
                 + TypeName(p_item.value_type_info)
                 + ", value " + value + " is invalid.",
                __FILE__, __LINE__);
        }

        #ifdef ENABLE_PARAMETER_DEBUG
            util::PrintMsg("Parameter " + name + " <- " + value);
        #endif

        it -> second.value = value;
        it -> second.use_default = false;
        return cudaSuccess;
    }

    template <typename T>
    cudaError_t Set(
        std::string name,
        T           value)
    {
        std::ostringstream ostr;
        ostr << value;
        return Set(name, ostr.str());
    } // Set()

    bool UseDefault(std::string name)
    {
        auto it = p_map.find(name);
        if (it == p_map.end())
        {
            GRError(cudaErrorInvalidValue,
                "Parameter " + name + " has not been defined", __FILE__, __LINE__);
            return false;
        }
        return it -> second.use_default;
    }

    cudaError_t Get(
        const std::string name,
        std::string &value) const
    {
        auto it = p_map.find(name);
        if (it == p_map.end())
        {
            return GRError(cudaErrorInvalidValue,
                "Parameter " + name + " has not been defined", __FILE__, __LINE__);
        }

        value = it -> second.value;
        return cudaSuccess;
    }

    template <typename T>
    cudaError_t Get(
        const std::string name,
        T          &value,
        int         base = 0) const
    {
        std::string str_value;
        cudaError_t retval = Get(name, str_value);
        if (retval) return retval;

        //std::istringstream istr(str_value);
        //istr >> value;
        char *str_end = NULL;
        value = strtoT<T>(str_value.c_str(), &str_end, base);
        if (str_end == NULL || (*str_end != '\0' && *str_end != ','))
        {
            //std::cout << int(*str_end) << "|" << str_end - str_value.c_str() << std::endl;
            return GRError(cudaErrorInvalidValue,
                "Value " + str_value + " is not a invalid "
                + TypeName(&typeid(T)) + " for parameter " + name,
                __FILE__, __LINE__);
        }
        //std::cout << "str_value = " << str_value << std::endl;
        return cudaSuccess;
    }

    template <typename T>
    T Get(const std::string name, int base = 0) const
    {
        T val;
        Get(name, val, base);
        return val;
    }

    template <typename T>
    T Get(const char* name, int base = 0) const
    {
        T val;
        Get(std::string(name), val, base);
        //std::cout << "val = " << val << std::endl;
        return val;
    }// Get()

    bool Have(std::string name)
    {
        auto it = p_map.find(name);
        if (it == p_map.end())
            return false;
        return true;
    }

    cudaError_t Check_Required()
    {
        for (auto it = p_map.begin(); it != p_map.end(); it++)
        {
            Parameter_Item &p_item = it -> second;
            if ((p_item.flag & REQUIRED_PARAMETER) != REQUIRED_PARAMETER)
                continue;
            if (p_item.value == "")
            {
                return GRError(cudaErrorInvalidValue,
                    "Required parameter " + p_item.name
                    + "(" + p_item.file_name
                    + ":" + std::to_string(p_item.line_num) + ")"
                    + " is not present.", __FILE__, __LINE__);
            }
        }
        return cudaSuccess;
    }

    cudaError_t Read_In_Opt(
        std::string option,
        std::string argument)
    {
        auto it = p_map.find(option);
        Parameter_Item &p_item = it -> second;
        if ((std::type_index(*(p_item.value_type_info)) == std::type_index(typeid(bool))
            || std::type_index(*(p_item.value_type_info)) == std::type_index(typeid(std::vector<bool>))) && argument == "")
        {
            argument = "true";
        }

        if ((p_item.flag & SINGLE_VALUE) == SINGLE_VALUE)
        {
            if (argument.find(",") != std::string::npos)
            {
                return GRError(cudaErrorInvalidValue, "Parameter " + p_item.name
                    + "(" + p_item.file_name + ":"
                    + std::to_string(p_item.line_num)
                    + ") only takes single argument.",
                    __FILE__, __LINE__);
            }

            if (!p_item.use_default)
            {
                std::cerr << "Warnning : Parameter " << p_item.name
                    << "(" << p_item.file_name << ":"
                    << p_item.line_num
                    << ") specified more than once, only latter value "
                    << argument << " is effective." << std::endl;
            }
        }

        if ((p_item.flag & MULTI_VALUE) == MULTI_VALUE)
        {
            if (!p_item.use_default)
            {
                std::cerr << "Warnning : Parameter " << p_item.name
                    << "(" << p_item.file_name << ":"
                    << p_item.line_num
                    << ") specified more than once, latter value "
                    << argument << " is appended to pervious ones." << std::endl;
                argument = p_item.value + "," + argument;
            }
        }

        if (!isValidString(argument, p_item.value_type_info))
        {
            return GRError(cudaErrorInvalidValue,
                "Parameter " + p_item.name
                + "(" + p_item.file_name +":"
                + std::to_string(p_item.line_num)
                + ") only takes in " + TypeName(p_item.value_type_info)
                + ", argument " + argument
                + " is invalid.", __FILE__, __LINE__);
        }

        return Set(option, argument);
    }

    cudaError_t Parse_CommandLine(
        const int   argc,
        char* const argv[])
    {
        cudaError_t retval = cudaSuccess;
        command_line = "";
        for (int i = 0; i < argc; i++)
            command_line = command_line + (i == 0 ? "" : " ")
                + std::string(argv[i]);

        typedef struct option Option;
        int num_options = p_map.size();
        Option *long_options = new Option[num_options + 1];
        std::string *names   = new std::string[num_options + 2];

        int i = 0;
        // load parameter list into long_options
        for (auto it = p_map.begin(); it != p_map.end(); it++)
        {
            long_options[i].name = it -> second.name.c_str();
            long_options[i].has_arg = ((it -> second.flag) & (0x07)) / 2;
            long_options[i].flag = NULL;
            long_options[i].val  = i+1;
            if (i+1 >= '?') long_options[i].val++;
            names[long_options[i].val] = it -> second.name;
            i++;
        }
        long_options[num_options].name = 0;
        long_options[num_options].has_arg = 0;
        long_options[num_options].flag = 0;
        long_options[num_options].val = 0;

        int option_index = 0;
        do {
            i = getopt_long_only (argc, argv, "", long_options, &option_index);
            switch (i)
            {
            case '?' :
                //std::cout << "Invalid parameter " << std::endl;
                break;

            case -1  :
                //end of known options
                break;

            default  :
                //std::cout << i << std::endl;
                if (i <= 0 || i > ((num_options + 1 >= '?') ? num_options + 2 : num_options + 1))
                {
                    std::cerr << "Invalid parameter" << std::endl;
                    break;
                }

                std::string argument(optarg == NULL ? "" : optarg);
                Read_In_Opt(names[i], argument);
                break;
            }

            if (retval) break;
        } while (i!=-1);

        #ifdef ENABLE_PARAMETER_DEBUG
            if (optind < argc-1)
                std::cout << "Left over arguments" << std::endl;
        #endif
        for (int i=optind; i<argc; i++)
        {
            bool valid_parameter = false;
            #ifdef ENABLE_PARAMETER_DEBUG
                std::cout << argv[i] << std::endl;
            #endif
            if (i == optind)
            {
                auto it = p_map.find("graph-type");
                if (it != p_map.end())
                {
                    Read_In_Opt("graph-type", std::string(argv[i]));
                    valid_parameter = true;
                }
            }

            if (i == optind + 1)
            {
                auto it = p_map.find("graph-type");
                if (it != p_map.end())
                {
                    it = p_map.find("graph-file");
                    if (it != p_map.end() && Get<std::string>("graph-type") == "market")
                    {
                        Read_In_Opt("graph-file", std::string(argv[i]));
                        valid_parameter = true;
                    }
                }
            }

            if (!valid_parameter)
            {
                GRError(cudaErrorInvalidValue,
                    "Unknown option " + std::string(argv[i]),
                    __FILE__, __LINE__);
            }
        }

        delete[] long_options; long_options = NULL;
        delete[] names; names = NULL;
        return retval;
    } // Phase_CommandLine()

    std::string Get_CommandLine()
    {
        return command_line;
    }

    cudaError_t Print_Para(Parameter_Item &item)
    {
        cudaError_t retval = cudaSuccess;

        std::cout << "--" << item.name << " : "
            << TypeName(item.value_type_info)
            << ", default = ";
        if (item.default_value != "")
        {
            if (std::type_index(*(item.value_type_info))
                == std::type_index(typeid(bool)))
                std::cout << ((item.default_value == "0") ? "false" : "true");
            else
                std::cout << item.default_value;
        }
        std::cout << std::endl << "\t" << item.description << std::endl;
        return retval;
    }

    cudaError_t Print_Help()
    {
        cudaError_t retval = cudaSuccess;
        std::cout << summary << std::endl;

        if (!UseDefault("graph-type"))
        {
            auto it = p_map.find(Get<std::string>("graph-type"));
            std::cout << "finding " << Get<std::string>("graph-type") << std::endl;
            if (it != p_map.end())
            {
                Print_Para(it -> second);
                if (it -> second.detailed_description != "")
                    std::cout<< "\t" << it -> second.detailed_description
                        << std::endl;
                return retval;
            }
        }

        for (int t=0; t<2; t++)
        {
            bool first_parameter = true;
            Parameter_Flag selected_parameters
                = ((t == 0) ? REQUIRED_PARAMETER : OPTIONAL_PARAMETER);

            for (auto it = p_map.begin(); it != p_map.end(); it++)
            {
                // jump if not the selected ones
                if ((it -> second.flag & selected_parameters)
                    != selected_parameters)
                    continue;
                //std::cout << it -> second.flag << std::endl;
                if (first_parameter)
                {
                    if (selected_parameters == REQUIRED_PARAMETER)
                        std::cout << std::endl << "Required arguments:" << std::endl;
                    else
                        std::cout << std::endl << "Optional arguments:" << std::endl;
                    first_parameter = false;
                }

                Print_Para(it -> second);
            }
        }

        return retval;
    } // Print_Help()

    std::map<std::string, std::string> List()
    {
        std::map<std::string, std::string> list;
        list.clear();

        for (auto it = p_map.begin(); it != p_map.end(); it ++)
        {
            list[it -> second.name] = it -> second.value;
        }
        return list;
    }


    template <typename InfoT>
    void List(InfoT& info)
    {
        for (auto it = p_map.begin(); it != p_map.end(); it ++)
        {
            auto item = it -> second;
            auto tidx = std::type_index(*(item.value_type_info));
            auto &str = item.value;
            std::string str_end;
            if      (tidx == std::type_index(typeid(         char )))
                info.SetVal(item.name, strtoT<               char >(str, str_end));
            else if (tidx == std::type_index(typeid(  signed char )))
                info.SetVal(item.name, strtoT<        signed char >(str, str_end));
            else if (tidx == std::type_index(typeid(unsigned char )))
                info.SetVal(item.name, strtoT<      unsigned char >(str, str_end));
            else if (tidx == std::type_index(typeid(         short)))
                info.SetVal(item.name, strtoT<               short>(str, str_end));
            else if (tidx == std::type_index(typeid(unsigned short)))
                info.SetVal(item.name, strtoT<      unsigned short>(str, str_end));
            else if (tidx == std::type_index(typeid(         int  )))
                info.SetVal(item.name, strtoT<                int64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(unsigned int  )))
                info.SetVal(item.name, strtoT<               uint64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(         long )))
                info.SetVal(item.name, strtoT<                int64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(unsigned long )))
                info.SetVal(item.name, strtoT<               uint64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(         long long)))
                info.SetVal(item.name, strtoT<                int64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(unsigned long long)))
                info.SetVal(item.name, strtoT<               uint64_t>(str, str_end));
            else if (tidx == std::type_index(typeid(         bool )))
                info.SetVal(item.name, strtoT<               bool >(str, str_end));
            else if (tidx == std::type_index(typeid(         float)))
                info.SetVal(item.name, strtoT<               float>(str, str_end));
            else if (tidx == std::type_index(typeid(         double)))
                info.SetVal(item.name, strtoT<               double>(str, str_end));
            else if (tidx == std::type_index(typeid(    long double)))
                info.SetVal(item.name, strtoT<               double>(str, str_end));
            else if (tidx == std::type_index(typeid(     std::string)))
                info.SetVal(item.name, strtoT<           std::string>(str, str_end));
            else if (tidx == std::type_index(typeid(         char*)))
                info.SetVal(item.name, strtoT<               char*>(str, str_end));

            else if (tidx == std::type_index(typeid(std::vector<         char >)))
                info.SetVal(item.name, strtoT<      std::vector<         char > >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<  signed char >)))
                info.SetVal(item.name, strtoT<      std::vector<  signed char > >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<unsigned char >)))
                info.SetVal(item.name, strtoT<      std::vector<unsigned char > >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         short>)))
                info.SetVal(item.name, strtoT<      std::vector<         short> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<unsigned short>)))
                info.SetVal(item.name, strtoT<      std::vector<unsigned short> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         int  >)))
                info.SetVal(item.name, strtoT<      std::vector<          int64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<unsigned int  >)))
                info.SetVal(item.name, strtoT<      std::vector<         uint64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         long >)))
                info.SetVal(item.name, strtoT<      std::vector<          int64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<unsigned long >)))
                info.SetVal(item.name, strtoT<      std::vector<         uint64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         long long>)))
                info.SetVal(item.name, strtoT<      std::vector<          int64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<unsigned long long>)))
                info.SetVal(item.name, strtoT<      std::vector<         uint64_t> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         bool >)))
            {
                auto vec = strtoT<      std::vector<     std::string> >(str, str_end);
                if (vec.size() == 1)
                    info.SetVal(item.name, vec[0]);
                else
                    info.SetVal(item.name, vec);
            } else if (tidx == std::type_index(typeid(std::vector<         float>)))
                info.SetVal(item.name, strtoT<      std::vector<         float> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         double>)))
                info.SetVal(item.name, strtoT<      std::vector<         double> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<    long double>)))
                info.SetVal(item.name, strtoT<      std::vector<         double> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<     std::string>)))
                info.SetVal(item.name, strtoT<      std::vector<     std::string> >(str, str_end));
            else if (tidx == std::type_index(typeid(std::vector<         char*>)))
                info.SetVal(item.name, strtoT<      std::vector<         char*> >(str, str_end));
            else info.SetVal(item.name,str);
        }
    }
}; // class Parameters;

} // namespace util
} // nanespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
