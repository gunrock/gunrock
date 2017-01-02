/*

1. Add new option in the long_options[] table
2. There are 4 options: {new option name, value option, NULL, class}
  2.1 Value option is an int: 0 for no value for this key, 1 for value must provided for this key,
      and 2 for value may or may not provided
  2.2 Class is an int: 1 for key without any values, 2 for key with only 1 value allowed.
  2.3 You may initialize new classes for special usage of the command line arguments
      If you want to add special usage, you must implement a new class in the switch table in the switch table
3. OptionValue is a class for special usage of the command line arguments, typical checking provided here are
  3.1 double_check() checks if the key has any duplicate values
    3.1.1 You may set up whether duplicate values allowed in disallowDuplicate()
  3.2 fileCheck checks if the key has values of file names that are NULL or cannot open

*/
#include <stdio.h>
#include <map>
#include <getopt.h>
#include <iostream>
#include <string.h>
#include <string>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;
//double check, use only the last one

enum labels{
  zeroValueClass = 1,
  oneValueClass = 2,
  multiValueClass = 3,
  marketClass = 4,
  mustHaveValue = 1
};

struct option long_options[] =
{
  {"market", mustHaveValue, NULL, marketClass},     //has to change her
  {"rgg",mustHaveValue, NULL, multiValueClass},
  {"instrument",mustHaveValue,NULL,oneValueClass},
  {"size_check",mustHaveValue,NULL,oneValueClass}, // has a not there
  {"debug_mode",mustHaveValue,NULL,oneValueClass},
  {"quick_mode",mustHaveValue,NULL,oneValueClass},
  {"quiet_mode",mustHaveValue,NULL,oneValueClass},
  {"idempotent",mustHaveValue,NULL,oneValueClass},// BFS
  {"mark_predecessors",mustHaveValue,NULL,oneValueClass},// BFS
  {"json",mustHaveValue,NULL,oneValueClass},
  {"jsonfile",mustHaveValue,NULL,multiValueClass},
  {"jsondir",mustHaveValue,NULL,multiValueClass},
  {"src",mustHaveValue,NULL,multiValueClass},  // (NOT SURE)
  {"grid-size",mustHaveValue,NULL,multiValueClass},
  {"iteration-num",mustHaveValue,NULL,multiValueClass},
  {"max-iter",mustHaveValue,NULL,multiValueClass},
  {"queue-sizing",mustHaveValue,NULL,multiValueClass},
  {"queue-sizing1",mustHaveValue,NULL,multiValueClass},
  {"partition_method",mustHaveValue,NULL,multiValueClass},
  {"partition-factor",mustHaveValue,NULL,multiValueClass},
  {"partition-seed",mustHaveValue,NULL,multiValueClass},
  {"traversal-mode",mustHaveValue,NULL,multiValueClass},
  {"ref_filename",mustHaveValue,NULL,multiValueClass},
  {"delta_factor",mustHaveValue,NULL,multiValueClass},
  {"delta",mustHaveValue,NULL,multiValueClass},
  {"error",mustHaveValue,NULL,multiValueClass},
  {"alpha",mustHaveValue,NULL,multiValueClass},
  {"beta",mustHaveValue,NULL,multiValueClass},
  {"top_nodes",mustHaveValue,NULL,multiValueClass},
  {"device_list",mustHaveValue,NULL,multiValueClass}, //NOT SURE IF WE STILL NEED THIS
  {"device",mustHaveValue,NULL,multiValueClass}, //IF ELSE
  //add your new options here above this line
  {0, 0, 0, 0},
};

class OptionValue{
  //this include file_type_setup,set_file_flag,content assign
private:
  vector<string> values;
  int num_iter = 0;
  bool valueDuplicateAllowed = true;

public:
  vector<string> keywords;
  OptionValue(){};

  void optionValueSetup(char* str) {contentParse(str);}
  void contentAssign(const string &name, std::multimap<string,string> &_m_map);
  bool double_check(multimap<string,string> &_m_map,string s_key,string s_value);
  void disallowDuplicate() {valueDuplicateAllowed = false;}
  void contentParse(char *str);
  void fileCheck();
  void clear();

  friend void keyword_check(char *str,char *_name,vector<string> _keywords);
};

class Commandline{
public:
  Commandline(){};
  void char_check(char *str);
  void printout(std::multimap<string,string> _m_map,string s);
  void keyword_check(char *str,char * _name,vector<string> _keywords);
  bool multiValuesCheck(std::multimap<string,string> &_m_map,string s_key,string s_value);

  multimap<string,string> commandlineArgument(int argc, char *argv[]);

};


//for multi value


multimap<string,string> Commandline::commandlineArgument(int argc, char *argv[]){
  int option_index;
  int c;
  char * pch;
  char *l_opt_arg;

  std::multimap<string,string> m_map;
  string s,tmpValue;
  OptionValue valueCheck;

  if (argc == 1){ //print help document
    printf("Help. \n");
    exit(1);
  }

  if(argv[1][0] != '-'){
    printf("Error. You need to put - for the options.\n");
    exit(1);
  }

  while((c = getopt_long_only (argc, argv, "", long_options, &option_index)) != -1){
    switch (c){
      case 0:
      printf("Invalid argument.\n");
      break;
      case zeroValueClass:
      s.assign(long_options[option_index].name);
      m_map.insert(make_pair(s,""));
      s.clear();
      break;
      case oneValueClass:
      s.assign(long_options[option_index].name);
      char_check(optarg);
      tmpValue.assign(optarg);
      if(tmpValue == "1") tmpValue = "true";
      if(tmpValue == "0") tmpValue = "false";
      if(!multiValuesCheck(m_map,s,tmpValue))
        m_map.insert(make_pair(s,tmpValue));
      s.clear();
      tmpValue.clear();
      break;
      case multiValueClass: //such as rgg
      s.assign(long_options[option_index].name);
      valueCheck.optionValueSetup(optarg);
      valueCheck.contentAssign(s,m_map);
      valueCheck.clear();
      s.clear();
      break;
			case marketClass: //market, argument
      s.assign(long_options[option_index].name);
      //valueCheck.keywords.push_back(".mtx");
      //valueCheck.keywords.push_back(".txt");
      valueCheck.optionValueSetup(optarg);
      valueCheck.fileCheck();
      valueCheck.disallowDuplicate();
      valueCheck.contentAssign(s,m_map);
      valueCheck.clear();
      break;
      //add new class here
      default:
      printf("Invalid argument.\n");
      break;
    }
}

  for(auto it=m_map.begin(), end = m_map.end(); it != end; it = m_map.upper_bound(it->first)){
    auto ret = m_map.equal_range(it->first);
    cout << it->first << "=> ";
    for(auto itValue = ret.first; itValue != ret.second; itValue++) cout << itValue->second << ' ';
    cout << endl;
  }
  return m_map;
}

//input map,string
bool Commandline::multiValuesCheck(std::multimap<string,string> &_m_map,string s_key,string s_value){
  multimap<string,string>::iterator it = _m_map.find(s_key);
  if(it != _m_map.end()) {
    if(s_value != it->second){
      printf("Warning: You have already set %s for the key --%s before.\nWarning: --%s is then set to %s.\n",it->second.c_str(),s_key.c_str(),s_key.c_str(),s_value.c_str());
      it->second = s_value;
    }
    return 1;
  }
  return 0;
}

void Commandline::char_check(char *str){
  int char_index=0;
  while (str[char_index]!='\0') {
    if(!std::isalnum(str[char_index])){
      printf("Value %s contains non-digit or non-letter characters.\n",str);
      exit(EXIT_FAILURE);
    }
    char_index++;
  }
}

void OptionValue::contentParse(char *str){ //check bracket and file/values
  int n = strlen(str);
  char *pch;

  pch = strtok (str," ,[]");
  while (pch != NULL){
    string s(pch);
    values.push_back(s);
    num_iter++;
    pch = strtok (NULL, " ,[]");
  }
}

void OptionValue::fileCheck(){
  for(string str:values){
    ifstream fp(str);
    if(str.empty() || !fp.is_open()){
      fprintf(stderr, "Input graph file %s does not exist.\n",str.c_str());
      exit (EXIT_FAILURE);
    }
    fp.close();
  }
}

bool OptionValue::double_check(multimap<string,string> &_m_map,string s_key,string s_value){
  // check if the key has duplicate values
  auto it = _m_map.find(s_key);
  if(it != _m_map.end()){
    auto ret = _m_map.equal_range(s_key);
    for(auto itValues = ret.first; itValues != ret.second; itValues++){
      if(s_value == itValues->second){
        if(valueDuplicateAllowed == false)
          printf("Duplicate value = %s for key = %s is not allowed. \n",itValues->second.c_str(), s_key.c_str());
          return true;
      }
    }
  }
  return false;
}

void OptionValue::contentAssign(const string &name, std::multimap<string,string> &_m_map){
  for(int i = 0; i < num_iter; i++){
    if(valueDuplicateAllowed == true || !double_check(_m_map,name,values[i]))
      _m_map.insert(make_pair(name,values[i]));
    else exit(1);
  }
}

void OptionValue::clear(){
  values.clear();
  num_iter = 0;
  keywords.clear();
  valueDuplicateAllowed = true;
}

//////////////////////////////// value,file checks checks

/*
void char_check_withcomma(char *str){
    int char_index=0;
    while (str[char_index]!='\0') {
      if(!std::isalnum(str[char_index])){
      printf("Value %s contains non-digit or non-letter characters.\n",str);
      exit(EXIT_FAILURE);
      }
      char_index++;
    }
}

void printout(std::multimap<string,string> _m_map,string s){
  multimap<string,string>::iterator beg,end;
  multimap<string,string>::iterator m;
  m = _m_map.find(s);
  beg = _m_map.lower_bound(s);
  end = _m_map.upper_bound(s);
  //int i = 0;
  for(m = beg;m != end;m++){
  cout<<m->first<<"--"<< m-> second <<endl;
  //cout << ++i << endl;
  }
}



void keyword_check(char *str,char * _name,vector<string> _keywords){
  if(!_keywords.empty()){
    //cout << _keywords[0] << endl;
    int i;
    string s(str);
    for(i = 0; i < _keywords.size(); i++){
      size_t found = s.find(_keywords[i]);
      if(found!=std::string::npos){
        break;
      }
    }
    if(i == _keywords.size()){
      printf("Your value %s does not match the keywords for key %s.\n",str,_name);
      exit(1);
    }
  }
}
*/
