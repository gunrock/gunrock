#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

class SummaryItem {
 public:
  string name;
  int num_search_tokens, num_record_tokens, offset;
  string* search_tokens;
  string* record_tokens;

  SummaryItem()
      : name(""),
        num_search_tokens(0),
        num_record_tokens(0),
        offset(0),
        search_tokens(NULL),
        record_tokens(NULL) {}

  void Init(const char name[], int num_search_tokens, string* search_tokens,
            int offset, int num_record_tokens) {
    this->name = string(name);
    this->num_search_tokens = num_search_tokens;
    this->num_record_tokens = num_record_tokens;
    if (num_search_tokens != 0) {
      this->search_tokens = new string[num_search_tokens];
      for (int i = 0; i < num_search_tokens; i++)
        this->search_tokens[i] = search_tokens[i];
    }
    this->record_tokens =
        new string[num_record_tokens == 0 ? 1 : num_record_tokens];
    this->offset = offset;
    Reset();
  }

  ~SummaryItem() {
    if (search_tokens != NULL) {
      delete[] search_tokens;
      search_tokens = NULL;
    }
    if (record_tokens != NULL) {
      delete[] record_tokens;
      record_tokens = NULL;
    }
  }

  void Reset() {
    for (int i = 0; i < num_record_tokens; i++) record_tokens[i] = "";
    if (num_record_tokens == 0) record_tokens[0] = "false_";
  }
};

bool isSeparator(char ch, int num_separators, char* separators) {
  for (int i = 0; i < num_separators; i++)
    if (ch == separators[i]) return true;
  return false;
}

string GetNextToken(ifstream* fin, int num_separators, char* separators) {
  string str = "";

  char ch = ' ';
  fin->get(ch);
  while (isSeparator(ch, num_separators, separators) && !fin->eof())
    fin->get(ch);

  while (!isSeparator(ch, num_separators, separators) && !fin->eof()) {
    str = str + ch;
    fin->get(ch);
  }
  return str;
}

bool TokensMatch(string* current_tokens, int offset, string* search_tokens,
                 int num_search_tokens) {
  if (num_search_tokens == 0) return false;
  for (int i = 0; i < num_search_tokens; i++)
    if (search_tokens[i] != current_tokens[offset + i]) return false;
  return true;
}

int GetSubstringPos(string org_str, string sub_str) {
  for (int i = 0; i < (signed)(org_str.length() - sub_str.length() + 1); i++) {
    bool is_same = true;
    for (unsigned int j = 0; j < sub_str.length(); j++)
      if (org_str[i + j] != sub_str[j]) {
        is_same = false;
        break;
      }
    if (is_same) return i;
  }
  return -1;
}

int main(int argc, char* argv[]) {
  DIR* directory;
  struct dirent* dir_item;
  struct stat filestat;
  int num_items = 27;
  SummaryItem* items = new SummaryItem[27];
  string search_tokens[5];
  string* current_tokens = NULL;
  char seperators[] = {' ', '\0', '\13', '\t', '\n', ':',
                       '[', ']',  ',',   ')',  '('};
  int num_seperators = 11;
  int max_length = 0;
  int num_tokens = 0;
  string** token_lists = new string*[1024];
  for (int i = 0; i < 1024; i++) token_lists[i] = NULL;
  ofstream fout;

  items[0].Init("test_name", 0, search_tokens, 0, 1);
  items[1].Init("flags", 0, search_tokens, 0, 1);
  search_tokens[0] = "Degree";
  search_tokens[1] = "Histogram";
  items[2].Init("#nodes", 2, search_tokens, 2, 1);
  items[3].Init("#edges", 2, search_tokens, 4, 1);
  search_tokens[0] = "CPU";
  search_tokens[1] = "PR";
  search_tokens[2] = " ";
  items[4].Init("CPU depth", 3, search_tokens, 9, 1);
  items[5].Init("CPU time", 2, search_tokens, 4, 1);
  search_tokens[0] = "GPU";
  search_tokens[1] = "PageRank";
  search_tokens[2] = " ";
  items[6].Init("GPU depth", 3, search_tokens, 10, 1);
  items[7].Init("GPU time", 2, search_tokens, 4, 1);
  items[8].Init("GPU rate", 3, search_tokens, 7, 1);
  search_tokens[0] = "Validity";
  search_tokens[1] = "Rank";
  items[9].Init("Rank v", 2, search_tokens, 2, 1);
  search_tokens[0] = "oversize";
  items[10].Init("Oversize", 1, search_tokens, 0, 0);
  search_tokens[0] = "Total";
  search_tokens[1] = "rank";
  items[11].Init("total rank", 2, search_tokens, 2, 1);
  search_tokens[0] = "#edges";
  search_tokens[1] = "visited";
  items[12].Init("#e visited", 2, search_tokens, 2, 1);
  search_tokens[0] = "10";
  search_tokens[1] = "Page";
  items[13].Init("Top Rank ID", 2, search_tokens, 5, 1);
  items[14].Init("Top Rank", 2, search_tokens, 8, 1);

  search_tokens[0] = "GPU_0";
  items[15].Init("GPU_0 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_1";
  items[16].Init("GPU_1 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_2";
  items[17].Init("GPU_2 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_3";
  items[18].Init("GPU_3 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_4";
  items[19].Init("GPU_4 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_5";
  items[20].Init("GPU_5 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_6";
  items[21].Init("GPU_6 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = "GPU_7";
  items[22].Init("GPU_7 mem", 1, search_tokens, 1, 1);
  search_tokens[0] = " ";
  items[23].Init("total mem", 1, search_tokens, 1, 1);

  search_tokens[0] = "queue_sizing";
  items[24].Init("q-sizing", 1, search_tokens, 2, 2);
  search_tokens[0] = "in_sizing";
  items[25].Init("in-sizing", 1, search_tokens, 2, 1);
  search_tokens[0] = "partition";
  search_tokens[1] = "end.";
  items[26].Init("partition time", 1, search_tokens, 2, 1);

  for (int i = 0; i < num_items; i++) {
    int length = 0;
    if (items[i].offset <= 0) {
      length = items[i].num_search_tokens - items[i].offset +
               items[i].num_record_tokens;
    } else {
      length = items[i].num_search_tokens + items[i].offset +
               items[i].num_record_tokens;
    }
    if (length > max_length) max_length = length;

    num_tokens += items[i].num_record_tokens;
    if (items[i].num_record_tokens == 0) num_tokens++;
  }
  current_tokens = new string[max_length];

  if (argc < 2) {
    cout << "Please give the directory name" << endl;
    return -1;
  }

  directory = opendir(argv[1]);
  if (directory == NULL) {
    cout << "Directory " << argv[1] << " can't be opened" << endl;
    return -1;
  }

  for (int i = 0; i < num_items; i++) items[i].Reset();
  string dir_name = string(argv[1]);
  cout << dir_name << "\t";
  fout.open(string(dir_name + ".txt").c_str());
  for (int i = 0; i < num_items; i++) {
    fout << items[i].name << "\t";
    for (int j = 1; j < items[i].num_record_tokens; j++) fout << "\t";
  }
  fout << endl;

  int num_data = 0;
  while ((dir_item = readdir(directory))) {
    string file_name = string(dir_item->d_name);
    string path_name = dir_name + "/" + file_name;
    if (stat(path_name.c_str(), &filestat)) continue;
    if (S_ISDIR(filestat.st_mode)) continue;

    if (file_name[0] == '.') continue;
    ifstream fin;
    fin.open(path_name.c_str());
    if (!fin.is_open()) {
      cout << path_name << " can't be opened" << endl;
      continue;
    }

    for (int i = 0; i < num_items; i++) items[i].Reset();
    int pos = GetSubstringPos(file_name, dir_name);
    string data_name = "", flag = "";
    if (pos >= 0) {
      for (int i = 0; i < pos - 1; i++) data_name = data_name + file_name[i];
      for (unsigned int i = pos + dir_name.length() + 1;
           i < file_name.length() - 4; i++)
        flag = flag + file_name[i];
    }
    // cout<<"new data: "<<data_name<<" "<<flag<<endl;
    items[0].record_tokens[0] = data_name;
    items[1].record_tokens[0] = flag;
    for (int i = 0; i < max_length; i++) current_tokens[i] = "";
    int last_count = 0;
    while (!fin.eof() || last_count < max_length) {
      if (fin.eof()) last_count++;
      string next_token =
          fin.eof() ? "" : GetNextToken(&fin, num_seperators, seperators);
      for (int i = 0; i < max_length - 1; i++)
        current_tokens[i] = current_tokens[i + 1];
      current_tokens[max_length - 1] = next_token;
      // cout<<next_token<<endl;

      for (int i = 0; i < num_items; i++) {
        bool hit = TokensMatch(
            current_tokens, items[i].offset < 0 ? (-items[i].offset) : 0,
            items[i].search_tokens, items[i].num_search_tokens);
        int num_r_tokens = items[i].num_record_tokens;
        if (!hit) continue;
        for (int j = 0; j < num_r_tokens; j++) {
          items[i].record_tokens[j] = items[i].offset < 0
                                          ? current_tokens[j]
                                          : current_tokens[items[i].offset + j];
          // cout<<items[i].record_tokens[j]<<endl;
        }
        if (num_r_tokens == 0) items[i].record_tokens[0] = "true_";
        // cout<<items[i].name<<" hit : ";
        // for (int j=0; j< (num_r_tokens == 0? 1: num_r_tokens); j++)
        //    cout<<items[i].record_tokens[j]<<" ";
        // cout<<endl;
        // for (int j=0; j<max_length; j++)
        //   cout<<current_tokens[j]<<" ";
        // cout<<endl;
      }
    }
    fin.close();

    long long total_mem = 0;
    for (int i = 1; i <= 8; i++)
      total_mem += atoll(items[i + 14].record_tokens[0].c_str());
    items[23].record_tokens[0] = std::to_string(total_mem);

    token_lists[num_data] = new string[num_tokens];
    int temp_counter = 0;
    for (int i = 0; i < num_items; i++)
      for (int j = 0;
           j <
           (items[i].num_record_tokens == 0 ? 1 : items[i].num_record_tokens);
           j++) {
        // fout<<items[i].record_tokens[j]<<"\t";
        token_lists[num_data][temp_counter] = items[i].record_tokens[j];
        temp_counter++;
      }
    // fout<<endl;
    num_data++;
  };
  cout << "num_data = " << num_data << endl;

  for (int i = 0; i < num_data - 1; i++)
    for (int j = i + 1; j < num_data; j++)
      if (token_lists[i][0] > token_lists[j][0] ||
          (token_lists[i][0] == token_lists[j][0] &&
           token_lists[i][1] > token_lists[j][1])) {
        for (int k = 0; k < num_tokens; k++) {
          string temp_str = token_lists[i][k];
          token_lists[i][k] = token_lists[j][k];
          token_lists[j][k] = temp_str;
        }
      }

  for (int i = 0; i < num_data; i++) {
    for (int j = 0; j < num_tokens - 1; j++) fout << token_lists[i][j] << "\t";
    fout << token_lists[i][num_tokens - 1] << endl;
  }

  fout.close();
  closedir(directory);
  return 0;
}
