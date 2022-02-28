#ifndef _UTIL_H
#define _UTIL_H

#include <dirent.h>
#include <vector>
#include <string>
#include <string.h>

static inline int read_image(const char *p_dir_name, std::vector<std::string> &files){
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr){
        return -1;
    }
    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr){
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            if (p_file->d_type != DT_DIR) {
                std::string file_name = p_file->d_name;
                int iPos = file_name.find_last_of(".");
                if (iPos != std::string::npos && 
                    (strcmp(".jpg", file_name.substr(iPos).c_str()) == 0)) {
                    files.push_back(std::string(p_dir_name) +"/"+ file_name);
                }
            }
        }
    }
    closedir(p_dir);
    return 0;
}

#endif // def _UTIL_H