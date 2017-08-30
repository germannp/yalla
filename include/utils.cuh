// Snippets used in several places
#pragma once

#include <sstream>
#include <string>
#include <vector>


std::vector<std::string> split(const std::string &s) {
    std::stringstream ss(s);
    std::string word;
    std::vector<std::string> words;
    while (std::getline(ss, word, ' ')) {
        words.push_back(word);
    }
    return words;
}
