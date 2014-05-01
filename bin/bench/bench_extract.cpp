#include <iostream>
#include <string>
#include <limits>
#include <algorithm>

int main(int argc, char** argv)
{
  size_t extracted = 3;
  if (argc > 1)
  {
    std::string opt(argv[1]);
    if      (opt == "-constants") extracted = 0;
    else if (opt == "-state-counts") extracted = 1;
    else if (opt == "-trans-counts") extracted = 2;
    else if (opt == "-times") extracted = 3;
  }

  std::string trash;
  std::getline(std::cin, trash); // Global options

  std::string model;
  while (std::getline(std::cin, model))
  {
    std::cout << model << std::endl;
    
    size_t prop_count = 0;
    size_t const_count = 0;
    std::cin >> prop_count >> const_count >> std::ws;

    std::string property;
    for (size_t pi = 0; pi < prop_count; ++pi)
    {
      std::getline(std::cin, property);
      std::cout << property << std::endl;
      
      for (size_t ci = 0; ci < const_count; ++ci)
      {
        for (size_t i = 0; i < 4; ++i)
        {
          std::getline(std::cin, trash);
          if (i == extracted) std::cout << trash << std::endl;
        }
      }
    }
  }
}
