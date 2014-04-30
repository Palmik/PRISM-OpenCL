#include <iostream>
#include <string>
#include <limits>
#include <algorithm>

int main()
{
  size_t trials = 0;
  std::cin >> trials >> std::ws;
  
  std::string options;
  std::getline(std::cin, options);
  std::cout << options << std::endl;

  std::string model;
  while (std::getline(std::cin, model))
  {
    std::cout << model << std::endl;
    
    size_t prop_count = 0;
    size_t const_count = 0;
    std::cin >> prop_count >> const_count >> std::ws;
    std::cout << prop_count << " " << const_count << std::endl;

    std::string property;
    for (size_t pi = 0; pi < prop_count; ++pi)
    {
      std::getline(std::cin, property);
      std::cout << property << std::endl;
      
      std::string constants;
      for (size_t ci = 0; ci < const_count; ++ci)
      {
        std::getline(std::cin, constants);
        std::cout << constants << std::endl;
        
        size_t model_states = 0;
        size_t model_transitions = 0;
        std::cin >> model_states >> model_transitions >> std::ws;
        std::cout << model_states << std::endl
                  << model_transitions << std::endl;
        
        double avg = 0;
        for (size_t i = 0; i < trials; ++i)
        {
          double t = 0;
          std::cin >> t;
          avg += t;
        }
        avg /= trials;

        std::cin >> std::ws;

        std::cout << avg << std::endl;
      } 
    }
  }
}


