//testing how to write the variable type onto a string

#include <typeinfo>
#include <string>
#include <iostream>

int main()
{
  int a = 0;
  float b = 0.0f;
   //std::cout << type_name<decltype(a)>() << '\n';
   //std::cout << type_name<decltype(b)>() << '\n';

  std::cout << typeid(a).name() << '\n'; //prints "i"
  std::cout << typeid(b).name() << '\n'; //prints "f"

  return 0;

}
