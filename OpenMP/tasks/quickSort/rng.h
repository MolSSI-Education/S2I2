/**********************************************************************
  rng.h - returns a std::vector<int> filled with random numbers. The size,
  lowest number to be generated, and highest number to be generated are
  provided by the client.

  Copyright (C) 2015 by Patrick Avery
  This source code is released under the New BSD License, (the "License").
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
***********************************************************************/


#ifndef RANDOM_H
#define RANDOM_H

#include <vector>

class rng
{
 public:
  static std::vector<int> generateRandomIntVector(int size, int low,
                                                  int high);
};


#endif
