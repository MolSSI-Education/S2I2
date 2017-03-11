/**********************************************************************
  rng.cpp - returns a std::vector<int> filled with random numbers. The size,
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

#include <random>

#include "rng.h"

std::vector<int> rng::generateRandomIntVector(int size, int low, int high)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::uintmax_t> dis(low, high);
  std::vector<int> list;
  for (size_t i = 0; i < size; i++) {
    list.push_back(dis(gen));
  }
  return list;
}

